use std::path::Path;

use burn::{
    data::{dataloader::DataLoaderBuilder, dataset::Dataset},
    module::AutodiffModule,
    nn::loss::MseLoss,
    optim::{AdamConfig, GradientsParams, Optimizer},
    prelude::*,
    record::CompactRecorder,
    tensor::{backend::AutodiffBackend, Tensor, Transaction},
    train::{
        metric::{Adaptor, ItemLazy, LossInput, LossMetric}, LearnerBuilder, TrainOutput, TrainStep, ValidStep
    },
};

use indicatif::{MultiProgress, ProgressBar, ProgressStyle};

use crate::sketchy_database::{
    sketchy_batcher::{SketchyBatch, SketchyBatcher},
    sketchy_dataset::{PhotoAugmentation, SketchAugmentation, SketchyDataset},
};

use super::{
    discriminator::{Pix2PixDescriminatorConfig, Pix2PixDiscriminator},
    generator::{Pix2PixGenerator, Pix2PixGeneratorConfig},
};

#[derive(Config, Debug)]
pub struct Pix2PixModelConfig {
    discriminator_config: Pix2PixDescriminatorConfig,
    generator_config: Pix2PixGeneratorConfig,
}

impl Pix2PixModelConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> Pix2PixModel<B> {
        Pix2PixModel {
            discriminator: self.discriminator_config.init(device),
            generator: self.generator_config.init(device),
            mse_loss: MseLoss::new(),
        }
    }
}

#[derive(Module, Debug)]
pub struct Pix2PixModel<B: Backend> {
    discriminator: Pix2PixDiscriminator<B>,
    generator: Pix2PixGenerator<B>,
    mse_loss: MseLoss,
}

#[derive(Debug)]
pub struct GanOutput<B: Backend> {
    /// real sketches
    /// dim [N, C, 256, 256]
    train_sketches: Tensor<B, 4>,
    /// generated sketches
    /// dim [N, C, 256, 256]
    fake_sketches: Tensor<B, 4>,
    /// discriminator result of real sketches
    /// dim [N, 1, 16, 16]
    real_sketch_output: Tensor<B, 4>,
    /// discriminator result of fake sketches
    /// dim [N, 1, 16, 16]
    fake_sketch_output: Tensor<B, 4>,
    /// loss of discriminator
    loss_discriminator: Tensor<B, 1>,
    /// loss of generator
    loss_generator: Tensor<B, 1>,
}

impl<B: Backend> ItemLazy for GanOutput<B> {
    type ItemSync = GanOutput<B>;

    fn sync(self) -> Self::ItemSync {
        let [
            train_sketches,
            fake_sketches,
            real_sketch_output,
            fake_sketch_output,
            loss_discriminator,
            loss_generator,
        ] = Transaction::default()
            .register(self.train_sketches)
            .register(self.fake_sketches)
            .register(self.real_sketch_output)
            .register(self.fake_sketch_output)
            .register(self.loss_discriminator)
            .register(self.loss_generator)
            .execute()
            .try_into()
            .expect("Correct amount of tensor data");

        let device = &Default::default();

        GanOutput {
            train_sketches: Tensor::from_data(train_sketches, device),
            fake_sketches: Tensor::from_data(fake_sketches, device),
            real_sketch_output: Tensor::from_data(real_sketch_output, device),
            fake_sketch_output: Tensor::from_data(fake_sketch_output, device),
            loss_discriminator: Tensor::from_data(loss_discriminator, device),
            loss_generator: Tensor::from_data(loss_generator, device),
        }
    }
}

impl<B: Backend> Pix2PixModel<B> {
    pub fn forward_training(&self, item: SketchyBatch<B>) -> GanOutput<B> {
        let generated_sketches = self.generator.forward(item.photos.clone());

        let real_result = self
            .discriminator
            .forward(item.photos.clone(), item.sketches.clone());
        let fake_result = self
            .discriminator
            .forward(generated_sketches.clone(), item.sketches.clone());
        // Erstelle Ziel-Tensoren: echte Bilder sollen als 1 klassifiziert werden, gefälschte als 0
        let real_labels = Tensor::ones_like(&real_result);
        let fake_labels = Tensor::zeros_like(&fake_result);
        let loss_real = self.mse_loss.forward(
            real_result.clone(),
            real_labels.clone(),
            nn::loss::Reduction::Auto,
        );
        let loss_fake = self.mse_loss.forward(
            fake_result.clone(),
            fake_labels.clone(),
            nn::loss::Reduction::Auto,
        );
        let output = GanOutput {
            train_sketches: item.sketches,
            fake_sketches: generated_sketches,
            real_sketch_output: real_result,
            fake_sketch_output: fake_result,
            loss_discriminator: loss_real.clone(),
            loss_generator: (loss_real.detach() + loss_fake) / 2.0,
        };
        output
    }
}

impl<B: AutodiffBackend> TrainStep<SketchyBatch<B>, GanOutput<B>> for Pix2PixModel<B> {
    fn step(&self, item: SketchyBatch<B>) -> TrainOutput<GanOutput<B>> {
        let output = self.forward_training(item);

        TrainOutput::new(self, output.loss_generator.backward(), output)
    }
}

impl<B: Backend> ValidStep<SketchyBatch<B>, GanOutput<B>> for Pix2PixModel<B> {
    fn step(&self, item: SketchyBatch<B>) -> GanOutput<B> {
        self.forward_training(item)
    }
}

impl<B: Backend> Adaptor<LossInput<B>> for GanOutput<B> {
    fn adapt(&self) -> LossInput<B> {
        LossInput::new(self.loss_discriminator.clone())
    }
}

#[derive(Config)]
pub struct TrainingConfig {
    pub model: Pix2PixModelConfig,
    pub optimizer_discriminator: AdamConfig,
    pub optimizer_generator: AdamConfig,
    #[config(default = 10)]
    pub num_epochs: usize,
    #[config(default = 4)]
    pub batch_size: usize,
    #[config(default = 10)]
    pub num_workers: usize,
    #[config(default = 42)]
    pub seed: u64,
    #[config(default = 0.0002)]
    pub learning_rate: f64,
    #[config(default = 2)]
    pub mini_batch_discriminator: usize,
    #[config(default = 2)]
    pub mini_batch_generator: usize,
}

fn create_artifact_dir(artifact_dir: &str) {
    // Remove existing artifacts before to get an accurate learner summary
    std::fs::remove_dir_all(artifact_dir).ok();
    std::fs::create_dir_all(artifact_dir).ok();
}

pub fn run_trainer<B: AutodiffBackend>(
    artifact_dir: &str,
    config: TrainingConfig,
    device: B::Device,
) {
    create_artifact_dir(artifact_dir);
    config
        .save(format!("{artifact_dir}/config.json"))
        .expect("Config should be saved successfully");

    B::seed(config.seed);

    let batcher_train = SketchyBatcher::<B>::new(device.clone());
    let batcher_valid = SketchyBatcher::<B::InnerBackend>::new(device.clone());

    let photo_path = Path::new("./data/sketchydb_256x256/256x256/photo/");

    let sketch_path = Path::new("./data/sketchydb_256x256/256x256/sketch/");

    let data = SketchyDataset::new(
        photo_path,
        sketch_path,
        PhotoAugmentation::ScaledAndCentered,
        SketchAugmentation::ScaledAndCentered,
    )
    .expect("");

    let (train, valid) = data.split(0.8);

    let dataloader_train = DataLoaderBuilder::new(batcher_train)
        .batch_size(config.batch_size)
        .shuffle(config.seed)
        .num_workers(config.num_workers)
        .build(train);

    let dataloader_test = DataLoaderBuilder::new(batcher_valid)
        .batch_size(config.batch_size)
        .shuffle(config.seed)
        .num_workers(config.num_workers)
        .build(valid);

    let learner = LearnerBuilder::new(artifact_dir)
        // .metric_train_numeric(AccuracyMetric::new())
        // .metric_valid_numeric(AccuracyMetric::new())
        .metric_train_numeric(LossMetric::new())
        .metric_valid_numeric(LossMetric::new())
        .with_file_checkpointer(CompactRecorder::new())
        .devices(vec![device.clone()])
        .num_epochs(config.num_epochs)
        .summary()
        .build(
            config.model.init::<B>(&device),
            config.optimizer_discriminator.init(),
            config.learning_rate,
        );

    let model_trained = learner.fit(dataloader_train, dataloader_test);

    model_trained
        .save_file(format!("{artifact_dir}/model"), &CompactRecorder::new())
        .expect("Trained model should be saved successfully");
}

pub fn run_custom_loop<B: AutodiffBackend>(
    artifact_dir: &str,
    config: TrainingConfig,
    device: B::Device,
) {
    create_artifact_dir(artifact_dir);
    config
        .save(format!("{artifact_dir}/config.json"))
        .expect("Config should be saved successfully");

    B::seed(config.seed);

    let batcher_train = SketchyBatcher::<B>::new(device.clone());
    let batcher_valid = SketchyBatcher::<B::InnerBackend>::new(device.clone());

    let photo_path = Path::new("./data/sketchydb_256x256/256x256/photo/");

    let sketch_path = Path::new("./data/sketchydb_256x256/256x256/sketch/");

    let mut model = config.model.init::<B>(&device);

    let data = SketchyDataset::new(
        photo_path,
        sketch_path,
        PhotoAugmentation::ScaledAndCentered,
        SketchAugmentation::ScaledAndCentered,
    )
    .expect("");

    let (train, valid) = data.split(0.8);

    let train_size = train.len();
    let valid_size = valid.len();

    let mut opt_discriminator = config.optimizer_discriminator.init();
    let mut opt_generator = config.optimizer_generator.init();

    let dataloader_train = DataLoaderBuilder::new(batcher_train)
        .batch_size(config.batch_size)
        .shuffle(config.seed)
        .num_workers(config.num_workers)
        .build(train);

    let dataloader_test = DataLoaderBuilder::new(batcher_valid)
        .batch_size(config.batch_size)
        .shuffle(config.seed)
        .num_workers(config.num_workers)
        .build(valid);

    let m = MultiProgress::new();
    let sty = ProgressStyle::with_template(
        "[{elapsed_precise}] {bar:40.cyan/blue} {pos:>7}/{len:7} {msg}",
    )
    .unwrap()
    .progress_chars("##-");
    let epoch_bar = m.add(ProgressBar::new(config.num_epochs as u64));
    epoch_bar.set_style(sty.clone());
    epoch_bar.set_message("Epochs");
    
    
    // Iterate over our training and validation loop for X epochs.
    for _epoch in 1..config.num_epochs + 1 {
        epoch_bar.inc(1);

        let training_bar = m.add(ProgressBar::new(train_size as u64));
        training_bar.set_style(sty.clone());
        training_bar.set_message("Training:");


        // Implement our training loop.
        for (_iteration, batch) in dataloader_train.iter().enumerate() {
            let output = model.forward_training(batch);

            let grad_d = output.loss_discriminator.clone().backward();
            let grad_d = GradientsParams::from_grads(grad_d, &model.discriminator);
            let grad_g = output.loss_generator.clone().backward();
            let grad_g = GradientsParams::from_grads(grad_g, &model.generator);

            model.discriminator =
                opt_discriminator.step(config.learning_rate, model.discriminator, grad_d);
            model.generator = opt_generator.step(config.learning_rate, model.generator, grad_g);
            training_bar.set_message(format!("Training - gen loss: {}, dis loss: {}",  output.loss_generator.mean().into_scalar(), output.loss_discriminator.mean().into_scalar()));
            training_bar.inc(config.batch_size as u64);
        }
        m.remove(&training_bar);
        // Get the model without autodiff.
        let model_valid = model.valid();
        
        let valid_bar = m.add(ProgressBar::new(valid_size as u64));
        valid_bar.set_style(sty.clone());
        valid_bar.set_message("Valid Progress");
        // Implement our validation loop.
        for (_iteration, batch) in dataloader_test.iter().enumerate() {
            let output = model_valid.forward_training(batch);
            valid_bar.set_message(format!("Valid - gen loss: {}, dis loss: {}",  output.loss_generator.mean().into_scalar(), output.loss_discriminator.mean().into_scalar()));
            valid_bar.inc(config.batch_size as u64);
        }
        m.remove(&valid_bar);
    }
}
