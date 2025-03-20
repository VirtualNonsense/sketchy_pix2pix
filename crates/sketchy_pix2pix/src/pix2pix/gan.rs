use std::path::Path;

use burn::{
    data::{dataloader::DataLoaderBuilder, dataset::Dataset},
    module::AutodiffModule,
    nn::loss::{BinaryCrossEntropyLoss, BinaryCrossEntropyLossConfig, MseLoss},
    optim::{AdamConfig, GradientsParams, Optimizer},
    prelude::*,
    tensor::{Tensor, Transaction, backend::AutodiffBackend},
    train::{
        TrainOutput, TrainStep, ValidStep,
        metric::{Adaptor, ItemLazy, LossInput},
    },
};

use indicatif::{MultiProgress, ProgressBar, ProgressStyle};

use crate::{
    sketchy_database::{
        sketchy_batcher::{SketchyBatch, SketchyBatcher},
        sketchy_dataset::{PhotoAugmentation, SketchAugmentation, SketchyClass, SketchyDataset},
    },
    logging::SketchyGanLogger,
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
            bce_loss: BinaryCrossEntropyLossConfig::new().init(device),
        }
    }
}

#[derive(Module, Debug)]
pub struct Pix2PixModel<B: Backend> {
    discriminator: Pix2PixDiscriminator<B>,
    generator: Pix2PixGenerator<B>,
    mse_loss: MseLoss,
    bce_loss: BinaryCrossEntropyLoss<B>,
}

#[derive(Debug)]
pub struct GanOutput<B: Backend> {
    /// real sketches
    /// dim [N, C, 256, 256]
    pub train_sketches: Tensor<B, 4>,
    /// generated sketches
    /// dim [N, C, 256, 256]
    pub fake_sketches: Tensor<B, 4>,
    /// discriminator result of real sketches
    /// dim [N, 1, 16, 16]
    pub real_sketch_output: Tensor<B, 4>,
    /// discriminator result of fake sketches
    /// dim [N, 1, 16, 16]
    pub fake_sketch_output: Tensor<B, 4>,
    /// loss of discriminator
    pub loss_discriminator: Tensor<B, 2>,
    /// loss of generator
    pub loss_generator: Tensor<B, 2>,
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
        let generated_sketches = self.generator.forward(item.photos.clone().detach());

        let real_result = self
            .discriminator
            .forward(item.sketches.clone().detach(), item.photos.clone().detach());
        let fake_result_for_discriminator = self
            .discriminator
            // IMPORTANT: detatch generated sketch from generator.
            .forward(generated_sketches.clone().detach(), item.photos.clone().detach());
        let fake_result_for_generator = self
            .discriminator
            // IMPORTANT the discriminator should not be included in autograd path.
            .clone()
            .no_grad()
            .forward(generated_sketches.clone(), item.photos.clone().detach());
        let loss_d_real = self.mse_loss.forward(
            real_result.clone(),
            Tensor::ones_like(&real_result),
            nn::loss::Reduction::Mean,
        );
        let loss_d_fake = self.mse_loss.forward(
            fake_result_for_discriminator.clone(),
            Tensor::zeros_like(&fake_result_for_discriminator),
            nn::loss::Reduction::Mean,
        );

        let loss_g_fake = self.mse_loss.forward(
            fake_result_for_generator.clone(),
            Tensor::ones_like(&fake_result_for_generator),
            nn::loss::Reduction::Mean,
        );
        let gen_loss = Tensor::cat(vec![loss_g_fake.unsqueeze_dims(&[1, -1])], 0);
        let dis_loss = Tensor::cat(
            vec![
                loss_d_real.unsqueeze_dims(&[1, -1]),
                loss_d_fake.unsqueeze_dims(&[1, -1]),
            ],
            0,
        );
        GanOutput {
            train_sketches: item.sketches,
            fake_sketches: generated_sketches,
            real_sketch_output: real_result,
            fake_sketch_output: fake_result_for_generator,
            loss_discriminator: dis_loss,
            loss_generator: gen_loss,
        }
    }
}

impl<B: AutodiffBackend> TrainStep<SketchyBatch<B>, GanOutput<B>> for Pix2PixModel<B> {
    fn step(&self, item: SketchyBatch<B>) -> TrainOutput<GanOutput<B>> {
        let output = self.forward_training(item);

        TrainOutput::new(
            self,
            output
                .loss_generator
                .clone()
                .mean_dim(0)
                .squeeze::<1>(0)
                .backward(),
            output,
        )
    }
}

impl<B: Backend> ValidStep<SketchyBatch<B>, GanOutput<B>> for Pix2PixModel<B> {
    fn step(&self, item: SketchyBatch<B>) -> GanOutput<B> {
        self.forward_training(item)
    }
}

impl<B: Backend> Adaptor<LossInput<B>> for GanOutput<B> {
    fn adapt(&self) -> LossInput<B> {
        let loss_mean = self.loss_discriminator.clone().mean_dim(0).squeeze(0);
        LossInput::new(loss_mean)
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
    #[config(default = 0.002)]
    pub discriminator_learning_rate: f64,
    #[config(default = 0.002)]
    pub generator_learning_rate: f64,
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


pub fn train_gan<B: AutodiffBackend>(
    artifact_dir: &str,
    config: TrainingConfig,
    device: B::Device,
    log: SketchyGanLogger,
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

    let (train, valid) = data.filter(|item| item.sketch_class == SketchyClass::Penguin).split(0.8);

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

    for _epoch in 1..config.num_epochs + 1 {
        epoch_bar.inc(1);

        let training_bar = m.add(ProgressBar::new(train_size as u64));
        training_bar.set_style(sty.clone());
        training_bar.set_message("Training:");
        // Implement our training loop.
        for (iteration, batch) in dataloader_train.iter().enumerate() {
            let photos = batch.photos.clone();
            let output = model.forward_training(batch);
            let loss_d = output
                .loss_discriminator
                .clone()
                .mean_dim(0)
                .squeeze::<1>(0);
            let loss_g = output.loss_generator.clone().mean_dim(0).squeeze::<1>(0);

            let grad_d = loss_d.clone().backward();
            let grad_g = loss_g.clone().backward();

            let grad_d_param = GradientsParams::from_grads(grad_d, &model.discriminator);
            let grad_g_param = GradientsParams::from_grads(grad_g, &model.generator);

            log.log_progress(photos, output, "training", iteration);

            model.discriminator = opt_discriminator.step(
                config.discriminator_learning_rate,
                model.discriminator,
                grad_d_param,
            );
            model.generator = opt_generator.step(
                config.generator_learning_rate,
                model.generator,
                grad_g_param,
            );
            training_bar.inc(config.batch_size as u64);
        }
        m.remove(&training_bar);
        // Get the model without autodiff.
        let model_valid = model.valid();

        let valid_bar = m.add(ProgressBar::new(valid_size as u64));
        valid_bar.set_style(sty.clone());
        valid_bar.set_message("Valid Progress");
        // Implement our validation loop.
        for (iteration, batch) in dataloader_test.iter().enumerate() {
            let output = model_valid.forward_training(batch.clone());
            
            log.log_progress(batch.photos, output, "validation", iteration);
            valid_bar.inc(config.batch_size as u64);
        }
        m.remove(&valid_bar);
    }
}
