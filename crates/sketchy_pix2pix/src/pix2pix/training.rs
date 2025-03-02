use std::path::Path;

use burn::{
    data::dataloader::DataLoaderBuilder,
    nn::loss::MseLoss,
    optim::AdamConfig,
    prelude::*,
    record::CompactRecorder,
    tensor::{backend::AutodiffBackend, Tensor, Transaction},
    train::{
        metric::{Adaptor, ItemLazy, LossInput, LossMetric}, LearnerBuilder, TrainOutput, TrainStep, ValidStep
    },
};

use crate::sketchy_database::{sketchy_batcher::{SketchyBatch, SketchyBatcher}, sketchy_dataset::{PhotoAugmentation, SketchAugmentation, SketchyDataset}};

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
    real_sketch_output: Tensor<B, 4>,
    fake_sketch_output: Tensor<B, 4>,
    loss_real: Tensor<B, 1>,
    loss_fake: Tensor<B, 1>,
    loss: Tensor<B, 1>,
}

impl<B: Backend> ItemLazy for GanOutput<B> {
    type ItemSync = GanOutput<B>;

    fn sync(self) -> Self::ItemSync {
        let [
            real_sketch_output,
            fake_sketch_output,
            loss_real,
            loss_fake,
            loss,
        ] = Transaction::default()
            .register(self.real_sketch_output)
            .register(self.fake_sketch_output)
            .register(self.loss_real)
            .register(self.loss_fake)
            .register(self.loss)
            .execute()
            .try_into()
            .expect("Correct amount of tensor data");

        let device = &Default::default();

        GanOutput {
            real_sketch_output: Tensor::from_data(real_sketch_output, device),
            fake_sketch_output: Tensor::from_data(fake_sketch_output, device),
            loss_real: Tensor::from_data(loss_real, device),
            loss_fake: Tensor::from_data(loss_fake, device),
            loss: Tensor::from_data(loss, device),
        }
    }
}

impl<B: Backend> Pix2PixModel<B> {
    pub fn forward_training(&self, item: SketchyBatch<B>) -> GanOutput<B> {
        let generated_images = self.generator.forward(item.photos.clone());

        let real_result = self
            .discriminator
            .forward(item.photos.clone(), item.sketches.clone());
        let fake_result = self
            .discriminator
            .forward(generated_images.clone(), item.sketches.clone());
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
            real_sketch_output: real_result,
            fake_sketch_output: fake_result,
            loss_real: loss_real.clone(),
            loss_fake: loss_fake.clone(),
            loss: (loss_real + loss_fake) / 2.0,
        };
        output
    }
}

impl<B: AutodiffBackend> TrainStep<SketchyBatch<B>, GanOutput<B>> for Pix2PixModel<B> {
    fn step(&self, item: SketchyBatch<B>) -> TrainOutput<GanOutput<B>> {
        let output = self.forward_training(item);

        TrainOutput::new(self, output.loss.backward(), output)
    }
}

impl<B: Backend> ValidStep<SketchyBatch<B>, GanOutput<B>> for Pix2PixModel<B> {
    fn step(&self, item: SketchyBatch<B>) -> GanOutput<B> {
        self.forward_training(item)
    }
}

impl<B: Backend> Adaptor<LossInput<B>> for GanOutput<B>{
    fn adapt(&self) -> LossInput<B> {
        LossInput::new(self.loss.clone())
    }
}

#[derive(Config)]
pub struct TrainingConfig {
    pub model: Pix2PixModelConfig,
    pub optimizer: AdamConfig,
    #[config(default = 10)]
    pub num_epochs: usize,
    #[config(default = 16)]
    pub batch_size: usize,
    #[config(default = 4)]
    pub num_workers: usize,
    #[config(default = 42)]
    pub seed: u64,
    #[config(default = 1.0e-4)]
    pub learning_rate: f64,
}

fn create_artifact_dir(artifact_dir: &str) {
    // Remove existing artifacts before to get an accurate learner summary
    std::fs::remove_dir_all(artifact_dir).ok();
    std::fs::create_dir_all(artifact_dir).ok();
}

pub fn train<B: AutodiffBackend>(artifact_dir: &str, config: TrainingConfig, device: B::Device) {
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
    ).expect("");

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
            config.optimizer.init(),
            config.learning_rate,
        );

    let model_trained = learner.fit(dataloader_train, dataloader_test);

    model_trained
        .save_file(format!("{artifact_dir}/model"), &CompactRecorder::new())
        .expect("Trained model should be saved successfully");
}
