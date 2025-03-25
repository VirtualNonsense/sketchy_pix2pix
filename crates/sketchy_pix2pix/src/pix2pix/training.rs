use std::path::PathBuf;

use burn::{
    config::{Config, ConfigError},
    data::{dataloader::DataLoaderBuilder, dataset::Dataset},
    module::{AutodiffModule, Module},
    optim::{AdamConfig, GradientsParams, Optimizer},
    record::{FileRecorder, RecorderError},
    tensor::backend::AutodiffBackend,
};

use indicatif::{MultiProgress, ProgressBar, ProgressStyle};
use serde::{Deserialize, Serialize};

use crate::{
    logging::SketchyGanLogger,
    sketchy_database::{
        sketchy_batcher::SketchyBatcher,
        sketchy_dataset::{
            PhotoAugmentation, SketchAugmentation, SketchyClass, SketchyDataset,
            SketchyDatasetError,
        },
    },
};

use super::{
    discriminator::Pix2PixDiscriminatorGradients,
    gan::{Pix2PixModel, Pix2PixModelConfig},
    generator::Pix2PixGeneratorGradients,
};
use thiserror::Error;

#[derive(Serialize, Deserialize, Clone)]
pub enum Pix2PixModelProvider {
    Config(Pix2PixModelConfig),
    Checkpoint {
        config_path: PathBuf,
        checkpoint_path: PathBuf,
    },
}
#[derive(Serialize, Deserialize, Clone)]
pub enum TrainDataConfig {
    /// Path to the photo directory
    PhotoDirectoryPath {
        sketch_path: PathBuf,
        photo_path: PathBuf,
        photo_augmentation: PhotoAugmentation,
        sketch_augmentation: SketchAugmentation,
        filter: Option<Vec<SketchyClass>>,
        train_ratio: f64,
    },
    /// Path to the training data and validation data
    ConfigPath {
        train_data: PathBuf,
        valid_data: PathBuf,
    },
}

#[derive(Config)]
pub struct TrainingConfig {
    pub optimizer_discriminator: AdamConfig,
    pub optimizer_generator: AdamConfig,
    #[config(default = 5000)]
    pub num_epochs: usize,
    #[config(default = 1)]
    pub batch_size: usize,
    #[config(default = 10)]
    pub num_workers: usize,
    #[config(default = 42)]
    pub seed: u64,
    #[config(default = 0.002)]
    pub discriminator_learning_rate: f64,
    #[config(default = 0.002)]
    pub generator_learning_rate: f64,
    pub train_data: TrainDataConfig,
}

fn create_artifact_dir(artifact_dir: &str) {
    // Remove existing artifacts before to get an accurate learner summary
    std::fs::remove_dir_all(artifact_dir).ok();
    std::fs::create_dir_all(artifact_dir).ok();
}
#[derive(Debug, Error)]
pub enum TrainingError {
    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),
    #[error("Serialization error: {0}")]
    DatasetError(#[from] SketchyDatasetError),
    #[error("Failed to load Pix2PixModelConfig due to: {0}")]
    ModelConfigDeserializationError(#[from] ConfigError),
    #[error("Failed to load Pix2PixModel weights due to: {0}")]
    ModelWeightsDeserializationError(#[from] RecorderError),
}

pub fn train_gan<B: AutodiffBackend, R: FileRecorder<B>>(
    artifact_dir: &str,
    config: TrainingConfig,
    model_provider: Pix2PixModelProvider,
    device: B::Device,
    log: SketchyGanLogger,
    recorder: R,
) -> Result<Pix2PixModel<B>, TrainingError> {
    create_artifact_dir(artifact_dir);
    config.save(format!("{artifact_dir}/training_config.json"))?;

    B::seed(config.seed);

    let batcher_train = SketchyBatcher::<B>::new(device.clone());
    let batcher_valid = SketchyBatcher::<B::InnerBackend>::new(device.clone());

    let mut model = match model_provider {
        Pix2PixModelProvider::Config(pix_2_pix_model_config) => {
            pix_2_pix_model_config.save(format!("{artifact_dir}/model_config.json"))?;
            pix_2_pix_model_config.init::<B>(&device)
        }
        Pix2PixModelProvider::Checkpoint {
            config_path,
            checkpoint_path,
        } => Pix2PixModelConfig::load(&config_path)?
            .init::<B>(&device)
            .load_file(&checkpoint_path, &recorder, &device)?,
    };
    let (train, valid) = match config.train_data {
        TrainDataConfig::PhotoDirectoryPath {
            sketch_path,
            photo_path,
            photo_augmentation,
            sketch_augmentation,
            filter,
            train_ratio,
        } => {
            let mut data = SketchyDataset::new(
                &photo_path,
                &sketch_path,
                photo_augmentation,
                sketch_augmentation,
            )?;

            if let Some(filter) = filter {
                data = data.filter(|item| filter.contains(&item.sketch_class))
            }
            data.split(train_ratio)
        }
        TrainDataConfig::ConfigPath {
            train_data,
            valid_data,
        } => {
            let train_res = SketchyDataset::load_from_ron(&train_data)?;

            let valid_res = SketchyDataset::load_from_ron(&valid_data)?;
            (train_res, valid_res)
        }
    };

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
    let checkpoint_rotate = 10;

    for epoch in 1..config.num_epochs + 1 {
        epoch_bar.inc(1);

        let training_bar = m.add(ProgressBar::new(train_size as u64));
        training_bar.set_style(sty.clone());
        training_bar.set_message("Training Progress");
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

            log.log_discriminator_gradient(
                Pix2PixDiscriminatorGradients::new(&model.discriminator, &grad_d_param),
                iteration,
            );
            log.log_generator_gradient(
                Pix2PixGeneratorGradients::new(&model.generator, &grad_g_param),
                iteration,
            );

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
        let _save_result = model.clone().save_file(
            format!(
                "{}/checkpoints/model_{:03}",
                &artifact_dir,
                epoch % checkpoint_rotate
            ),
            &recorder,
        );
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
    Ok(model)
}
