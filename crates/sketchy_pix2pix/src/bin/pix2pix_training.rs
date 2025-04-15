use std::{path::PathBuf, str::FromStr};

use burn::{
    backend::{Autodiff, Wgpu},
    optim::AdamConfig,
    record::CompactRecorder,
};
use sketchy_pix2pix::{
    logging::SketchyGanLogger,
    pix2pix::{
        discriminator::Pix2PixDescriminatorConfig,
        gan::Pix2PixModelConfig,
        generator::Pix2PixGeneratorConfig,
        training::{train_gan, Pix2PixModelProvider, TrainDataConfig, TrainingConfig},
    }, sketchy_database::sketchy_dataset::SketchyClass,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    type MyBackend = Wgpu<f32, i32>;
    type MyAutodiffBackend = Autodiff<MyBackend>;

    let device = burn::backend::wgpu::WgpuDevice::default();
    let artifact_dir = "./tmp";
    let stream =rerun::RecordingStreamBuilder::new("train sketchy gan").spawn()?;
    let rec =
        SketchyGanLogger::new(stream.clone());

    rerun::Logger::new(stream) // recording streams are ref-counted
        .with_path_prefix("logs")
        .with_filter(rerun::default_log_filter())
        .init()?;

    let base_path = PathBuf::from_str("./data/sketchydb_256x256/256x256")?;

    let photo_path = base_path.join("photo");
    let sketch_path = base_path.join("sketch");

    train_gan::<MyAutodiffBackend, _>(
        artifact_dir,
        TrainingConfig::new(
            AdamConfig::new().with_beta_1(0.5),
            AdamConfig::new().with_beta_1(0.5),
            TrainDataConfig::PhotoDirectoryPath { 
                    sketch_path,
                    photo_path,
                    photo_augmentation: sketchy_pix2pix::sketchy_database::sketchy_dataset::PhotoAugmentation::ScaledAndCentered, 
                    sketch_augmentation: sketchy_pix2pix::sketchy_database::sketchy_dataset::SketchAugmentation::ScaledAndCentered, 
                    filter: Some(vec![SketchyClass::Penguin]), 
                    train_ratio: 0.8, 
                },
        ),
        Pix2PixModelProvider::Config(
            Pix2PixModelConfig::new(
                Pix2PixDescriminatorConfig::new(),
                Pix2PixGeneratorConfig::new(),
        )),
        device,
        rec,
        CompactRecorder::new(),
    )?;
    Ok(())
}
