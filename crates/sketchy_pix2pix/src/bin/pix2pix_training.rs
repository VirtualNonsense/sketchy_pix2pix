
use burn::{backend::{Autodiff, Wgpu}, optim::AdamConfig, record::CompactRecorder};
use sketchy_pix2pix::{logging::SketchyGanLogger, pix2pix::{discriminator::Pix2PixDescriminatorConfig, gan::{train_gan, Pix2PixModelConfig, TrainingConfig}, generator::Pix2PixGeneratorConfig}};


fn main() -> Result<(), Box<dyn std::error::Error>>{
    type MyBackend = Wgpu<f32, i32>;
    type MyAutodiffBackend = Autodiff<MyBackend>;

    let device = burn::backend::wgpu::WgpuDevice::default();
    let artifact_dir = "./tmp";

    let rec = SketchyGanLogger::new(rerun::RecordingStreamBuilder::new("train sketchy gan").spawn()?);


    train_gan::<MyAutodiffBackend, _>(
        artifact_dir,
        TrainingConfig::new(Pix2PixModelConfig::new(Pix2PixDescriminatorConfig::new(), Pix2PixGeneratorConfig::new()), 
        AdamConfig::new().with_beta_1(0.5),
        AdamConfig::new().with_beta_1(0.5)),
        device.clone(),
        rec,
        CompactRecorder::new(),
    );
    Ok(())
}