
use burn::{backend::{Autodiff, Wgpu}, optim::AdamConfig};
use sketchy_pix2pix::pix2pix::{discriminator::Pix2PixDescriminatorConfig, generator::Pix2PixGeneratorConfig, gan::{train, Pix2PixModelConfig, TrainingConfig}};


fn main() {
    type MyBackend = Wgpu<f32, i32>;
    type MyAutodiffBackend = Autodiff<MyBackend>;

    let device = burn::backend::wgpu::WgpuDevice::default();
    let artifact_dir = "./tmp";
    train::<MyAutodiffBackend>(
        artifact_dir,
        TrainingConfig::new(Pix2PixModelConfig::new(Pix2PixDescriminatorConfig::new(), Pix2PixGeneratorConfig::new()), 
        AdamConfig::new()),
        device.clone(),
    );
}