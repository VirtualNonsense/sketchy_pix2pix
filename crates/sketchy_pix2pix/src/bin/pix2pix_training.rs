
use burn::{backend::{Autodiff, Wgpu}, optim::AdamConfig};
use sketchy_pix2pix::pix2pix::{discriminator::Pix2PixDescriminatorConfig, gan::{train_gan, Pix2PixModelConfig, TrainingConfig}, generator::Pix2PixGeneratorConfig};


fn main() {
    type MyBackend = Wgpu<f32, i32>;
    type MyAutodiffBackend = Autodiff<MyBackend>;

    let device = burn::backend::wgpu::WgpuDevice::default();
    let artifact_dir = "./tmp";
    train_gan::<MyAutodiffBackend>(
        artifact_dir,
        TrainingConfig::new(Pix2PixModelConfig::new(Pix2PixDescriminatorConfig::new(), Pix2PixGeneratorConfig::new()), 
        AdamConfig::new().with_beta_1(0.5),
        AdamConfig::new().with_beta_1(0.5)),
        device.clone(),
    );
}