use burn::{
    nn::loss::{BinaryCrossEntropyLoss, BinaryCrossEntropyLossConfig, MseLoss},
    prelude::*,
    tensor::{Tensor, Transaction, backend::AutodiffBackend},
    train::{
        TrainOutput, TrainStep, ValidStep,
        metric::{Adaptor, ItemLazy, LossInput},
    },
};

use crate::sketchy_database::sketchy_batcher::SketchyBatch;

use super::{
    discriminator::{Pix2PixDescriminatorConfig, Pix2PixDiscriminator},
    generator::{Pix2PixGenerator, Pix2PixGeneratorConfig},
};

#[derive(Config, Debug)]
pub struct Pix2PixModelConfig {
    pub discriminator_config: Pix2PixDescriminatorConfig,
    pub generator_config: Pix2PixGeneratorConfig,
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
    pub discriminator: Pix2PixDiscriminator<B>,
    pub generator: Pix2PixGenerator<B>,
    pub mse_loss: MseLoss,
    pub bce_loss: BinaryCrossEntropyLoss<B>,
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
            .forward(
                generated_sketches.clone().detach(),
                item.photos.clone().detach(),
            );
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
