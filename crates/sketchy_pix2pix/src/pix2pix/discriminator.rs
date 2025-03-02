use burn::{
    nn::conv::{Conv2d, Conv2dConfig},
    prelude::*,
};
use nn::{
    BatchNorm, BatchNormConfig, Initializer, LeakyRelu, LeakyReluConfig, PaddingConfig2d, Sigmoid,
};

#[derive(Module, Debug)]
pub struct Pix2PixDiscriminator<B: Backend> {
    conv1: Conv2d<B>,
    conv2: Conv2d<B>,
    conv3: Conv2d<B>,
    conv4: Conv2d<B>,
    conv5: Conv2d<B>,
    conv_out: Conv2d<B>,
    bn2: BatchNorm<B, 2>,
    bn3: BatchNorm<B, 2>,
    bn4: BatchNorm<B, 2>,
    bn5: BatchNorm<B, 2>,
    lrelu: LeakyRelu,
    sigmoid: Sigmoid,
}

#[derive(Config, Debug)]
pub struct Pix2PixDescriminatorConfig {
    #[config(default = "1")]
    in_channels: usize,
    #[config(default = "0.02")]
    init_stddev: f64,
}

impl Pix2PixDescriminatorConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> Pix2PixDiscriminator<B> {
        let init = Initializer::Normal {
            mean: 0.0,
            std: self.init_stddev,
        };
        // first Conv: input channel = in_channels*2 (input and output will be concatinated -> their channels will be stacked), Filter = 64
        let conv1 = Conv2dConfig::new([self.in_channels * 2, 64], [4, 4])
            .with_stride([2, 2])
            .with_padding(PaddingConfig2d::Explicit(1, 1)) // "same" Padding
            .with_initializer(init.clone())
            .init(device);
        // second Conv: Filter = 128
        let conv2 = Conv2dConfig::new([64, 128], [4, 4])
            .with_stride([2, 2])
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .with_initializer(init.clone())
            .init(device);
        let bn2 = BatchNormConfig::new(128).init(device);
        // third Conv: Filter = 256
        let conv3 = Conv2dConfig::new([128, 256], [4, 4])
            .with_stride([2, 2])
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .with_initializer(init.clone())
            .init(device);
        let bn3 = BatchNormConfig::new(256).init(device);
        // forth Conv: Filter = 512
        let conv4 = Conv2dConfig::new([256, 512], [4, 4])
            .with_stride([2, 2])
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .with_initializer(init.clone())
            .init(device);
        let bn4 = BatchNormConfig::new(512).init(device);
        // firth Conv: Filter = 512, Stride = 1 (Standard)
        let conv5 = Conv2dConfig::new([512, 512], [4, 4])
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .with_initializer(init.clone())
            .init(device);
        let bn5 = BatchNormConfig::new(512).init(device);
        // Output-Conv (PatchGAN-output): 1 Filter, 1x1 Patch (4x4 Kernel with "same" results in 1x1 per Patch-Region)
        let conv_out = Conv2dConfig::new([512, 1], [4, 4])
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .with_initializer(init.clone())
            .init(device);
        // LeakyReLU-Aktivierung with alpha=0.2 und Sigmoid für den Output
        let lrelu = LeakyReluConfig::new().with_negative_slope(0.2).init();
        let sigmoid = Sigmoid::new();
        Pix2PixDiscriminator {
            conv1,
            conv2,
            conv3,
            conv4,
            conv5,
            conv_out,
            bn2,
            bn3,
            bn4,
            bn5,
            lrelu,
            sigmoid,
        }
    }
}
impl<B: Backend> Pix2PixDiscriminator<B> {
    pub fn forward(&self, img_src: Tensor<B, 4>, img_target: Tensor<B, 4>) -> Tensor<B, 4> {
        let x = Tensor::cat(vec![img_src, img_target], 1);
        let x = self.lrelu.forward(self.conv1.forward(x));
        let x = self.lrelu.forward(self.bn2.forward(self.conv2.forward(x)));
        let x = self.lrelu.forward(self.bn3.forward(self.conv3.forward(x)));
        let x = self.lrelu.forward(self.bn4.forward(self.conv4.forward(x)));
        let x = self.lrelu.forward(self.bn5.forward(self.conv5.forward(x)));
        let logits = self.conv_out.forward(x);
        self.sigmoid.forward(logits)
    }
}
