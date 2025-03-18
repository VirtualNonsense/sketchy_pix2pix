use burn::{
    config::Config, module::Module, nn::{
        conv::{Conv2d, Conv2dConfig, ConvTranspose2d, ConvTranspose2dConfig}, BatchNorm, BatchNormConfig, Dropout, DropoutConfig, Initializer, LeakyRelu, LeakyReluConfig, PaddingConfig2d, Relu, Tanh
    }, prelude::Backend, tensor::Tensor
};

#[derive(Module, Debug)]
pub struct Pix2PixGenerator<B: Backend> {
    // Encoder-Layer
    enc_conv1: Conv2d<B>,
    enc_conv2: Conv2d<B>,
    enc_bn2: BatchNorm<B, 2>,
    enc_conv3: Conv2d<B>,
    enc_bn3: BatchNorm<B, 2>,
    enc_conv4: Conv2d<B>,
    enc_bn4: BatchNorm<B, 2>,
    enc_conv5: Conv2d<B>,
    enc_bn5: BatchNorm<B, 2>,
    enc_conv6: Conv2d<B>,
    enc_bn6: BatchNorm<B, 2>,
    enc_conv7: Conv2d<B>,
    enc_bn7: BatchNorm<B, 2>,
    enc_conv8: Conv2d<B>, // Bottleneck (ohne BatchNorm)

    // Decoder-Layer
    dec_conv_t1: ConvTranspose2d<B>,
    dec_bn1: BatchNorm<B, 2>,
    dec_conv_t2: ConvTranspose2d<B>,
    dec_bn2: BatchNorm<B, 2>,
    dec_conv_t3: ConvTranspose2d<B>,
    dec_bn3: BatchNorm<B, 2>,
    dec_conv_t4: ConvTranspose2d<B>,
    dec_bn4: BatchNorm<B, 2>,
    dec_conv_t5: ConvTranspose2d<B>,
    dec_bn5: BatchNorm<B, 2>,
    dec_conv_t6: ConvTranspose2d<B>,
    dec_bn6: BatchNorm<B, 2>,
    dec_conv_t7: ConvTranspose2d<B>,
    dec_bn7: BatchNorm<B, 2>,
    dec_conv_t8: ConvTranspose2d<B>, // letzter Transposed-Conv (Output, ohne BatchNorm)

    // Aktivierungen und Dropout
    lrelu: LeakyRelu, // LeakyReLU(alpha=0.2)
    relu: Relu,       // ReLU für Decoder
    tanh: Tanh,       // Tanh für den Ausgang
    dropout: Dropout, // Dropout 50%
}

#[derive(Config, Debug)]
pub struct Pix2PixGeneratorConfig {
    #[config(default = "1")]
    pub in_channels: usize,
    #[config(default = "1")]
    pub out_channels: usize,
}

impl Pix2PixGeneratorConfig {

    /// Initialisiert das Generator-Modul (Parameter auf gegebener Device allocieren)
    pub fn init<B: Backend>(&self, device: &B::Device) -> Pix2PixGenerator<B> {
        // Gemeinsame Konfigurationen für Convs: Kernel 4x4, Stride 2, "same" Padding, Normal(0, 0.02) Initialisierung
        let conv_cfg = |cin, cout| {
            Conv2dConfig::new([cin, cout], [4, 4])
                .with_stride([2, 2])
                .with_padding(PaddingConfig2d::Explicit(1, 1))
                .with_initializer(Initializer::Normal {
                    mean: 0.0,
                    std: 0.02,
                })
        };
        let conv_t_cfg = |cin, cout| {
            ConvTranspose2dConfig::new([cin, cout], [4, 4])
                .with_stride([2, 2])
                .with_padding([1, 1]) // entsprechend "same" für ConvTranspose2d
                .with_padding_out([0, 0]) // kein zusätzliches Output-Padding
                .with_initializer(Initializer::Normal {
                    mean: 0.0,
                    std: 0.02,
                })
        };
        Pix2PixGenerator {
            // Encoder-Blöcke
            enc_conv1: conv_cfg(self.in_channels, 64).init(device), // 1. Encoder (kein BatchNorm)
            enc_conv2: conv_cfg(64, 128).init(device),
            enc_bn2: BatchNormConfig::new(128).init(device),
            enc_conv3: conv_cfg(128, 256).init(device),
            enc_bn3: BatchNormConfig::new(256).init(device),
            enc_conv4: conv_cfg(256, 512).init(device),
            enc_bn4: BatchNormConfig::new(512).init(device),
            enc_conv5: conv_cfg(512, 512).init(device),
            enc_bn5: BatchNormConfig::new(512).init(device),
            enc_conv6: conv_cfg(512, 512).init(device),
            enc_bn6: BatchNormConfig::new(512).init(device),
            enc_conv7: conv_cfg(512, 512).init(device),
            enc_bn7: BatchNormConfig::new(512).init(device),
            enc_conv8: conv_cfg(512, 512).init(device), // Bottleneck-Conv (ohne BatchNorm)

            // Decoder-Blöcke (mit Skip-Verbindungen)
            dec_conv_t1: conv_t_cfg(512, 512).init(device),
            dec_bn1: BatchNormConfig::new(512).init(device),
            dec_conv_t2: conv_t_cfg(512 * 2, 512).init(device), // Input = d1-Ausgang (512) + Skip e7 (512) = 1024 Kanäle
            dec_bn2: BatchNormConfig::new(512).init(device),
            dec_conv_t3: conv_t_cfg(512 * 2, 512).init(device),
            dec_bn3: BatchNormConfig::new(512).init(device),
            dec_conv_t4: conv_t_cfg(512 * 2, 512).init(device),
            dec_bn4: BatchNormConfig::new(512).init(device),
            dec_conv_t5: conv_t_cfg(512 * 2, 256).init(device),
            dec_bn5: BatchNormConfig::new(256).init(device),
            dec_conv_t6: conv_t_cfg(256 * 2, 128).init(device),
            dec_bn6: BatchNormConfig::new(128).init(device),
            dec_conv_t7: conv_t_cfg(128 * 2, 64).init(device),
            dec_bn7: BatchNormConfig::new(64).init(device),
            dec_conv_t8: conv_t_cfg(64 * 2, self.out_channels).init(device), // letzter Upsampling-Conv (Ausgabe)

            // Dropout und Aktivierungen
            dropout: DropoutConfig::new(0.5).init(),
            lrelu: LeakyReluConfig::new().with_negative_slope(0.2).init(), // alpha=0.2 wie Keras LeakyReLU
            relu: Relu::new(),
            tanh: Tanh::new(),
        }
    }
}

impl<B: Backend> Pix2PixGenerator<B> {
    /// Forward-Pass: wendet Encoder-Decoder mit Skip-Connections an
    pub fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        // Encoder
        // TODO: maybe it's possible to get rid of clone.
        let e1 = self.lrelu.forward(self.enc_conv1.forward(x)); // 64  @ 128x128
        let e2 = self
            .lrelu
            .forward(self.enc_bn2.forward(self.enc_conv2.forward(e1.clone()))); // 128 @ 64x64
        let e3 = self
            .lrelu
            .forward(self.enc_bn3.forward(self.enc_conv3.forward(e2.clone()))); // 256 @ 32x32
        let e4 = self
            .lrelu
            .forward(self.enc_bn4.forward(self.enc_conv4.forward(e3.clone()))); // 512 @ 16x16
        let e5 = self
            .lrelu
            .forward(self.enc_bn5.forward(self.enc_conv5.forward(e4.clone()))); // 512 @ 8x8
        let e6 = self
            .lrelu
            .forward(self.enc_bn6.forward(self.enc_conv6.forward(e5.clone()))); // 512 @ 4x4

        let e7 = self
            .lrelu
            .forward(self.enc_bn7.forward(self.enc_conv7.forward(e6.clone()))); // 512 @ 2x2
        let b = self.relu.forward(self.enc_conv8.forward(e7.clone())); // 512 @ 1x1 (Bottleneck, ReLU)

        // Decoder (mit Skip-Verknüpfungen; Dropout in den ersten 3 Decodern)
        let d1_u = self.dec_conv_t1.forward(b);
        let d1 = self.relu.forward(Tensor::cat(
            vec![self.dropout.forward(self.dec_bn1.forward(d1_u)), e7],
            1,
        )); // 512->1024 @ 2x2
        let d2_u = self.dec_conv_t2.forward(d1);
        let d2 = self.relu.forward(Tensor::cat(
            vec![self.dropout.forward(self.dec_bn2.forward(d2_u)), e6],
            1,
        )); // 512->1024 @ 4x4
        let d3_u = self.dec_conv_t3.forward(d2);
        let d3 = self.relu.forward(Tensor::cat(
            vec![self.dropout.forward(self.dec_bn3.forward(d3_u)), e5],
            1,
        )); // 512->1024 @ 8x8
        let d4_u = self.dec_conv_t4.forward(d3);
        let d4 = self
            .relu
            .forward(Tensor::cat(vec![self.dec_bn4.forward(d4_u), e4], 1)); // 512->1024 @ 16x16
        let d5_u = self.dec_conv_t5.forward(d4);
        let d5 = self
            .relu
            .forward(Tensor::cat(vec![self.dec_bn5.forward(d5_u), e3], 1)); // 256->512 @ 32x32
        let d6_u = self.dec_conv_t6.forward(d5);
        let d6 = self
            .relu
            .forward(Tensor::cat(vec![self.dec_bn6.forward(d6_u), e2], 1)); // 128->256 @ 64x64
        let d7_u = self.dec_conv_t7.forward(d6);
        let d7 = self
            .relu
            .forward(Tensor::cat(vec![self.dec_bn7.forward(d7_u), e1], 1)); // 64->128 @ 128x128
        let d8 = self.dec_conv_t8.forward(d7); // 3   @ 256x256
        self.tanh.forward(d8) // Tanh-Ausgabe im Bereich [-1, 1]
    }
}
