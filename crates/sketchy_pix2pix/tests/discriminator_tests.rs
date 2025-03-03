#[cfg(test)]
mod discriminator {
    use std::{
        path::PathBuf,
        str::FromStr,
    };

    use burn::{backend::Wgpu, data::dataloader::batcher::Batcher, tensor::Shape};
    use sketchy_pix2pix::{
        pix2pix::discriminator::{Pix2PixDiscriminator, Pix2PixDescriminatorConfig},
        sketchy_database::{
            sketchy_batcher::SketchyBatcher,
            sketchy_dataset::{SketchyClass, SketchyItem},
        },
    };

    #[test]
    fn init() {
        type MyBackend = Wgpu<f32, i32>;

        let device = burn::backend::wgpu::WgpuDevice::default();

        let _generator: Pix2PixDiscriminator<MyBackend> =
        Pix2PixDescriminatorConfig::new().init(&device);
    }

    #[test]
    fn test_forward_and_check_shape() {
        type MyBackend = Wgpu<f32, i32>;

        let device = burn::backend::wgpu::WgpuDevice::default();

        let generator: Pix2PixDiscriminator<MyBackend> =
        Pix2PixDescriminatorConfig::new().init(&device);

        let batcher: SketchyBatcher<MyBackend> = SketchyBatcher::new(device);

        let items = vec![
            SketchyItem{
                sketch_class: SketchyClass::Airplane,
                photo: PathBuf::from_str("./data/sketchydb_256x256/256x256/photo/tx_000000000000/airplane/n02691156_58.jpg").unwrap(),
                sketch: PathBuf::from_str("./data/sketchydb_256x256/256x256/sketch/tx_000000000000/airplane/n02691156_58-1.png").unwrap(),
            }, 
            SketchyItem{
                sketch_class: SketchyClass::Airplane,
                photo: PathBuf::from_str("./data/sketchydb_256x256/256x256/photo/tx_000000000000/airplane/n02691156_58.jpg").unwrap(),
                sketch: PathBuf::from_str("./data/sketchydb_256x256/256x256/sketch/tx_000000000000/airplane/n02691156_58-2.png").unwrap(),
            }
        ];

        let b = batcher.batch(items);

        let result = generator.forward(b.sketches.clone(), b.sketches.clone());
        
        assert_eq!(Shape::new([2, 1, 14, 14]), result.shape());

    }
}
