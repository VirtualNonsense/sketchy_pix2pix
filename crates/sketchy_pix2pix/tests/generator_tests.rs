#[cfg(test)]
mod generator {
    use std::{
        path::{Path, PathBuf},
        str::FromStr,
    };

    use burn::{backend::Wgpu, data::dataloader::batcher::Batcher, tensor::Shape};
    use sketchy_pix2pix::{
        pix2pix::generator::{Pix2PixGenerator, Pix2PixGeneratorConfig},
        sketchy_database::{
            sketchy_batcher::SketchyBatcher,
            sketchy_dataset::{SketchyClass, SketchyItem},
        },
    };

    #[test]
    fn init() {
        type MyBackend = Wgpu<f32, i32>;

        let device = burn::backend::wgpu::WgpuDevice::default();

        let generator: Pix2PixGenerator<MyBackend> =
            Pix2PixGeneratorConfig::new().init(&device);
    }

    #[test]
    fn test_forward_and_check_shape() {
        type MyBackend = Wgpu<f32, i32>;

        let device = burn::backend::wgpu::WgpuDevice::default();

        let generator: Pix2PixGenerator<MyBackend> =
            Pix2PixGeneratorConfig::new().init(&device);

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

        let result = generator.forward(b.photos);
        
        assert_eq!(Shape::new([2, 1, 256, 256]), result.shape());

    }
}
