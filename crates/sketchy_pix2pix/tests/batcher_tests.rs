#[cfg(test)]
mod tests {
    use std::{path::PathBuf, str::FromStr};

    use burn::{backend::Wgpu, data::dataloader::batcher::Batcher, tensor::Shape};
    use sketchy_pix2pix::sketchy_database::{sketchy_batcher::SketchyBatcher, sketchy_dataset::{SketchyClass, SketchyItem}};

    #[test]
    fn test_shape() {
        type MyBackend = Wgpu<f32, i32>;

        let device = burn::backend::wgpu::WgpuDevice::default();
        
        let batcher: SketchyBatcher<MyBackend> = SketchyBatcher::new(device);
        let item = SketchyItem{
            sketch_class: SketchyClass::Airplane,
            photo: PathBuf::from_str("./data/sketchydb_256x256/256x256/photo/tx_000000000000/airplane/n02691156_58.jpg").unwrap(),
            sketch: PathBuf::from_str("./data/sketchydb_256x256/256x256/sketch/tx_000000000000/airplane/n02691156_58-1.png").unwrap(),
        };

        assert!(item.photo.exists());
        assert!(item.sketch.exists());
        let items = vec![item];
        
        let b = batcher.batch(items);
    
        assert_eq!(Shape::new([1, 3, 256, 256]), b.photos.shape());
        assert_eq!(Shape::new([1, 1, 256, 256]), b.sketches.shape());

    }
}