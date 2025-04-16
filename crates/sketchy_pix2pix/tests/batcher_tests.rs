#[cfg(test)]
mod tests {
    use std::{path::PathBuf, str::FromStr};

    use burn::{
        backend::Wgpu,
        data::dataloader::batcher::Batcher,
        tensor::{Device, Shape},
    };
    use sketchy_pix2pix::sketchy_database::{
        sketchy_batcher::{SketchyBatch, SketchyBatcher},
        sketchy_dataset::{SketchyClass, SketchyItem},
    };

    #[test]
    fn test_shape() {
        type TestBackend = Wgpu<f32, i32>;

        let device = Device::<TestBackend>::default();

        let batcher: SketchyBatcher = SketchyBatcher::new();
        let item = SketchyItem{
            sketch_class: SketchyClass::Airplane,
            photo: PathBuf::from_str("../../data/sketchydb_256x256/256x256/photo/tx_000000000000/airplane/n02691156_58.jpg").unwrap(),
            sketch: PathBuf::from_str("../../data/sketchydb_256x256/256x256/sketch/tx_000000000000/airplane/n02691156_58-1.png").unwrap(),
        };

        assert!(
            item.photo.exists(),
            "{:?} does not exist",
            std::env::current_dir().unwrap().join(item.photo)
        );
        assert!(
            item.sketch.exists(),
            "{:?} does not exist",
            std::env::current_dir().unwrap().join(item.sketch)
        );
        let items: Vec<SketchyItem> = vec![item];

        let b: SketchyBatch<TestBackend> = batcher.batch(items, &device);

        assert_eq!(Shape::new([1, 3, 256, 256]), b.photos.shape());
        assert_eq!(Shape::new([1, 1, 256, 256]), b.sketches.shape());
    }
}
