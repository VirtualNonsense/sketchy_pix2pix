#[cfg(test)]
mod end_to_end_tests {
    use std::{fs, path::Path};

    use burn::data::dataset::Dataset;
    use sketchy_pix2pix::sketchy_database::sketchy_dataset::{
        PhotoAugmentation, SketchAugmentation, SketchyDataset,
    };

    /// tests if the serialization / deserialization is as good as the loading function
    #[test]
    fn test_read_all_serialize_deserialize_compare() {
        let photo_path = Path::new("./data/sketchydb_256x256/256x256/photo/");

        let sketch_path = Path::new("./data/sketchydb_256x256/256x256/sketch/");
        let serialization_path =
            Path::new("./tests/test_read_all_serialize_deserialize_compare.ron");
        let r = SketchyDataset::new(
            photo_path,
            sketch_path,
            PhotoAugmentation::ScaledAndCentered,
            SketchAugmentation::ScaledAndCentered,
        )
        .expect("unable to generate dataset");

        r.save_to_ron(serialization_path)
            .expect("unable to serialize dataset");
        let deser = SketchyDataset::load_from_ron(serialization_path)
            .expect("unable to deserialize ron rile");
        fs::remove_file(serialization_path).expect("failed to remove serialization file");
        assert_eq!(r, deser);
    }

    #[test]
    fn load_photos() {
        let photo_path = Path::new("./data/sketchydb_256x256/256x256/photo/");

        let sketch_path = Path::new("./data/sketchydb_256x256/256x256/sketch/");
        let r = SketchyDataset::new(
            photo_path,
            sketch_path,
            PhotoAugmentation::ScaledAndCentered,
            SketchAugmentation::ScaledAndCentered,
        )
        .expect("unable to generate dataset");

        let item = r.get(0).expect("failed to load item");
        let r = item.load_photo();
        assert!(r.is_ok());
    }

    #[test]
    fn load_sketches() {
        let photo_path = Path::new("./data/sketchydb_256x256/256x256/photo/");

        let sketch_path = Path::new("./data/sketchydb_256x256/256x256/sketch/");
        let r = SketchyDataset::new(
            photo_path,
            sketch_path,
            PhotoAugmentation::ScaledAndCentered,
            SketchAugmentation::ScaledAndCentered,
        )
        .expect("unable to generate dataset");

        let item = r.get(0).expect("failed to load item");
        let r = item.load_sketch();
        assert!(r.is_ok());
    }
}
