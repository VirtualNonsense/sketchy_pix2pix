use burn::{data::dataloader::batcher::Batcher, prelude::Backend, tensor::Tensor};

use super::sketchy_dataset::SketchyItem;
#[derive(Clone, Debug)]
pub struct SketchyBatch<B: Backend> {
    pub photos: Tensor<B, 4>,
    pub sketches: Tensor<B, 4>,
}

#[derive(Clone, Debug)]
pub struct SketchyBatcher<B: Backend> {
    device: B::Device,
}

impl<B: Backend> SketchyBatcher<B> {
    pub fn new(device: B::Device) -> Self {
        Self { device }
    }
}

impl<B: Backend> Batcher<SketchyItem, SketchyBatch<B>> for SketchyBatcher<B> {
    fn batch(&self, items: Vec<SketchyItem>) -> SketchyBatch<B> {
        let (photos, sketches) = items
            .iter()
            .map(|item| (item.load_photo(), item.load_sketch()))
            .filter(|result_tuple| result_tuple.0.is_ok() && result_tuple.1.is_ok())
            .map(|ok_tuple| {
                (
                    ok_tuple.0.expect("failed to extract photo"),
                    ok_tuple.1.expect("failed to extract sketch"),
                )
            }).fold((vec![], vec![]), |(mut photos, mut sketches), tuple: (burn::prelude::TensorData, burn::prelude::TensorData)| {
                photos.push(Tensor::<B, 4>::from_data(tuple.0, &self.device));
                sketches.push(Tensor::<B, 4>::from_data(tuple.1, &self.device));
                (photos, sketches)
            });
        let mut photos = Tensor::cat(photos, 0).to_device(&self.device);
        let mut sketches = Tensor::cat(sketches, 0).to_device(&self.device);
        // normalize [-1, 1]
        photos = (photos - 127.5) / 127.5;
        sketches = (sketches - 127.5) / 127.5;
        SketchyBatch{
            photos,
            sketches
        }
    }
}
