use burn::{
    prelude::{Backend, Tensor},
    tensor::{BasicOps, Int, backend::AutodiffBackend, cast::ToElement},
};
use rerun::{
    AsComponents, RecordingStream,
    external::ndarray::{self},
};

use crate::pix2pix::{
    discriminator::Pix2PixDiscriminatorGradients, gan::GanOutput,
    generator::Pix2PixGeneratorGradients,
};

pub struct SketchyGanLogger {
    stream: RecordingStream,
    pub log_image_result_interval: usize,
    pub log_grad_interval: usize,
    pub image_grid_options: ImageGrindOptions,
}

fn convert_to_picture<B: Backend>(image_tensor: Tensor<B, 4>) -> Tensor<B, 4, Int> {
    let image_tensor = image_tensor * 127.5 + 127.5;
    let image_tensor = image_tensor.int();
    image_tensor.permute([0, 2, 3, 1])
}

impl SketchyGanLogger {
    pub fn new(stream: RecordingStream) -> Self {
        Self {
            stream,
            log_image_result_interval: 25,
            log_grad_interval: 100,
            image_grid_options: ImageGrindOptions::Auto,
        }
    }

    pub fn log_progress<B: Backend>(
        &self,
        photos: Tensor<B, 4>,
        output: GanOutput<B>,
        base_path: &str,
        batch: usize,
    ) {
        let loss_d = output.loss_discriminator.clone();
        let loss_g = output.loss_generator.clone();
        let gen_loss = loss_g.into_scalar().to_f64();
        let dis_loss = loss_d.into_scalar().to_f64();
        if (batch % self.log_image_result_interval == 0) || gen_loss.is_nan() || dis_loss.is_nan() {
            match LogAble::from_burn_4d_tensoru8(
                convert_to_picture(output.real_sketch_output),
                self.image_grid_options.clone(),
            ) {
                Ok(c) => {
                    let _ = self
                        .stream
                        .log(format!("{}/raw/real_result", base_path), &c);
                }
                Err(e) => {
                    let _ = self.stream.log(
                        format!("{}/raw/real_result", base_path),
                        &rerun::TextLog::new(format!(
                            "Failed to convert input real_sketch_output due to {:?}",
                            e
                        ))
                        .with_level(rerun::TextLogLevel::ERROR),
                    );
                }
            }
            match LogAble::from_burn_4d_tensoru8(
                convert_to_picture(output.fake_sketch_output),
                self.image_grid_options.clone(),
            ) {
                Ok(c) => {
                    let _ = self
                        .stream
                        .log(format!("{}/raw/fake_result", base_path), &c);
                }
                Err(e) => {
                    let _ = self.stream.log(
                        format!("{}/raw/fake_result", base_path),
                        &rerun::TextLog::new(format!(
                            "Failed to convert input fake_sketch_output due to {:?}",
                            e
                        ))
                        .with_level(rerun::TextLogLevel::ERROR),
                    );
                }
            }

            match LogAble::from_burn_4d_tensoru8(
                convert_to_picture(photos),
                self.image_grid_options.clone(),
            ) {
                Ok(c) => match self
                    .stream
                    .log(format!("{}/images/input_images", base_path), &c)
                {
                    Ok(_) => (),
                    _ => {}
                },
                Err(e) => {
                    let _ = self.stream.log(
                        format!("{}/images/input_images", base_path),
                        &rerun::TextLog::new(format!(
                            "Failed to convert input photos due to {:?}",
                            e
                        ))
                        .with_level(rerun::TextLogLevel::ERROR),
                    );
                }
            }
            match LogAble::from_burn_4d_tensoru8(
                convert_to_picture(output.train_sketches),
                self.image_grid_options.clone(),
            ) {
                Ok(c) => {
                    let _ = self
                        .stream
                        .log(format!("{}/images/real_sketches", base_path), &c);
                }
                Err(e) => {
                    let _ = self.stream.log(
                        format!("{}/images/real_sketches", base_path),
                        &rerun::TextLog::new(format!(
                            "Failed to convert train_sketches due to {:?}",
                            e
                        ))
                        .with_level(rerun::TextLogLevel::ERROR),
                    );
                }
            }
            match LogAble::from_burn_4d_tensoru8(
                convert_to_picture(output.fake_sketches),
                self.image_grid_options.clone(),
            ) {
                Ok(c) => {
                    let _ = self
                        .stream
                        .log(format!("{}/images/fake_sketches", base_path), &c);
                }
                Err(e) => {
                    let _ = self.stream.log(
                        format!("{}/images/fake_sketches", base_path),
                        &rerun::TextLog::new(format!(
                            "Failed to convert fake_sketches due to {:?}",
                            e
                        ))
                        .with_level(rerun::TextLogLevel::ERROR),
                    );
                }
            }
        }
        let _ = self.stream.log(
            format!("graphs/{}/loss/generator", base_path),
            &rerun::Scalar::new(gen_loss),
        );
        let _ = self.stream.log(
            format!("graphs/{}/loss/discriminator", base_path),
            &rerun::Scalar::new(dis_loss),
        );

    }

    pub fn log_discriminator_gradient<B: AutodiffBackend>(
        &self,
        disc_grad: Pix2PixDiscriminatorGradients<B>,
        iteration: usize,
    ) {
        if iteration % self.log_grad_interval != 0 {
            return;
        }

        macro_rules! log_grad {
            ($name:ident, $grad_test:expr) => {
                if let Some(grad) = $grad_test {
                    if let Ok(comp) = LogAble::from_burn_tensorf32(
                        grad.mean_dim(2).mean_dim(3),
                        ["out_channel", "in_channel", "row", "column"],
                    ) {
                        let _ = self.stream.log(
                            format!("gradient/discriminator/{}", stringify!($name)),
                            &comp,
                        );
                    }
                }
            };
        }
        log_grad!(conv1, disc_grad.conv1);
        log_grad!(conv2, disc_grad.conv2);
        log_grad!(conv3, disc_grad.conv3);
        log_grad!(conv4, disc_grad.conv4);
        log_grad!(conv5, disc_grad.conv5);
        log_grad!(conv_out, disc_grad.conv_out);
    }

    pub fn log_generator_gradient<B: AutodiffBackend>(
        &self,
        gen_grad: Pix2PixGeneratorGradients<B>,
        iteration: usize,
    ) {
        if iteration % self.log_grad_interval != 0 {
            return;
        }

        macro_rules! log_grad {
            ($block:ident, $name:ident, $grad_test:expr) => {
                if let Some(grad) = $grad_test {
                    if let Ok(comp) = LogAble::from_burn_tensorf32(
                        grad.mean_dim(2).mean_dim(3),
                        ["out_channel", "in_channel", "row", "column"],
                    ) {
                        let _ = self.stream.log(
                            format!(
                                "gradient/generator/{}/{}",
                                stringify!($block),
                                stringify!($name)
                            ),
                            &comp,
                        );
                    }
                }
            };
        }
        log_grad!(enc, conv1, gen_grad.enc_conv1);
        log_grad!(enc, conv2, gen_grad.enc_conv2);
        log_grad!(enc, conv3, gen_grad.enc_conv3);
        log_grad!(enc, conv4, gen_grad.enc_conv4);
        log_grad!(enc, conv5, gen_grad.enc_conv5);
        log_grad!(enc, conv6, gen_grad.enc_conv6);
        log_grad!(enc, conv7, gen_grad.enc_conv7);
        log_grad!(enc, conv8, gen_grad.enc_conv8);
        log_grad!(dec, conv1, gen_grad.dec_conv1);
        log_grad!(dec, conv2, gen_grad.dec_conv2);
        log_grad!(dec, conv3, gen_grad.dec_conv3);
        log_grad!(dec, conv4, gen_grad.dec_conv4);
        log_grad!(dec, conv5, gen_grad.dec_conv5);
        log_grad!(dec, conv6, gen_grad.dec_conv6);
        log_grad!(dec, conv7, gen_grad.dec_conv7);
        log_grad!(dec, conv8, gen_grad.dec_conv8);
    }
}

pub struct LogAble<K: ?Sized + AsComponents> {
    component: K,
}

impl<K: ?Sized + AsComponents> AsComponents for LogAble<K> {
    fn as_serialized_batches(&self) -> Vec<rerun::SerializedComponentBatch> {
        self.component.as_serialized_batches()
    }
    fn to_arrow(
        &self,
    ) -> rerun::SerializationResult<
        Vec<(
            rerun::external::arrow::datatypes::Field,
            rerun::external::arrow::array::ArrayRef,
        )>,
    > {
        self.component.to_arrow()
    }
}

#[derive(thiserror::Error, Debug, Clone)]
pub enum LogAbleParsingError {
    #[error("Failed to parse shape to [usize; {}] due to {}", .0, .1)]
    ShapeParsingError(usize, String),
    #[error("Failed to tensor data to Vec<{}> due to {}", .0, .1)]
    VecParsingError(String, String),
    #[error("{} is not implemented at this point", .0)]
    DimensionError(usize),
    #[error("Unable to parse tensor due to {}", .0)]
    RerunTensorParsingError(String),
    #[error("failed to convert from image due to {}", .0)]
    ImageConstructionError(String),
}

fn get_val<IT: Copy, const ID: usize>(
    data: &Vec<IT>,
    data_shape: &[usize; ID],
    pos: &[usize; ID],
) -> IT {
    let mut idx = pos[ID - 1];
    for i in 1..ID {
        let mut tmp = pos[i - 1];
        for j in i..ID {
            tmp *= data_shape[j];
        }
        idx += tmp;
    }
    data[idx]
}

macro_rules! impl_from_burn_tensore {
    ($function_name:ident, $d_ty:ty) => {
        impl LogAble<rerun::Tensor> {
            pub fn $function_name<B: Backend, const D: usize, K: BasicOps<B>>(
                burn_tensor: Tensor<B, D, K>,
                dim_label: [&str; D],
            ) -> Result<Self, LogAbleParsingError> {
                let tensor_data = burn_tensor.to_data();
                let shape: Result<[usize; D], _> = tensor_data.shape.clone().try_into();

                if let Err(err) = shape {
                    return Err(LogAbleParsingError::ShapeParsingError(
                        D,
                        format!("{:?}", err),
                    ));
                }
                let shape = shape.unwrap();

                let tensor_vec: Result<Vec<$d_ty>, _> = tensor_data.into_vec();

                if let Err(err) = tensor_vec {
                    return Err(LogAbleParsingError::VecParsingError(
                        stringify!($d_ty).into(),
                        format!("{:?}", err),
                    ));
                }
                let tensor_vec = tensor_vec.unwrap();

                let t = match D {
                    1 => {
                        let ishape: [usize; 1] = [shape[0]];
                        let nd = ndarray::Array1::<$d_ty>::from_shape_fn(ishape, |n: usize| {
                            tensor_vec[n]
                        });
                        let t = rerun::Tensor::try_from(nd);
                        if let Err(err) = t {
                            return Err(LogAbleParsingError::RerunTensorParsingError(format!(
                                "{:?}",
                                err
                            )));
                        }
                        t.unwrap().with_dim_names(dim_label)
                    }
                    2 => {
                        let ishape: [usize; 2] = [shape[0], shape[1]];
                        let nd = ndarray::Array2::<$d_ty>::from_shape_fn(ishape, |n| {
                            let n = n.into();
                            get_val(&tensor_vec, &ishape, &n)
                        });
                        let t = rerun::Tensor::try_from(nd);
                        if let Err(err) = t {
                            return Err(LogAbleParsingError::RerunTensorParsingError(format!(
                                "{:?}",
                                err
                            )));
                        }
                        t.unwrap().with_dim_names(dim_label)
                    }
                    3 => {
                        let ishape: [usize; 3] = [shape[0], shape[1], shape[2]];
                        let nd = ndarray::Array3::<$d_ty>::from_shape_fn(ishape, |n| {
                            let n = n.into();
                            get_val(&tensor_vec, &ishape, &n)
                        });
                        let t = rerun::Tensor::try_from(nd);
                        if let Err(err) = t {
                            return Err(LogAbleParsingError::RerunTensorParsingError(format!(
                                "{:?}",
                                err
                            )));
                        }
                        t.unwrap().with_dim_names(dim_label)
                    }
                    4 => {
                        let ishape: [usize; 4] = [shape[0], shape[1], shape[2], shape[3]];
                        let nd = ndarray::Array4::<$d_ty>::from_shape_fn(ishape, |n| {
                            let n = n.into();
                            get_val(&tensor_vec, &ishape, &n)
                        });
                        let t = rerun::Tensor::try_from(nd);
                        if let Err(err) = t {
                            return Err(LogAbleParsingError::RerunTensorParsingError(format!(
                                "{:?}",
                                err
                            )));
                        }
                        t.unwrap().with_dim_names(dim_label)
                    }
                    5 => {
                        let ishape: [usize; 5] = [shape[0], shape[1], shape[2], shape[3], shape[4]];
                        let nd = ndarray::Array5::<$d_ty>::from_shape_fn(ishape, |n| {
                            let n = n.into();
                            get_val(&tensor_vec, &ishape, &n)
                        });
                        let t = rerun::Tensor::try_from(nd);
                        if let Err(err) = t {
                            return Err(LogAbleParsingError::RerunTensorParsingError(format!(
                                "{:?}",
                                err
                            )));
                        }
                        t.unwrap().with_dim_names(dim_label)
                    }
                    6 => {
                        let ishape: [usize; 6] =
                            [shape[0], shape[1], shape[2], shape[3], shape[4], shape[5]];
                        let nd = ndarray::Array6::<$d_ty>::from_shape_fn(ishape, |n| {
                            let n = n.into();
                            get_val(&tensor_vec, &ishape, &n)
                        });
                        let t = rerun::Tensor::try_from(nd);
                        if let Err(err) = t {
                            return Err(LogAbleParsingError::RerunTensorParsingError(format!(
                                "{:?}",
                                err
                            )));
                        }
                        t.unwrap().with_dim_names(dim_label)
                    }
                    d => {
                        return Err(LogAbleParsingError::DimensionError(d));
                    }
                };
                Ok(Self { component: t })
            }
        }
    };
}

impl_from_burn_tensore!(from_burn_tensorf32, f32);
impl_from_burn_tensore!(from_burn_tensorf64, f64);
impl_from_burn_tensore!(from_burn_tensoru8, u8);
impl_from_burn_tensore!(from_burn_tensoru16, u16);
impl_from_burn_tensore!(from_burn_tensoru32, u32);
impl_from_burn_tensore!(from_burn_tensoru64, u64);
impl_from_burn_tensore!(from_burn_tensori16, i16);
impl_from_burn_tensore!(from_burn_tensori32, i32);
impl_from_burn_tensore!(from_burn_tensori64, i64);

#[derive(Default, Clone, Debug, PartialEq)]
pub enum ImageGrindOptions {
    Columns(usize),
    Rows(usize),
    Exact {
        rows: usize,
        columns: usize,
    },
    #[default]
    Auto,
}

impl ImageGrindOptions {
    fn into_row_column(self, batch_size: usize) -> (usize, usize) {
        match self {
            ImageGrindOptions::Columns(c) => {
                let r = batch_size as f32 / c as f32;
                (r.floor() as usize + 1, c)
            }
            ImageGrindOptions::Rows(r) => {
                let c = batch_size as f32 / r as f32;
                (r, c.floor() as usize + 1)
            }
            ImageGrindOptions::Exact { rows, columns } => {
                if rows * columns < batch_size {
                    panic!(
                        "Invalid input! Make sure that {} x {} >= {}",
                        rows, columns, batch_size
                    );
                };
                (rows, columns)
            }
            ImageGrindOptions::Auto => {
                let root = (batch_size as f32).sqrt();
                if root % 1. == 0. {
                    (root as usize, root as usize)
                } else {
                    ((root.floor() as usize + 1), root.round() as usize)
                }
            }
        }
    }
}

impl LogAble<rerun::Image> {
    /// Converts burn tesnor into a logcontainer that will log an image.
    /// This function assumes the following u8 tensor shape: [height, width]
    pub fn from_burn_2d_tensoru8<B: Backend>(
        burn_tensor: Tensor<B, 2, burn::tensor::Int>,
    ) -> Result<Self, LogAbleParsingError> {
        let tensor_data = burn_tensor.to_data();
        let shape: Result<[usize; 2], _> = tensor_data.shape.clone().try_into();

        if let Err(err) = shape {
            return Err(LogAbleParsingError::ShapeParsingError(
                2,
                format!("{:?}", err),
            ));
        }
        let shape = shape.unwrap();

        let tensor_vec = tensor_data.into_vec::<i32>();

        if let Err(err) = tensor_vec {
            return Err(LogAbleParsingError::VecParsingError(
                "i32".into(),
                format!("{:?}", err),
            ));
        }
        let tensor_vec = tensor_vec.unwrap();

        let ishape: [usize; 2] = [shape[0], shape[1]];
        let nd = ndarray::Array2::<i32>::from_shape_fn(ishape, |n| {
            let n = n.into();
            get_val(&tensor_vec, &ishape, &n)
        });
        let image = rerun::Image::from_color_model_and_tensor(rerun::ColorModel::L, nd);
        if let Err(err) = image {
            return Err(LogAbleParsingError::ImageConstructionError(format!(
                "{:?}",
                err
            )));
        }
        Ok(Self {
            component: image.unwrap(),
        })
    }

    /// Converts burn tesnor into a logcontainer that will log an image.
    /// This function assumes the following u8 tensor shape: [height, width, rgb_color/rbga_color]
    pub fn from_burn_3d_tensoru8<B: Backend>(
        burn_tensor: Tensor<B, 3, burn::tensor::Int>,
    ) -> Result<Self, LogAbleParsingError> {
        let tensor_data = burn_tensor.to_data();
        let shape: Result<[usize; 3], _> = tensor_data.shape.clone().try_into();

        if let Err(err) = shape {
            return Err(LogAbleParsingError::ShapeParsingError(
                3,
                format!("{:?}", err),
            ));
        }
        let shape = shape.unwrap();

        let h = shape[0];
        let w = shape[1];
        let c = shape[2];

        let tensor_vec: Result<Vec<i32>, _> = tensor_data.into_vec();

        if let Err(err) = tensor_vec {
            return Err(LogAbleParsingError::VecParsingError(
                "i32".into(),
                format!("{:?}", err),
            ));
        }
        let tensor_vec = tensor_vec.unwrap();

        let ishape: [usize; 3] = [h, w, c];
        let nd = ndarray::Array3::<i32>::from_shape_fn(ishape, |n| {
            let n = n.into();
            get_val(&tensor_vec, &ishape, &n)
        });

        let image;
        if c == 1 {
            image = rerun::Image::from_color_model_and_tensor(rerun::ColorModel::L, nd);
        } else if c == 3 {
            image = rerun::Image::from_color_model_and_tensor(rerun::ColorModel::RGB, nd);
        } else if c == 4 {
            image = rerun::Image::from_color_model_and_tensor(rerun::ColorModel::RGBA, nd);
        } else {
            return Err(LogAbleParsingError::ImageConstructionError(format!(
                "{} is not a valid option for the color channel! choose either 1 (L), 3 (rgb) or 4 (rgba)",
                c
            )));
        }
        if let Err(err) = image {
            return Err(LogAbleParsingError::ImageConstructionError(format!(
                "{:?}",
                err
            )));
        }
        Ok(Self {
            component: image.unwrap(),
        })
    }

    /// Converts burn tesnor into a logcontainer that will log an image.
    /// This function assumes the following u8 tensor shape: [batch, height, width, rgb_color/rbga_color]
    pub fn from_burn_4d_tensoru8<B: Backend>(
        burn_tensor: Tensor<B, 4, burn::tensor::Int>,
        grid_settings: ImageGrindOptions,
    ) -> Result<Self, LogAbleParsingError> {
        let shape: [usize; 4] = burn_tensor.shape().dims();
        let b = shape[0];
        let h = shape[1];
        let w = shape[2];
        let c = shape[3];

        if b == 1 {
            let burn_tensor = burn_tensor.reshape([shape[1], shape[2], shape[3]]);
            return Self::from_burn_3d_tensoru8(burn_tensor);
        }

        let (rows, columns) = grid_settings.into_row_column(b);

        let height = rows * h;
        let width = columns * w;
        let mut stitched_tensor: Tensor<B, 3, burn::tensor::Int> =
            Tensor::zeros([height, width, c], &burn_tensor.device());

        for i in 0..b {
            let row = i / columns;
            let col = i % columns;
            let start_row = row * h;
            let start_col = col * w;
            let end_row = start_row + h;
            let end_col = start_col + w;

            let slice = burn_tensor.clone().slice([i..i + 1, 0..h, 0..w, 0..c]);
            let reshaped_slice = slice.reshape([h, w, c]);
            stitched_tensor = stitched_tensor.slice_assign(
                [start_row..end_row, start_col..end_col, 0..c],
                reshaped_slice,
            );
        }
        Self::from_burn_3d_tensoru8(stitched_tensor)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::Wgpu;
    use burn::tensor::Tensor;

    #[test]
    fn test_from_burn_2d_tensoru8() {
        let tensor: Tensor<Wgpu, 2, burn::prelude::Int> = Tensor::from_data(
            [[1, 2], [3, 4]],
            &burn::backend::wgpu::WgpuDevice::default(),
        );
        let log_container = LogAble::from_burn_2d_tensoru8(tensor);
        assert!(log_container.is_ok());
    }
    #[test]
    fn test_from_burn_4d_tensoru8() {
        let tensor: Tensor<Wgpu, 4, burn::prelude::Int> = Tensor::from_data(
            [[[[1], [2]], [[3], [4]]], [[[5], [6]], [[7], [8]]]],
            &burn::backend::wgpu::WgpuDevice::default(),
        );
        let log_container = LogAble::from_burn_4d_tensoru8(tensor, ImageGrindOptions::Auto);
        assert!(log_container.is_ok());
    }
    #[test]
    fn test_auto_batch_size_2() {
        let grid = ImageGrindOptions::Auto;
        let (rows, columns) = grid.into_row_column(2);
        assert_eq!(rows, 2);
        assert_eq!(columns, 1);
    }
    #[test]
    fn test_auto_batch_size_3() {
        let grid = ImageGrindOptions::Auto;
        let (rows, columns) = grid.into_row_column(3);
        assert_eq!(rows, 2);
        assert_eq!(columns, 2);
    }
    #[test]
    fn test_auto_batch_size_4() {
        let grid = ImageGrindOptions::Auto;
        let (rows, columns) = grid.into_row_column(4);
        assert_eq!(rows, 2);
        assert_eq!(columns, 2);
    }
}
