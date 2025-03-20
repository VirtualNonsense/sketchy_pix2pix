use std::usize;

use burn::{
    prelude::{Backend, Tensor},
    tensor::BasicOps,
};
use rerun::{
    external::ndarray::{self}, AsComponents,
};

pub struct LogContainer<K: ?Sized + AsComponents> {
    component: K,
}

impl<K: ?Sized + AsComponents> AsComponents for LogContainer<K> {
    fn as_serialized_batches(&self) -> Vec<rerun::SerializedComponentBatch> {
        self.component.as_serialized_batches()
    }
    fn to_arrow(
            &self,
        ) -> rerun::SerializationResult<Vec<(rerun::external::arrow::datatypes::Field, rerun::external::arrow::array::ArrayRef)>> {
        self.component.to_arrow()
    }
}

#[derive(thiserror::Error, Debug, Clone)]
pub enum LogContainerParsingError
{
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
        impl LogContainer<rerun::Tensor> {
            pub fn $function_name<B: Backend, const D: usize, K: BasicOps<B>>(
                burn_tensor: Tensor<B, D, K>,
                dim_label: [&str; D],
            ) -> Result<Self, LogContainerParsingError> {
                let tensor_data = burn_tensor.to_data();
                let shape: Result<[usize; D], _> = tensor_data.shape.clone().try_into();

                if let Err(err) = shape {
                    return Err(LogContainerParsingError::ShapeParsingError(
                        D,
                        format!("{:?}", err),
                    ));
                }
                let shape = shape.unwrap();

                let tensor_vec: Result<Vec<$d_ty>, _> = tensor_data.into_vec();

                if let Err(err) = tensor_vec {
                    return Err(LogContainerParsingError::VecParsingError(
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
                            return Err(LogContainerParsingError::RerunTensorParsingError(
                                format!("{:?}", err),
                            ));
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
                            return Err(LogContainerParsingError::RerunTensorParsingError(
                                format!("{:?}", err),
                            ));
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
                            return Err(LogContainerParsingError::RerunTensorParsingError(
                                format!("{:?}", err),
                            ));
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
                            return Err(LogContainerParsingError::RerunTensorParsingError(
                                format!("{:?}", err),
                            ));
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
                            return Err(LogContainerParsingError::RerunTensorParsingError(
                                format!("{:?}", err),
                            ));
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
                            return Err(LogContainerParsingError::RerunTensorParsingError(
                                format!("{:?}", err),
                            ));
                        }
                        t.unwrap().with_dim_names(dim_label)
                    }
                    d => {
                        return Err(LogContainerParsingError::DimensionError(d));
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
pub enum ImageGrindOptions{
    Columns(usize),
    Rows(usize),
    Exact{rows: usize, columns: usize},
    #[default]
    Auto,
}

impl LogContainer<rerun::Image>{
    
    /// Converts burn tesnor into a logcontainer that will log an image.
    /// This function assumes the following u8 tensor shape: [height, width]
    pub fn from_burn_2d_tensoru8<B: Backend>(burn_tensor: Tensor<B, 2, burn::tensor::Int>) -> Result<Self, LogContainerParsingError>{
        
        let tensor_data = burn_tensor.to_data();
        let shape: Result<[usize; 2], _> = tensor_data.shape.clone().try_into();

        if let Err(err) = shape {
            return Err(LogContainerParsingError::ShapeParsingError(
                2,
                format!("{:?}", err),
            ));
        }
        let shape = shape.unwrap();

        let tensor_vec = tensor_data.into_vec::<i32>();

        if let Err(err) = tensor_vec {
            return Err(LogContainerParsingError::VecParsingError(
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
        if let Err(err) = image{
            return Err(LogContainerParsingError::ImageConstructionError(format!("{:?}", err)))
        }
        Ok(Self { component: image.unwrap() })
    }

    /// Converts burn tesnor into a logcontainer that will log an image.
    /// This function assumes the following u8 tensor shape: [height, width, rgb_color/rbga_color]
    pub fn from_burn_3d_tensoru8<B: Backend>(burn_tensor: Tensor<B, 3, burn::tensor::Int>) -> Result<Self, LogContainerParsingError>{
        
        let tensor_data = burn_tensor.to_data();
        let shape: Result<[usize; 3], _> = tensor_data.shape.clone().try_into();

        if let Err(err) = shape {
            return Err(LogContainerParsingError::ShapeParsingError(
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
            return Err(LogContainerParsingError::VecParsingError(
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
        }
        else if c == 3{
            image = rerun::Image::from_color_model_and_tensor(rerun::ColorModel::RGB, nd);
        }
        else if c == 4 {
            image = rerun::Image::from_color_model_and_tensor(rerun::ColorModel::RGBA, nd);
        }
        else{
            return Err(LogContainerParsingError::ImageConstructionError(format!("{} is not a valid option for the color channel! choose either 1 (L), 3 (rgb) or 4 (rgba)", c)));
        }
        if let Err(err) = image{
            return Err(LogContainerParsingError::ImageConstructionError(format!("{:?}", err)))
        }
        Ok(Self { component: image.unwrap() })
    }

    /// Converts burn tesnor into a logcontainer that will log an image.
    /// This function assumes the following u8 tensor shape: [batch, height, width, rgb_color/rbga_color]
    pub fn from_burn_4d_tensoru8<B: Backend>(burn_tensor: Tensor<B, 4, burn::tensor::Int>, grid_settings: ImageGrindOptions) -> Result<Self, LogContainerParsingError> {
        let shape: [usize; 4] = burn_tensor.shape().dims();
        let b = shape[0];
        let h = shape[1];
        let w = shape[2];
        let c = shape[3];
    
        if b == 1 {
            let burn_tensor = burn_tensor.reshape([shape[1], shape[2], shape[3]]);
            return Self::from_burn_3d_tensoru8(burn_tensor);
        }
    
        let (rows, columns) = match grid_settings {
            ImageGrindOptions::Columns(c) => {
                let r = b as f32 / c as f32;
                (r.floor() as usize + 1, c)
            },
            ImageGrindOptions::Rows(r) => {
                let c = b as f32 / r as f32;
                (r, c.floor() as usize + 1)
            },
            ImageGrindOptions::Exact { rows, columns } => {
                if rows * columns < b {
                    panic!("Invalid input! Make sure that {} x {} >= {}", rows, columns, b);
                };
                (rows, columns)
            },
            ImageGrindOptions::Auto => {
                let root = (b as f32).sqrt();
                if root % 1. == 0. {
                    (root as usize, root as usize)
                } else {
                    ((root.floor() as usize + 1), root.floor() as usize)
                }
            },
        };
    
        let height = rows * h;
        let width = columns * w;
        let mut stitched_tensor: Tensor<B, 3, burn::tensor::Int> = Tensor::zeros([height, width, c], &burn_tensor.device());
    
        for i in 0..b {
            let row = i / columns;
            let col = i % columns;
            let start_row = row * h;
            let start_col = col * w;
            let end_row = start_row + h;
            let end_col = start_col + w;
    
            let slice = burn_tensor.clone().slice([i..i+1, 0..h, 0..w, 0..c]);
            let reshaped_slice = slice.reshape([h, w, c]);
            stitched_tensor = stitched_tensor.slice_assign([start_row..end_row, start_col..end_col, 0..c], reshaped_slice);
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
        let tensor: Tensor<Wgpu, 2, burn::prelude::Int> = Tensor::from_data([
                [1, 2], 
                [3, 4]
            ], &burn::backend::wgpu::WgpuDevice::default());
        let log_container = LogContainer::from_burn_2d_tensoru8(tensor);
        assert!(log_container.is_ok());
    }
    #[test]
    fn test_from_burn_4d_tensoru8() {
        let tensor: Tensor<Wgpu, 4, burn::prelude::Int> = Tensor::from_data([
            [
                [[1], [2]], 
                [[3], [4]]
            ],
            [
                [[5], [6]], 
                [[7], [8]]
            ]
            
        ], &burn::backend::wgpu::WgpuDevice::default());
        let log_container = LogContainer::from_burn_4d_tensoru8(tensor, ImageGrindOptions::Auto);
        assert!(log_container.is_ok());
    }
}