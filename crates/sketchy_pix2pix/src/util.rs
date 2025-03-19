use burn::{prelude::{Backend, Tensor}, tensor::BasicOps};
use rerun::{external::ndarray::{self}, AsComponents, EntityPath, RecordingStream, RecordingStreamError};


pub struct LogContainer<K: ?Sized + AsComponents>{
    component: K
}

#[derive(thiserror::Error, Debug, Clone)]
pub enum LogContainerParsingError{
    #[error("Failed to parse shape to [usize; {}] due to {}", .0, .1)]
    ShapeParsingError(usize, String),
    #[error("Failed to tensor data to Vec<{}> due to {}", .0, .1)]
    VecParsingError(String, String),
    #[error("{} is not implemented at this point", .0)]
    DimensionError(usize),
    #[error("Unable to parse tensor due to {}", .0)]
    RerunTensorParsingError(String)


}

impl<K:  ?Sized + AsComponents> LogContainer<K>{
    pub fn log_to_stream(&self, stream: &RecordingStream, ent_path: impl Into<EntityPath>) -> Result<(), RecordingStreamError>{
        stream.log(ent_path, &self.component)
    }
}



macro_rules! impl_from_burn_tensore {
    ($function_name:ident, $d_ty:ty) => {
impl LogContainer<rerun::Tensor>  
{
    pub fn $function_name<B: Backend, const D: usize, K: BasicOps<B>>(burn_tensor: Tensor<B, D, K>, dim_label: [&str; D]) -> Result<Self, LogContainerParsingError>
    {
        let tensor_data = burn_tensor.to_data();
        let shape: Result<[usize; D], _> = tensor_data
            .shape
            .clone()
            .try_into();

        if let Err(err) = shape{
            return Err(LogContainerParsingError::ShapeParsingError(D, format!("{:?}", err)));
        }
        let shape = shape.unwrap();
        
        let tensor_vec: Result<Vec<$d_ty>, _> = tensor_data
            .into_vec();


        if let Err(err) = tensor_vec{
            return Err(LogContainerParsingError::VecParsingError(stringify!($d_ty).into(), format!("{:?}", err)));
        }
        let tensor_vec = tensor_vec.unwrap();

            fn get_val<IT: Copy, const ID: usize>(
                data: &Vec<IT>,
                data_shape: &[usize; ID],
                pos: &[usize; ID],
            ) -> IT {
                // let index = n * data_shape[1] * data_shape[2] * data_shape[3]
                //     + c * data_shape[2] * data_shape[3]
                //     + h * data_shape[3]
                //     + w;
                let mut idx = pos[ID-1];
                for i in 1..ID{
                    let mut tmp = pos[i - 1];
                    for j in i..ID {
                        tmp *= data_shape[j];
                    }
                    idx += tmp;
                }
                data[idx]
            }

            let t = match D{
                1 => {
                    let ishape: [usize; 1] = [shape[0]];
                    let nd = ndarray::Array1::<$d_ty>::from_shape_fn(ishape, |n: usize| {
                        tensor_vec[n]
                    });
                    let t = rerun::Tensor::try_from(nd);
                    if let Err(err) = t{
                        return Err(LogContainerParsingError::RerunTensorParsingError(format!("{:?}", err)));
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
                    if let Err(err) = t{
                        return Err(LogContainerParsingError::RerunTensorParsingError(format!("{:?}", err)));
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
                    if let Err(err) = t{
                        return Err(LogContainerParsingError::RerunTensorParsingError(format!("{:?}", err)));
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
                    if let Err(err) = t{
                        return Err(LogContainerParsingError::RerunTensorParsingError(format!("{:?}", err)));
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
                    if let Err(err) = t{
                        return Err(LogContainerParsingError::RerunTensorParsingError(format!("{:?}", err)));
                    }
                    t.unwrap().with_dim_names(dim_label)
                }
                6 => {
                    let ishape: [usize; 6] = [shape[0], shape[1], shape[2], shape[3], shape[4], shape[5]];
                    let nd = ndarray::Array6::<$d_ty>::from_shape_fn(ishape, |n| {
                        let n = n.into();
                        get_val(&tensor_vec, &ishape, &n)
                    });
                    let t = rerun::Tensor::try_from(nd);
                    if let Err(err) = t{
                        return Err(LogContainerParsingError::RerunTensorParsingError(format!("{:?}", err)));
                    }
                    t.unwrap().with_dim_names(dim_label)
                }
                d => {
                    return Err(LogContainerParsingError::DimensionError(d));
                }
            };
            Ok(Self{
                component: t
            })         
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