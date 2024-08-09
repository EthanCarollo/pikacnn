import gleam/javascript/array.{type Array}
import gleam/javascript/promise.{type Promise}

pub type Model

pub type Layer

pub type Tensor

pub type UsableDataset

pub type Dataset {
  Dataset(xs: Tensor, ys: Tensor, label_map: Array(String))
}

@external(javascript, "./tensorgleam_ffi.mjs", "disableWarning")
pub fn disable_warning() -> Nil

@external(javascript, "./tensorgleam_ffi.mjs", "getSequentialModel")
pub fn get_sequential_model() -> Model

@external(javascript, "./tensorgleam_ffi.mjs", "addLayerToModel")
pub fn add_layer_to_model(model: Model, layer: Layer) -> Model

@external(javascript, "./tensorgleam_ffi.mjs", "log")
pub fn log(info: a) -> a

@external(javascript, "./tensorgleam_ffi.mjs", "getConvolution2DLayerNoInput")
pub fn get_convolution_2d_layer_no_input(
  filters: Int,
  kernel_size: Int,
  activation: String,
) -> Layer

@external(javascript, "./tensorgleam_ffi.mjs", "getConvolution2DLayerPadding")
pub fn get_convolution_2d_layer_no_input_padding(
  filters: Int,
  kernel_size: Int,
  padding: String,
  activation: String,
) -> Layer

@external(javascript, "./tensorgleam_ffi.mjs", "getConvolution2DLayer")
pub fn get_convolution_2d_layer(
  input_shape: Array(Int),
  filters: Int,
  kernel_size: Int,
  activation: String,
) -> Layer

@external(javascript, "./tensorgleam_ffi.mjs", "getMaxPooling2D")
pub fn get_max_pooling_2d_layer(pool_size: Array(Int)) -> Layer

@external(javascript, "./tensorgleam_ffi.mjs", "getFlatten")
pub fn get_flatten() -> Layer

@external(javascript, "./tensorgleam_ffi.mjs", "getBatchNormalization")
pub fn get_batch_normalization() -> Layer

@external(javascript, "./tensorgleam_ffi.mjs", "getDense")
pub fn get_dense(units: Int, activation: String) -> Layer

@external(javascript, "./tensorgleam_ffi.mjs", "getDropOut")
pub fn get_drop_out(rate: Float) -> Layer

@external(javascript, "./tensorgleam_ffi.mjs", "modelCompile")
pub fn model_compile(
  model: Model,
  optimizer: String,
  loss: String,
  metrics: Array(String),
  learning_rate: Float,
) -> Model

@external(javascript, "./tensorgleam_ffi.mjs", "modelSummary")
pub fn model_summary(model: Model) -> Model

// TODO : MODIFY THIS TO PUT REAL TYPE
@external(javascript, "./tensorgleam_ffi.mjs", "modelFit")
pub fn model_fit(
  model: Model,
  train_data: UsableDataset,
  batch_size: Int,
  validation_split: Float,
  epochs: Int,
) -> Promise(String)

// TODO : MODIFY THIS TO PUT REAL TYPE AGAIN LOL
@external(javascript, "./tensorgleam_ffi.mjs", "modelSave")
pub fn model_save(model: Model, path: String) -> Promise(String)

/// Node and img
@external(javascript, "./tensorgleam_ffi.mjs", "decodeImage")
pub fn decode_image(buffer: String, image_name: String) -> Tensor

@external(javascript, "./tensorgleam_ffi.mjs", "tensorResizeNearestNeighbor")
pub fn tensor_resize_nearest_neighbor(
  tensor: Tensor,
  tensor_size: Int,
) -> Tensor

@external(javascript, "./tensorgleam_ffi.mjs", "tensorToFloat")
pub fn tensor_to_float(tensor: Tensor) -> Tensor

@external(javascript, "./tensorgleam_ffi.mjs", "tensorDivScalar")
pub fn tensor_div_scalar(tensor: Tensor, scalar_factor: Float) -> Tensor

@external(javascript, "./tensorgleam_ffi.mjs", "tensorExpandDims")
pub fn tensor_expand_dims(tensor: Tensor) -> Tensor

@external(javascript, "./tensorgleam_ffi.mjs", "tfConcat")
pub fn concat(tensors: Array(Tensor)) -> Tensor

@external(javascript, "./tensorgleam_ffi.mjs", "oneHot")
pub fn one_hot(tensors: Tensor, oh_length: Int) -> Tensor

@external(javascript, "./tensorgleam_ffi.mjs", "isTensorImageShape")
pub fn is_tensor_image_shape(tensors: Tensor, size: Int) -> Bool

@external(javascript, "./tensorgleam_ffi.mjs", "tensor1D")
pub fn tensor_1d(inputs: Array(Int), in_type: String) -> Tensor

@external(javascript, "./tensorgleam_ffi.mjs", "datasetToDatabaseUsableObject")
pub fn dataset_to_usable(dataset: Dataset) -> UsableDataset

@external(javascript, "./tensorgleam_ffi.mjs", "addNoise")
pub fn tensor_add_noise(image: Tensor, noise: Int) -> Tensor

@external(javascript, "./tensorgleam_ffi.mjs", "flipHorizontal")
pub fn tensor_flip_horizontal(image: Tensor) -> Tensor

@external(javascript, "./tensorgleam_ffi.mjs", "startScope")
pub fn start_scope() -> Nil

@external(javascript, "./tensorgleam_ffi.mjs", "endScope")
pub fn end_scope() -> Nil

@external(javascript, "./tensorgleam_ffi.mjs", "disposeVariables")
pub fn dispose_variables() -> Nil
