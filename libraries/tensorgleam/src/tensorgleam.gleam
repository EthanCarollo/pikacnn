import gleam/javascript/array.{type Array}

pub type Model

pub type Layer

@external(javascript, "./tensorgleam_ffi.mjs", "getSequentialModel")
pub fn get_sequential_model() -> Model

@external(javascript, "./tensorgleam_ffi.mjs", "addLayerToModel")
pub fn add_layer_to_model(model: Model, layer: Layer) -> Model

@external(javascript, "./tensorgleam_ffi.mjs", "logModel")
pub fn log_model(model: Model) -> Model

@external(javascript, "./tensorgleam_ffi.mjs", "getConvolution2DLayerNoInput")
pub fn get_convolution_2d_layer_no_input(
  filters: Int,
  kernel_size: Int,
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
) -> Model

@external(javascript, "./tensorgleam_ffi.mjs", "modelSummary")
pub fn model_summary(model: Model) -> Model
