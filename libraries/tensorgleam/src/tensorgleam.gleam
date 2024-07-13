import gleam/io

pub type Model

@external(javascript, "./tensorgleam_ffi.mjs", "getSequentialModel")
pub fn get_sequential_model() -> Model

@external(javascript, "./tensorgleam_ffi.mjs", "logModel")
pub fn log_model(model: Model) -> Model
