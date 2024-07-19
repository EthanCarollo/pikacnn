import tensorgleam.{type Layer}

@external(javascript, "../blackandwhite_ffi.mjs", "getBlackAndWhiteLayer")
pub fn get_black_and_white_layer() -> Layer

@external(javascript, "../blackandwhite_ffi.mjs", "getBlackAndWhiteLayerInput")
pub fn get_black_and_white_input(image_size: Int) -> Layer
