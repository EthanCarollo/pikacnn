import gleam/javascript/array.{type Array}

// This function just return a buffer
@external(javascript, "./fsgleam_ffi.mjs", "readFileSync")
pub fn read_file_sync(path: String) -> String

@external(javascript, "./fsgleam_ffi.mjs", "readDirSync")
pub fn read_dir_sync(path: String) -> Array(String)
