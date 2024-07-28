import gleam/bit_array
import gleam/dynamic.{type DecodeError, type Dynamic}
import gleam/io
import gleam/list
import gleam/option.{type Option, None, Some}
import gleam/result
import gleam/string_builder.{type StringBuilder}

@external(javascript, "./pklgleam_ffi.mjs", "pklDecode")
fn pkl_decode_string(path: String) -> Dynamic

// Todo : update this thing to make it more flexible and explanative
pub fn pkl_decode(
  from path: String,
  using decoder: dynamic.Decoder(t),
) -> Result(t, List(DecodeError)) {
  decoder(pkl_decode_string(path))
}
