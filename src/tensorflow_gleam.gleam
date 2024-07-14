import data/load.{load_dataset}
import fsgleam
import gleam/io
import gleam/javascript/array
import gleam/javascript/promise.{await, new, tap}
import model/construct.{construct_model}
import tensorgleam

pub fn main() {
  let dataset = load.load_dataset("./data/pokemon-light")
  io.debug(dataset)
}

pub fn comqd() {
  let model =
    construct_model(150)
    |> tensorgleam.model_summary

  tap(tensorgleam.model_save(model, "file://./models/pokemon-model"), fn(toto) {
    io.debug(toto)
  })
}
///
/// await(tensorgleam.model_fit(model, "toto", "toto"), fn(history) {
///    io.debug(history)
///    new(fn(toto: fn(a) -> Nil) -> Nil { Nil })
///    // todo : show the history normally
///  })
/// 
