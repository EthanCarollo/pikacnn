import data/load.{load_dataset}
import fsgleam
import gleam/io
import gleam/javascript/array
import gleam/javascript/promise.{await, new, tap}
import gleam/list
import model/construct.{construct_model}
import tensorgleam

pub fn main() {
  let dataset = load.load_dataset("./data/pokemon-light")
  let model = get_model(list.length(dataset.label_map))
}

pub fn get_model(class: Int) {
  construct_model(class)
  |> tensorgleam.model_summary
}
///
/// await(tensorgleam.model_fit(model, "toto", "toto"), fn(history) {
///    io.debug(history)
///    new(fn(toto: fn(a) -> Nil) -> Nil { Nil })
///    // todo : show the history normally
///  })
/// 
