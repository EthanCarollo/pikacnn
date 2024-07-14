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

  let model = get_model(list.length(array.to_list(dataset.label_map)))
  let usable = tensorgleam.dataset_to_usable(dataset)

  tap(tensorgleam.model_fit(model, usable, 0.2), fn(history) {
    io.debug(history)
  })
}

pub fn get_model(class: Int) {
  construct_model(class)
  |> tensorgleam.model_summary
}
