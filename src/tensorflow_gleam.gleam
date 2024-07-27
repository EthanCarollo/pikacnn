import config
import data/load.{load_dataset}
import gleam/io
import gleam/javascript/array
import gleam/javascript/promise.{await, new, tap}
import gleam/list
import model/construct.{construct_model}
import tensorgleam

pub fn main() {
  tensorgleam.disable_warning()
  tensorgleam.log("Start to load dataset")
  let dataset = load.load_dataset(config.data_path)
  tensorgleam.log("Loaded dataset")
  tensorgleam.log(dataset)

  let model = get_model(list.length(array.to_list(dataset.label_map)))
  let usable = tensorgleam.dataset_to_usable(dataset)
  train_model(model, usable)
}

pub fn get_model(class: Int) {
  construct_model(class)
  |> tensorgleam.model_summary
}

pub fn train_model(
  model: tensorgleam.Model,
  usable_dataset: tensorgleam.UsableDataset,
) {
  tap(
    tensorgleam.model_fit(
      model,
      usable_dataset,
      config.batch_size,
      0.2,
      config.epoch,
    ),
    fn(history) { io.debug(history) },
  )
}
