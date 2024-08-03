import config.{type Config, Config}
import data/load.{load_dataset}
import gleam/dynamic.{field, int, list, string}
import gleam/io
import gleam/javascript/array
import gleam/javascript/promise.{await, new, tap}
import gleam/list
import gleam/result
import model/construct.{construct_model}
import pklgleam
import tensorgleam

///pub type TestRecord{
///  TestRecord(name: String)
///}
pub fn main() {
  {
    let configuration =
      pklgleam.pkl_decode(
        "config.pkl",
        dynamic.decode7(
          Config,
          field("data_path", of: string),
          field("image_size", of: int),
          field("max_image_per_label", of: int),
          field("max_label_taken_per_train", of: int),
          field("epoch", of: int),
          field("batch_size", of: int),
          field("learning_rate", of: dynamic.float),
        ),
      )
  }
}

pub fn launch_model(config: config.Config) {
  tensorgleam.disable_warning()
  tensorgleam.log("Start to load dataset")
  let dataset = load.load_dataset(config)
  tensorgleam.log("Loaded dataset")
  tensorgleam.log(dataset)

  let model = get_model(list.length(array.to_list(dataset.label_map)), config)
  let usable = tensorgleam.dataset_to_usable(dataset)
  train_model(model, usable, config)
}

pub fn get_model(class: Int, config: config.Config) {
  construct_model(class, config)
  |> tensorgleam.model_summary
}

pub fn train_model(
  model: tensorgleam.Model,
  usable_dataset: tensorgleam.UsableDataset,
  config: config.Config,
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
