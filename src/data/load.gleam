import config.{image_size}
import fsgleam
import gleam/io
import gleam/javascript/array
import gleam/list
import gleam/string
import tensorgleam.{type Dataset, type Tensor, Dataset}

pub fn load_dataset(path: String) {
  let labels = array.to_list(fsgleam.read_dir_sync(path))

  let temp_dataset =
    list.fold(labels, #([], [], []), fn(dataset, label) {
      io.debug("Load label : ")
      io.debug(label)
      let every_tensor_and_index =
        get_tensor_and_index_img_in_label(
          string.concat([path, "/", label]),
          list.length(dataset.2),
        )
      #(
        list.append(dataset.0, every_tensor_and_index.0),
        list.append(dataset.1, every_tensor_and_index.1),
        list.append(dataset.2, [label]),
      )
    })
  let dataset =
    Dataset(
      tensorgleam.concat(array.from_list(temp_dataset.0)),
      tensorgleam.one_hot(
        tensorgleam.tensor_1d(array.from_list(temp_dataset.1), "int32"),
        list.length(temp_dataset.2),
      ),
      array.from_list(temp_dataset.2),
    )
}

pub fn get_tensor_and_index_img_in_label(
  directory_path: String,
  index: Int,
) -> #(List(Tensor), List(Int)) {
  let images_path = array.to_list(fsgleam.read_dir_sync(directory_path))
  list.fold(images_path, #([], []), fn(ts_id, image) {
    case ends_with_image_extension(image) {
      True -> {
        let image_path = string.concat([directory_path, "/", image])
        let image_augmented = load_and_augment_image(image_path, index)
        #(
          list.append(ts_id.0, image_augmented.0),
          list.append(ts_id.1, image_augmented.1),
        )
      }
      _ -> ts_id
    }
  })
}

fn load_and_augment_image(
  file_path: String,
  index: Int,
) -> #(List(Tensor), List(Int)) {
  let image1 = load_and_preprocess_image(file_path)
  let image2 =
    load_and_preprocess_image(file_path)
    |> tensorgleam.tensor_add_noise(25)
  #([image1, image2], [index, index])
}

fn load_and_preprocess_image(file_path: String) -> Tensor {
  let buffer = fsgleam.read_file_sync(file_path)
  tensorgleam.decode_image(buffer, file_path)
  |> tensorgleam.tensor_resize_nearest_neighbor(image_size)
  |> tensorgleam.tensor_to_float
  |> tensorgleam.tensor_div_scalar(255.0)
  |> tensorgleam.tensor_expand_dims
}

fn ends_with_image_extension(s: String) -> Bool {
  string.ends_with(s, ".jpeg")
  || string.ends_with(s, ".png")
  || string.ends_with(s, ".jpg")
}
