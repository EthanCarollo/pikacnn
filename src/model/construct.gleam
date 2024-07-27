import config.{image_size}
import custom_layer/custom_layer
import gleam/javascript/array.{from_list}
import tensorgleam.{type Model}

pub fn construct_model(class: Int) -> Model {
  tensorgleam.get_sequential_model()
  |> tensorgleam.add_layer_to_model(tensorgleam.get_convolution_2d_layer(
    from_list([image_size, image_size, 3]),
    32,
    3,
    "relu",
  ))
  |> tensorgleam.add_layer_to_model(
    tensorgleam.get_max_pooling_2d_layer(from_list([2, 2])),
  )
  |> tensorgleam.add_layer_to_model(tensorgleam.get_batch_normalization())
  |> tensorgleam.add_layer_to_model(
    tensorgleam.get_convolution_2d_layer_no_input(64, 3, "relu"),
  )
  |> tensorgleam.add_layer_to_model(tensorgleam.get_batch_normalization())
  |> tensorgleam.add_layer_to_model(
    tensorgleam.get_max_pooling_2d_layer(from_list([2, 2])),
  )
  |> tensorgleam.add_layer_to_model(
    tensorgleam.get_convolution_2d_layer_no_input(128, 3, "relu"),
  )
  |> tensorgleam.add_layer_to_model(tensorgleam.get_batch_normalization())
  |> tensorgleam.add_layer_to_model(
    tensorgleam.get_max_pooling_2d_layer(from_list([2, 2])),
  )
  |> tensorgleam.add_layer_to_model(
    tensorgleam.get_convolution_2d_layer_no_input(256, 3, "relu"),
  )
  |> tensorgleam.add_layer_to_model(tensorgleam.get_batch_normalization())
  |> tensorgleam.add_layer_to_model(
    tensorgleam.get_max_pooling_2d_layer(from_list([2, 2])),
  )
  |> tensorgleam.add_layer_to_model(tensorgleam.get_flatten())
  |> tensorgleam.add_layer_to_model(tensorgleam.get_dense(1024, "relu"))
  |> tensorgleam.add_layer_to_model(tensorgleam.get_batch_normalization())
  |> tensorgleam.add_layer_to_model(tensorgleam.get_dense(class, "softmax"))
  |> tensorgleam.model_compile(
    "adam",
    "categoricalCrossentropy",
    from_list(["accuracy"]),
    config.learning_rate,
  )
}
