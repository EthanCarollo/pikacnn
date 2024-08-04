import config
import custom_layer/custom_layer
import gleam/javascript/array.{from_list}
import tensorgleam.{type Model}

pub fn construct_model(class: Int, config: config.Config) -> Model {
  tensorgleam.get_sequential_model()
  |> tensorgleam.add_layer_to_model(tensorgleam.get_convolution_2d_layer(
    from_list([config.image_size, config.image_size, 3]),
    64,
    3,
    "relu",
  ))
  |> tensorgleam.add_layer_to_model(
    tensorgleam.get_max_pooling_2d_layer(from_list([2, 2])),
  )
  |> tensorgleam.add_layer_to_model(tensorgleam.get_batch_normalization())
  |> tensorgleam.add_layer_to_model(tensorgleam.get_drop_out(0.2))
  |> tensorgleam.add_layer_to_model(
    tensorgleam.get_convolution_2d_layer_no_input_padding(
      128,
      3,
      "same",
      "relu",
    ),
  )
  |> tensorgleam.add_layer_to_model(
    tensorgleam.get_max_pooling_2d_layer(from_list([2, 2])),
  )
  |> tensorgleam.add_layer_to_model(tensorgleam.get_batch_normalization())
  |> tensorgleam.add_layer_to_model(tensorgleam.get_drop_out(0.2))
  |> tensorgleam.add_layer_to_model(
    tensorgleam.get_convolution_2d_layer_no_input_padding(
      256,
      3,
      "same",
      "relu",
    ),
  )
  |> tensorgleam.add_layer_to_model(
    tensorgleam.get_max_pooling_2d_layer(from_list([2, 2])),
  )
  |> tensorgleam.add_layer_to_model(tensorgleam.get_batch_normalization())
  |> tensorgleam.add_layer_to_model(tensorgleam.get_drop_out(0.2))
  |> tensorgleam.add_layer_to_model(tensorgleam.get_flatten())
  |> tensorgleam.add_layer_to_model(tensorgleam.get_dense(256, "relu"))
  |> tensorgleam.add_layer_to_model(tensorgleam.get_drop_out(0.5))
  |> tensorgleam.add_layer_to_model(tensorgleam.get_dense(class, "softmax"))
  |> tensorgleam.model_compile(
    "adam",
    "categoricalCrossentropy",
    from_list(["accuracy"]),
    config.learning_rate,
  )
}
