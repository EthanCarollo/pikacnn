import config.{image_size}
import custom_layer/custom_layer
import gleam/javascript/array.{from_list}
import tensorgleam.{type Model}

pub fn construct_model(class: Int) -> Model {
  tensorgleam.get_sequential_model()
  |> tensorgleam.add_layer_to_model(tensorgleam.get_convolution_2d_layer(
    from_list([image_size, image_size, 3]),
    64,
    4,
    "relu",
  ))
  |> tensorgleam.add_layer_to_model(
    tensorgleam.get_max_pooling_2d_layer(from_list([2, 2])),
  )
  |> tensorgleam.add_layer_to_model(tensorgleam.get_drop_out(0.25))
  |> tensorgleam.add_layer_to_model(
    tensorgleam.get_convolution_2d_layer_no_input_padding(
      128,
      4,
      "same",
      "relu",
    ),
  )
  |> tensorgleam.add_layer_to_model(
    tensorgleam.get_convolution_2d_layer_no_input(256, 4, "relu"),
  )
  |> tensorgleam.add_layer_to_model(
    tensorgleam.get_max_pooling_2d_layer(from_list([2, 2])),
  )
  |> tensorgleam.add_layer_to_model(tensorgleam.get_drop_out(0.25))
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
