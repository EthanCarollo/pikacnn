defmodule CNNModel do
  alias Axon.Loop

  def build_model(input_shape, num_classes) do
    Axon.input({nil, input_shape, input_shape, 3})
    |> Axon.conv(32, kernel_size: {3, 3}, activation: :relu)
    |> Axon.max_pool(kernel_size: {2, 2})
    |> Axon.conv(64, kernel_size: {3, 3}, activation: :relu)
    |> Axon.max_pool(kernel_size: {2, 2})
    |> Axon.conv(128, kernel_size: {3, 3}, activation: :relu)
    |> Axon.max_pool(kernel_size: {2, 2})
    |> Axon.conv(256, kernel_size: {3, 3}, activation: :relu)
    |> Axon.max_pool(kernel_size: {2, 2})
    |> Axon.conv(512, kernel_size: {3, 3}, activation: :relu)
    |> Axon.max_pool(kernel_size: {2, 2})
    |> Axon.flatten()
    |> Axon.dense(2048, activation: :relu)
    |> Axon.dropout(rate: 0.5)
    |> Axon.dense(num_classes, activation: :softmax)
  end

  def train_model(model, training_data, num_classes) do
    optimizer = Axon.Optimizers.adam(0.001)

    # Compile and train the model
    Loop.trainer(model, :categorical_cross_entropy, optimizer)
    |> Loop.metric(:accuracy)
    |> Loop.run(training_data, epochs: 10, compiler: EXLA)
  end

  defp prepare_data do
    # Load or prepare your dataset here
    {training_images, training_labels} = # load or generate data...
    training_data = {Nx.tensor(training_images), Nx.tensor(training_labels)}
    training_data
  end

  def start(input_shape, num_classes) do
    model = build_model(input_shape, num_classes)
    training_data = prepare_data()
    train_model(model, training_data, num_classes)
  end
end
