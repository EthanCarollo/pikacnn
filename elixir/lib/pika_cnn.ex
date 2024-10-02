defmodule PikaCNN do
  def run do
    directory_path = "../data/pokemon-128augmented"
    max_images_per_label = 20

    # This should be good
    {image_paths, labels, label_names} = ImageLoader.load_images_from_directory(directory_path, max_images_per_label)

    {images, labels} = ImageLoader.images_to_tensor(image_paths, labels, label_names)
    {x_train, x_test, y_train, y_test} = ImageLoader.train_test_split(images, labels, 0.2)

    IO.inspect("Training images: #{elem(Nx.shape(x_train), 0)}")
    IO.inspect("Testing images: #{elem(Nx.shape(x_test),0)}")
    IO.inspect("Number of classes: #{length(label_names)}")
    IO.inspect("Class names: #{label_names}")
  end
end
