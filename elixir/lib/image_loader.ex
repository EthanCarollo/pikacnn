defmodule ImageLoader do
  require Evision
  alias Nx.Tensor

  @input_shape 128

  def load_and_preprocess_image(filename) do
    IO.puts("Loading image: #{filename}")

    case Evision.imread(filename) do
      {:error, reason} ->
        IO.puts("Failed to load image #{filename}: #{reason}")
        nil # Return nil if there is an error loading the image
      img ->
        # Use the Torchx.Backend here : https://alongtheaxon.com/blog/elixir_nx_backends
        img_tensor = Evision.Mat.to_nx(img, {Torchx.Backend, []})
        Nx.divide(img_tensor, 255) # Normalize pixel values between 0 and 1
    end
  end


  def load_images_from_directory(directory_path, max_images_per_label \\ nil) do
    image_paths = []
    labels = []
    label_names = []

    # Use a helper function to accumulate results
    {image_paths, labels, label_names} =
      directory_path
      |> File.ls!()
      |> Enum.reduce({image_paths, labels, label_names}, fn label, {acc_image_paths, acc_labels, acc_label_names} ->
        IO.inspect(label) # Debug output to see the label being processed
        label_path = Path.join(directory_path, label)

        if File.dir?(label_path) do
          label_image_count = 0

          # List images in the label directory and accumulate paths
          {new_image_paths, new_labels, _} =
            label_path
            |> File.ls!()
            |> Enum.reduce({acc_image_paths, acc_labels, label_image_count}, fn filename, {img_acc, lbl_acc, count} ->
              if Path.extname(filename) in [".png", ".jpg", ".jpeg"] do
                if max_images_per_label == nil or count < max_images_per_label do
                  new_img_acc = [Path.join(label_path, filename) | img_acc]
                  new_lbl_acc = [label | lbl_acc]
                  {new_img_acc, new_lbl_acc, count + 1}
                else
                  {img_acc, lbl_acc, count}
                end
              else
                {img_acc, lbl_acc, count} # No change if not a valid image
              end
            end)

          # Update label names only if the label is new
          if label not in acc_label_names do
            new_label_names = [label | acc_label_names]
            {new_image_paths, new_labels, new_label_names}
          else
            {new_image_paths, new_labels, acc_label_names}
          end
        else
          {acc_image_paths, acc_labels, acc_label_names} # No change if not a directory
        end
      end)

    label_names = Enum.sort(label_names) # Sort labels alphabetically
    {image_paths, labels, label_names}
  end


  def images_to_tensor(image_paths, labels, label_names) do
    images =
      image_paths
      |> Enum.map(&load_and_preprocess_image/1) # Preprocess each image
      |> Nx.stack()                            # Stack into a tensor

    # Convert labels to indices based on label_names
    label_indices = Enum.map(labels, &Enum.find_index(label_names, fn name -> name == &1 end))

    {images, Nx.tensor(label_indices)}
  end

  def train_test_split(images, labels, test_size \\ 0.2) do
    # Get the total number of images
    num_images = elem(Nx.shape(images), 0)  # Get the size of the first dimension (number of images)

    # Calculate the split index for training set size
    IO.inspect(num_images)
    split_index = round(num_images * (1.0 - test_size))

    # Ensure split_index is within bounds
    split_index = min(split_index, num_images)

    # Split images and labels into train and test sets
    # TODO : here there is a problem, will fix later cause 😴
    {X_train, X_test} = Nx.split(images, split_index)
    {y_train, y_test} = Nx.split(labels, split_index)

    {X_train, X_test, y_train, y_test}
  end

end
