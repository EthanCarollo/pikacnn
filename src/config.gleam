pub type Config {
  Config(
    data_path: String,
    image_size: Int,
    max_image_per_label: Int,
    max_label_taken_per_train: Int,
    epoch: Int,
    batch_size: Int,
    learning_rate: Float,
  )
}
