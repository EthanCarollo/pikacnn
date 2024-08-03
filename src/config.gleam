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

pub fn default_config() {
  Config(
    data_path: "data/pokemon",
    image_size: 150,
    max_image_per_label: 10_000,
    max_label_taken_per_train: 10_000,
    epoch: 50,
    batch_size: 32,
    learning_rate: 0.001,
  )
}
