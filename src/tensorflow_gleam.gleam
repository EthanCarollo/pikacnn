import gleam/io
import tensorgleam

pub fn main() {
  io.println("Hello from tensorflow_gleam!")
  tensorgleam.log_model(tensorgleam.get_sequential_model())
}
