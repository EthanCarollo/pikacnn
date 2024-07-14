import gleam/io
import model/construct.{construct_model}
import tensorgleam

pub fn main() {
  let _model =
    construct_model(150)
    |> tensorgleam.model_summary
}
