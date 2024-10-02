defmodule PikaCNN.MixProject do
  use Mix.Project

  def project do
    [
      app: :pika_cnn,
      version: "0.1.0",
      elixir: "~> 1.12",
      start_permanent: Mix.env() == :prod,
      deps: deps()
    ]
  end

  def application do
    [
      extra_applications: [:logger]
    ]
  end

  defp deps do
    [
      {:axon, "~> 0.6.1"},
      {:nx, "~> 0.7.0", optional: true},
      {:evision, "~> 0.1.5"},
      {:exla, "~> 0.7.0", optional: true},
      {:torchx, "~> 0.3"}
    ]
  end
end
