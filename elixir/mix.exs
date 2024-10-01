defmodule YourProject.MixProject do
  use Mix.Project

  def project do
    [
      app: :your_project,
      version: "0.1.0",
      elixir: "~> 1.12",
      start_permanent: Mix.env() == :prod,
      deps: deps()
    ]
  end

  # Here we use `def` instead of `defp`
  def application do
    [
      extra_applications: [:logger]
    ]
  end

  defp deps do
    [
      {:axon, "~> 0.6.1"},
      {:nx, "~> 0.7.0", optional: true},
      {:exla, "~> 0.7.0", optional: true}
    ]
  end
end
