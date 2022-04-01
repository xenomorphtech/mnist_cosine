defmodule MnistCosine.MixProject do
  use Mix.Project

  def project do
    [
      app: :mnist_cosine,
      version: "0.1.0",
      elixir: "~> 1.13.1",
      start_permanent: Mix.env() == :prod,
      deps: deps()
    ]
  end

  # Run "mix help compile.app" to learn about applications.
  def application do
    [
      extra_applications: [:logger, :ssl],
    ]
  end

  # Run "mix help deps" to learn about dependencies.
  defp deps do
    [
      {:axon, "~> 0.1.0-dev", github: "elixir-nx/axon", branch: "main"},
      {:exla, "~> 0.1.0-dev", github: "elixir-nx/nx", sparse: "exla", override: true},
      {:nx, "~> 0.1.0-dev", github: "elixir-nx/nx", sparse: "nx", override: true},
    ]
  end
end