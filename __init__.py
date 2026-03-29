[build-system]
requires = ["setuptools>=68.0", "wheel>=0.41"]
build-backend = "setuptools.backends.legacy:build"

[project]
name = "strategic-supply-chain-intelligence"
version = "1.2.0"
description = "Probabilistic multi-horizon demand forecasting with stochastic optimization and executive decision intelligence"
readme = "README.md"
requires-python = "==3.10.*"
license = { text = "MIT" }
keywords = ["supply-chain", "forecasting", "bayesian", "optimization", "mlops"]
classifiers = [
    "Development Status :: 4 - Beta",
    "Intended Audience :: Science/Research",
    "Topic :: Scientific/Engineering :: Artificial Intelligence",
    "Programming Language :: Python :: 3.10",
]

[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-v --tb=short"
filterwarnings = ["ignore::FutureWarning", "ignore::DeprecationWarning"]

[tool.black]
line-length = 100
target-version = ["py310"]

[tool.isort]
profile = "black"
line_length = 100

[tool.mypy]
python_version = "3.10"
warn_return_any = true
warn_unused_configs = true
ignore_missing_imports = true
