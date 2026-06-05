# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [2.0.0] - 2024-06-04
### Added
- Interactive Gradio demo app (`app.py`) with live digit drawing and probability bar charts.
- GradCAM Saliency Maps to visualize pixel importance (explainability).
- Professional Model Card (`MODEL_CARD.md`) documenting biases, limitations, and evaluation metrics.
- Complete Python packaging (`pyproject.toml`) and `Makefile` for developer tooling.
- Multi-stage `Dockerfile` for containerized deployment of the web app.

## [1.1.0] - 2024-06-04
### Added
- Extended model evaluation: Confusion matrix, per-class classification report (precision/recall/F1).
- Architecture comparison (Small vs Medium vs Large ANN variants).
- CNN benchmark comparison against the best ANN model.
- Hyperparameter sensitivity analysis (batch size and dropout rate sweeps).
- Early stopping with patience of 3 epochs.
- Comprehensive Jupyter Notebook walkthrough (`notebook.ipynb`).
- Extended `pytest` suite.

### Changed
- Replaced basic ANN pipeline with a robust, modular design in `mnist_ann_classifier.py`.
- Updated `README.md` with real measured results, plots, and architecture insights.

## [1.0.0] - Initial Release
### Added
- Basic fully connected neural network for MNIST digit classification.
- Client-side JavaScript inference demo hosted on GitHub Pages (`docs/`).
- Initial test suite and GitHub Actions CI workflow.
