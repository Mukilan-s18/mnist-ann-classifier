.PHONY: install train demo test lint clean help

PYTHON := /opt/homebrew/bin/python3.11
PIP    := /opt/homebrew/bin/pip3.11

help:  ## Show this help message
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
	awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'
	@echo ""

install:  ## Install all dependencies
	$(PIP) install -r requirements.txt
	$(PIP) install gradio
	@echo "✅ All dependencies installed"

train:  ## Train the model and generate all plots
	$(PYTHON) mnist_ann_classifier.py
	@echo "✅ Training complete"

demo:  ## Launch the interactive Gradio demo locally
	$(PYTHON) app.py
	@echo "🚀 Demo running at http://localhost:7860"

test:  ## Run the test suite
	/opt/homebrew/bin/pytest tests/ -v
	@echo "✅ All tests passed"

lint:  ## Run code quality checks with ruff
	/opt/homebrew/bin/python3.11 -m ruff check .
	@echo "✅ Lint passed"

notebook:  ## Launch the Jupyter notebook
	/opt/homebrew/bin/jupyter notebook notebook.ipynb

clean:  ## Remove generated files (keeps saved model)
	rm -f confusion_matrix.png training_history.png sample_predictions.png
	rm -f architecture_comparison.png benchmark_comparison.png hyperparameter_analysis.png
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
	@echo "✅ Cleaned generated files"

clean-all: clean  ## Remove everything including saved model
	rm -rf saved_model/
	@echo "✅ Full clean complete"
