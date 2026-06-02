# Contributing to MNIST ANN Classifier

Thank you for your interest in contributing! This document details the process for setting up your local environment and running verification tests.

## Local Development Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/YOUR-USERNAME/mnist-ann-classifier.git
   cd mnist-ann-classifier
   ```

2. **Create and activate a virtual environment:**
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate  # On Windows use: .venv\Scripts\activate
   ```

3. **Install dependencies (including development packages):**
   ```bash
   pip install -r requirements.txt
   ```

## Code Quality & Verification

We use `ruff` to format and check code quality, and `pytest` for tests. Please ensure all commands pass before submitting pull requests.

### 1. Running Unit Tests
To verify shapes, normalization limits, and network compile setups:
```bash
pytest tests/
```

### 2. Running Linting
To check syntax and formatting conforms to python guidelines:
```bash
ruff check .
```

## Running the Web Demo Locally

To load the interactive drawing interface on your local machine:

1. **Train the network to generate the weights file:**
   ```bash
   python mnist_ann_classifier.py
   ```
   This will export `web/model_weights.json`.

2. **Serve the `web` folder:**
   Since the web app uses `fetch` to read the weights database, modern browsers restrict loading JSON directly from the local file system (`file://` protocol) due to CORS security rules. You must run a local server:
   ```bash
   python -m http.server 8000 --directory web
   ```
   Now navigate to `http://localhost:8000` in your web browser.

## Submitting Pull Requests

1. Create a descriptive feature branch (`git checkout -b feature/cool-new-thing`).
2. Implement your changes and add matching tests if applicable.
3. Verify all tests pass (`pytest` and `ruff check`).
4. Commit your changes with clear, concise messages.
5. Push to your fork and submit a Pull Request.
