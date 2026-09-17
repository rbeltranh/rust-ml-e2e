# Default shell
SHELL := /bin/bash

# Target help
help:
	@echo "Available commands:"
	@echo "  make setup          - Run full environment setup (Python + Rust tooling)"
	@echo "  make install-python - Create Python .venv and install requirements"
	@echo "  make install-rust   - Install cargo-lambda and cross-compilation helpers"
	@echo "  make check          - Verify installed versions of required CLI tools"
	@echo "  make clean          - Remove build artifacts, virtual environments, and cache"

# Run full setup
setup: install-python install-rust check

# Python setup target
install-python:
	@echo "Setting up Python virtual environment..."
	python3 -m venv .venv
	.venv/bin/pip install --upgrade pip
	.venv/bin/pip install -r requirements.txt
	pip freeze > requirements.txt

# Rust setup target
install-rust:
	@echo "Setting up Rust tools..."
	@command -v cargo-lambda >/dev/null 2>&1 || cargo install cargo-lambda
	@if [ "$$(uname)" = "Darwin" ]; then \
		command -v zig >/dev/null 2>&1 || brew install zig; \
	fi

# Train model
train:
	python model_pipeline/train.py

# Verify versions
check:
	@echo "--- Tool Versions ---"
	@.venv/bin/python --version
	@cargo lambda --version
	@gh --version

# Clean build artifacts
clean:
	rm -rf .venv
	rm -rf inference_engine/target
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete