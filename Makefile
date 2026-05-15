# Airframe development targets. AI-friendly minimal output by default;
# set VERBOSE=1 for full output.

.PHONY: help install test test-fast test-cov lint typecheck format format-fix check ci clean

UV ?= uv

ifeq ($(VERBOSE),1)
  Q :=
  PYTEST_QUIET :=
  RUFF_QUIET :=
else
  Q := @
  PYTEST_QUIET := -q
  RUFF_QUIET := --quiet
endif

help: ## Show this help message
	$(Q)awk 'BEGIN {FS = ":.*##"; printf "Usage:\n  make \033[36m<target>\033[0m\n\nTargets:\n"} /^[a-zA-Z_-]+:.*##/ { printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2 }' $(MAKEFILE_LIST)

install: ## Install all deps (incl. dev group)
	$(Q)$(UV) sync --all-extras --group dev

test: ## Run the full test suite
	$(Q)$(UV) run pytest $(PYTEST_QUIET)

test-fast: ## Run tests excluding integration markers
	$(Q)$(UV) run pytest $(PYTEST_QUIET) -m "not integration"

test-cov: ## Run tests with coverage
	$(Q)$(UV) run pytest --cov=airframe --cov-report=term-missing

lint: ## Run ruff check
	$(Q)$(UV) run ruff check $(RUFF_QUIET) src tests examples

typecheck: ## Run mypy
	$(Q)$(UV) run mypy src/airframe

format: ## Check formatting (no edits)
	$(Q)$(UV) run ruff format --check src tests examples

format-fix: ## Apply formatting
	$(Q)$(UV) run ruff format src tests examples

check: lint typecheck test ## Lint + typecheck + test

ci: ## CI mode — fail-fast on any error
	$(Q)$(UV) run ruff check src tests examples
	$(Q)$(UV) run ruff format --check src tests examples
	$(Q)$(UV) run mypy src/airframe
	$(Q)$(UV) run pytest

clean: ## Remove build artifacts and caches
	$(Q)rm -rf build dist *.egg-info .pytest_cache .mypy_cache .ruff_cache htmlcov .coverage
	$(Q)find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
