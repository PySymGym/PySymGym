POETRY ?= poetry
PYTEST ?= $(POETRY) run pytest
COV_ARGS = --cov=AIAgent --cov=tools --cov-report=term-missing --cov-report=xml

.PHONY: test-unit test-integration test-all test-cov

# Fast unit tier (the default): no network, GPU, .NET or binary fixtures.
test-unit:
	$(PYTEST)

# In-process integration tier (fakes + golden fixtures).
test-integration:
	$(PYTEST) -m integration

# Everything, including the e2e pipelines (needs the built toolchain).
test-all:
	$(PYTEST) -o addopts="--import-mode=importlib"

# Unit tier with a coverage report.
test-cov:
	$(PYTEST) $(COV_ARGS)
