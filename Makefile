.PHONY: \
	black \
	clean \
	clean-build \
	clean-pyc \
	clean-test \
	clean-test-lite \
	coverage \
	develop \
	dist \
	docs \
	format \
	help \
	install \
	isort \
	lint \
	mypy \
	release \
	security \
	servedocs \
	test \
	test-all

.DEFAULT_GOAL := help

define BROWSER_PYSCRIPT
import os, webbrowser, sys

from urllib.request import pathname2url

webbrowser.open("file://" + pathname2url(os.path.abspath(sys.argv[1])))
endef
export BROWSER_PYSCRIPT

define PRINT_HELP_PYSCRIPT
import re, sys

for line in sys.stdin:
	match = re.match(r'^([a-zA-Z_-]+):.*?## (.*)$$', line)
	if match:
		target, help = match.groups()
		print("%-20s %s" % (target, help))
endef
export PRINT_HELP_PYSCRIPT

BROWSER := python -c "$$BROWSER_PYSCRIPT"

black:  ## format python with black
	black tiramisu_brulee
	black tests

clean: clean-build clean-pyc clean-test-lite ## remove all build, test, coverage and Python artifacts

clean-build: ## remove build artifacts
	rm -fr build/
	rm -fr dist/
	rm -fr .eggs/
	find . -name '*.egg-info' -exec rm -fr {} +
	find . -name '*.egg' -exec rm -f {} +

clean-pyc: ## remove Python file artifacts
	find . -name '*.pyc' -exec rm -f {} +
	find . -name '*.pyo' -exec rm -f {} +
	find . -name '*~' -exec rm -f {} +
	find . -name '__pycache__' -exec rm -fr {} +

clean-test: clean-test-lite ## remove test and coverage artifacts
	rm -fr .tox/
	rm -fr .mypy_cache

clean-test-lite:  ## remove test artifacts minus tox
	rm -fr file:.
	rm -fr private
	rm -fr tests/private
	rm -f .coverage
	rm -fr htmlcov/
	rm -fr .pytest_cache
	rm -fr Users/
	rm -fr lightning_logs/
	rm -fr mlruns/

coverage: ## check code coverage quickly with the default Python
	coverage run --source tiramisu_brulee -m pytest
	coverage report -m
	coverage html
	$(BROWSER) htmlcov/index.html

develop: clean ## install the package to the active Python's site-packages
	python setup.py develop

dist: clean ## builds source and wheel package
	python setup.py sdist
	python setup.py bdist_wheel
	ls -l dist

docs: ## generate Sphinx HTML documentation, including API docs
	rm -f docs/tiramisu_brulee.*rst
	rm -f docs/tiramisu_brulee.rst
	rm -f docs/modules.rst
	sphinx-apidoc -o docs/ tiramisu_brulee
	$(MAKE) -C docs clean
	$(MAKE) -C docs html
	$(BROWSER) docs/_build/html/index.html

format:  black isort mypy lint security  ## run various code quality checks and formatters

help:
	@python -c "$$PRINT_HELP_PYSCRIPT" < $(MAKEFILE_LIST)

install: clean ## install the package to the active Python's site-packages
	python setup.py install

isort:  ## format python code with isort
	isort tiramisu_brulee
	isort tests

lint: ## check style with flake8
	flake8 tiramisu_brulee tests

mypy:  ## type-check python with mypy
	mypy tiramisu_brulee
	mypy tests

release: dist ## package and upload a release
	twine upload dist/*

security:  ## run various security checks on the python code
	bandit -r tiramisu_brulee -c pyproject.toml
	bandit -r tests -c pyproject.toml
	snyk test --file=requirements_dev.txt --package-manager=pip --fail-on=all

servedocs: docs ## compile the docs watching for changes
	watchmedo shell-command -p '*.rst' -c '$(MAKE) -C docs html' -R -D .

test: ## run tests quickly with the default Python
	pytest

test-all: ## run tests on every Python version with tox
	tox
