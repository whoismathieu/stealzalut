# =============================================================================
# StealZaLut — Automation Makefile
# =============================================================================
# Compatible Linux, macOS et Windows (PyCharm, WSL, Git Bash, MSYS2)
# =============================================================================

UV ?= uv
RUN = $(UV) run --no-sync
PYTHON = $(RUN) python

.PHONY: help install sync format lint typecheck test test-cov check run clean clean-all build

help:
	@echo StealZaLut - Commandes disponibles :
	@echo   make install    - Installe/synchronise les dependances (uv sync --all-extras)
	@echo   make format     - Formate le code avec ruff (format et autocheck fix)
	@echo   make lint       - Verifie le style et les erreurs PEP 8 avec ruff
	@echo   make typecheck  - Verifie le typage statique strict avec mypy
	@echo   make test       - Execute la suite de tests unitaires avec pytest
	@echo   make test-cov   - Execute les tests avec rapport de couverture de code
	@echo   make check      - Lance la validation complete (lint + typecheck + test)
	@echo   make run        - Lance l'application StealZaLut
	@echo   make clean      - Supprime les caches temporaires (__pycache__, pytest, ruff, mypy...)
	@echo   make clean-all  - Supprime les caches et l'environnement virtuel .venv
	@echo   make build      - Construit un executable autonome avec PyInstaller

install: sync

sync:
	$(UV) sync --all-extras

format:
	$(RUN) ruff format .
	$(RUN) ruff check --fix .

lint:
	$(RUN) ruff check .

typecheck:
	$(RUN) mypy src/

test:
	$(RUN) pytest tests/ -v

test-cov:
	$(RUN) pytest --cov=stealzalut --cov-report=term-missing tests/

check: lint typecheck test

run:
	$(UV) run stealzalut

clean:
	@$(PYTHON) -c "import pathlib, shutil; [shutil.rmtree(p, ignore_errors=True) for p in pathlib.Path('.').rglob('__pycache__')]; [shutil.rmtree(p, ignore_errors=True) for p in ['.pytest_cache', '.mypy_cache', '.ruff_cache', 'build', 'dist', 'htmlcov']]; [p.unlink() for p in pathlib.Path('.').glob('*.cube')]; [pathlib.Path(f).unlink(missing_ok=True) for f in ['.coverage', 'coverage.xml']]"
	@echo Nettoyage effectue avec succes.

clean-all: clean
	@$(PYTHON) -c "import shutil; shutil.rmtree('.venv', ignore_errors=True)"
	@echo Environnement virtuel .venv supprime.

build:
	$(RUN) pyinstaller --noconfirm --onedir --windowed --name StealZaLut src/stealzalut/main.py

