.PHONY: install download run test lint format clean

install:
	pip install -r requirements.txt
	pip install -e .

download:
	python scripts/download_models.py

run:
	uvicorn src.api.server:app --host 0.0.0.0 --port 8000 --reload

test:
	pytest tests/ -v

lint:
	ruff check src/ tests/ scripts/

format:
	black src/ tests/ scripts/
	ruff check --fix src/ tests/ scripts/

eval:
	python scripts/eval_streamlaal.py --testset data/eval/dev.tsv --src en --tgt de

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	rm -rf .pytest_cache .ruff_cache
