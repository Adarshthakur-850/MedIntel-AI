.PHONY: help install data train test run-backend run-frontend docker-up clean

help:
	@echo "MedIntel AI — Development Makefile"
	@echo "Commands:"
	@echo "  make install       Install python dependencies"
	@echo "  make data          Generate synthetic multimodal medical datasets"
	@echo "  make train         Train all ML models (Tabular, Vision, NLP, Fusion)"
	@echo "  make test          Run pytest suite"
	@echo "  make run-backend   Launch FastAPI backend server"
	@echo "  make run-frontend  Launch React frontend dashboard"
	@echo "  make docker-up     Launch full stack via docker-compose"

install:
	py -3.13 -m pip install -e .[dev]

data:
	py -3.13 pipelines/data_generator.py

train:
	py -3.13 ml/tabular/train.py
	py -3.13 ml/vision/train.py

test:
	py -3.13 -m pytest tests/

run-backend:
	py -3.13 -m uvicorn apps.backend.main:app --reload --port 8000

run-frontend:
	cd apps/frontend && npm start

docker-up:
	docker-compose up --build

clean:
	rm -rf *.egg-info .pytest_cache .coverage htmlcov mlruns
