# Makefile for Intelligent Serverless Framework

.PHONY: install test lint docker-build deploy

install:
	pip install -r requirements.txt

test:
	pytest tests/ -v --cov=src

lint:
	black src/ tests/
	flake8 src/ tests/
	mypy src/

docker-build:
	docker build -t intelligent-serverless:latest -f docker/Dockerfile .

deploy:
	kubectl apply -f kubernetes/

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
