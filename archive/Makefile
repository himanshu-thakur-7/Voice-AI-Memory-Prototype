# Voice AI Prototype — developer entrypoints.
.DEFAULT_GOAL := help
SHELL := /bin/bash

.PHONY: help proto up down logs test test-go test-py lint demo fmt

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

proto: ## Generate gRPC stubs for Go + Python from proto/orchestrator.proto
	@command -v protoc >/dev/null || { echo "install protoc (brew install protobuf)"; exit 1; }
	protoc -I proto \
		--go_out=scheduler/internal/pb --go_opt=paths=source_relative \
		--go-grpc_out=scheduler/internal/pb --go-grpc_opt=paths=source_relative \
		proto/orchestrator.proto
	python -m grpc_tools.protoc -I proto \
		--python_out=backend/app/pb --grpc_python_out=backend/app/pb \
		proto/orchestrator.proto
	@echo "✓ stubs generated"

up: ## Build + run the full local stack
	docker compose up --build

down: ## Tear down the stack
	docker compose down -v

logs: ## Tail all service logs
	docker compose logs -f --tail=100

test: test-go test-py ## Run all tests

test-go: ## Go unit tests (fair-share scheduler, twiml, webhook)
	cd scheduler && go test ./... -race -count=1

test-py: ## Python tests (protocol, chunker, barge-in, contradiction engine)
	cd backend && pytest -q

lint: ## Lint Go + Python
	cd scheduler && go vet ./...
	cd backend && ruff check . && mypy app

fmt: ## Format Go + Python
	cd scheduler && gofmt -w .
	cd backend && ruff format .

demo: ## Replay a recorded µ-law stream against the local /media socket
	cd backend && python -m tests.sim_twilio --print-latency

ui: ## Launch the Ringg-styled test console at http://localhost:8000
	cd backend && uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
