.PHONY: all build up down restart logs clean test

# Variables
DOCKER_COMPOSE=docker-compose

all: build up

# Build all services
build:
	$(DOCKER_COMPOSE) build

# Start all services
up:
	$(DOCKER_COMPOSE) up -d

# Stop all services
down:
	$(DOCKER_COMPOSE) down

# Restart all services
restart: down up

# View logs
logs:
	$(DOCKER_COMPOSE) logs -f

# Clean up resources
clean:
	$(DOCKER_COMPOSE) down -v --remove-orphans
	docker system prune -f

# Run tests for all services
test:
	cd services/api-gateway && go test ./... -v
	cd services/search-api && go test ./... -v
	cd services/content-indexer && go test ./... -v
	cd services/content-classifier && pytest -v

# Build and run individual services

api-gateway:
	cd services/api-gateway && go run main.go

search-api:
	cd services/search-api && go run main.go

content-classifier:
	cd services/content-classifier && python app.py

content-indexer:
	cd services/content-indexer && go run main.go

# Generate go.sum files if they don't exist
go-deps:
	cd services/api-gateway && go mod tidy
	cd services/search-api && go mod tidy
	cd services/content-indexer && go mod tidy

# Install Python dependencies
python-deps:
	cd services/content-classifier && pip install -r requirements.txt 