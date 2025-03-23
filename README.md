# PureSearch Backend

A microservices-based backend for PureSearch, designed to serve authentic, human-written content.

## Architecture Overview

PureSearch uses a microservices architecture with services written in Go and Python:

```
                                     ┌─────────────────┐
                                     │                 │
                                     │  API Gateway    │
                                     │  (Go - Gin)     │
                                     │                 │
                                     └─────┬───────────┘
                                           │
                       ┌───────────────────┼───────────────────┐
                       │                   │                   │
              ┌────────▼─────────┐ ┌───────▼──────────┐ ┌─────▼────────────┐
              │                  │ │                  │ │                  │
              │   Search API     │ │ Content          │ │ Content          │
              │   (Go - Gin)     │ │ Classifier       │ │ Indexer          │
              │                  │ │ (Python - Flask) │ │ (Go)             │
              │                  │ │                  │ │                  │
              └────────┬─────────┘ └─────────┬────────┘ └─────────┬────────┘
                       │                     │                    │
                       └─────────┬───────────┘                    │
                                 │                                │
                       ┌─────────▼─────────────────────────┐      │
                       │                                   │      │
                       │           Database                │◄─────┘
                       │     (MongoDB/Elasticsearch)       │
                       │                                   │
                       └───────────────────────────────────┘
```

## Services

### API Gateway
- **Language**: Go with Gin framework
- **Purpose**: Entry point for all client requests, handles routing to appropriate services
- **Features**: Authentication, rate limiting, request/response transformation

### Search API
- **Language**: Go with Gin framework
- **Purpose**: Process search queries and retrieve results
- **Features**: Query processing, results ranking, filtering

### Content Classifier
- **Language**: Python with Flask and ML libraries
- **Purpose**: Analyze and classify content as human or AI-generated
- **Features**: ML-based classification, confidence scoring

### Content Indexer
- **Language**: Go
- **Purpose**: Crawl, parse, and index web content
- **Features**: Content discovery, metadata extraction, index management

## Data Storage

- **Elasticsearch**: For fast full-text search capabilities
- **MongoDB**: For storing content metadata and classification results
- **Redis**: For caching frequent queries

## Communication

- **REST APIs**: For synchronous service-to-service communication
- **RabbitMQ**: For asynchronous processing and event-driven architecture

## Deployment

The services are containerized using Docker and can be deployed using Docker Compose for development and Kubernetes for production.

## Getting Started

### Prerequisites

- Docker and Docker Compose
- Go 1.21+ (for local development)
- Python 3.11+ (for local development)

### Running the Backend

1. Clone the repository:

```bash
git clone https://github.com/abubow/pure-search.git
cd pure-search/backend
```

2. Start all services using Docker Compose:

```bash
docker-compose up -d
```

3. Verify that all services are running:

```bash
docker-compose ps
```

4. Access the API via the API Gateway:

```
http://localhost:8080/api/v1/search?q=your_search_term
```

### Running Individual Services Locally

#### API Gateway

```bash
cd services/api-gateway
go mod download
go run main.go
```

#### Search API

```bash
cd services/search-api
go mod download
go run main.go
```

#### Content Classifier

```bash
cd services/content-classifier
pip install -r requirements.txt
python app.py
```

#### Content Indexer

```bash
cd services/content-indexer
go mod download
go run main.go
```

## API Documentation

### Search API

- `GET /api/v1/search?q=query`: Search for content
  - Query Parameters:
    - `q`: Search query (required)
    - `page`: Page number (default: 1)
    - `per_page`: Results per page (default: 10)

### Content Classifier API

- `POST /api/v1/classify`: Classify text content
  - Request Body:
    ```json
    {
      "text": "Content to classify",
      "url": "https://example.com/source" (optional)
    }
    ```

### Content Indexer API

- `POST /api/v1/index`: Index new content
  - Request Body:
    ```json
    {
      "url": "https://example.com/content",
      "title": "Content Title",
      "description": "Content Description",
      "content": "Full content text"
    }
    ```

## Development

### Adding a New Service

1. Create a new directory in the `services` folder
2. Implement the service using Go or Python
3. Add a Dockerfile
4. Add the service to the docker-compose.yml file
5. Update the API Gateway to route requests to the new service

## Testing

Each service has its own test suite. Run the tests with:

```bash
# For Go services
go test ./...

# For Python services
pytest
```
