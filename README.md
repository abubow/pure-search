# PureSearch Backend

The backend system for PureSearch, a search engine focused on human-written content.

## Architecture

The backend consists of several microservices:

1. **API Gateway**: Acts as the entry point for all client requests, routing them to the appropriate service
2. **Search API**: Handles search queries and returns results
3. **Content Classifier**: Classifies content as human-written or AI-generated
4. **Content Indexer**: Indexes content for searching
5. **Crawler Service**: Crawls websites to discover and index content

## Features

### Real-time Web Crawling

PureSearch implements real-time web crawling for search queries:

- When a user searches for a query, the system immediately returns any existing results
- Simultaneously, a background task is started to crawl the web for fresh content related to the query
- Results are available in subsequent searches without blocking the user experience
- The search API and crawler service work together to maintain the content index

To use this feature:
- Regular search: `GET /api/v1/search?q=your+search+query`
- Force refresh (trigger crawling even if results exist): `GET /api/v1/search?q=your+search+query&refresh=true`

This approach offers the best of both worlds - fast results from existing data combined with fresh content discovery.

## Technologies Used

- **Go**: For the API Gateway, Search API, and Content Indexer
- **Python**: For the Content Classifier and Crawler Service
- **Elasticsearch**: For full-text search capabilities
- **MongoDB**: For storing content metadata
- **Redis**: For caching
- **RabbitMQ**: For message queues
- **Docker & Docker Compose**: For containerization and orchestration

## Setup and Installation

### Prerequisites

- Docker and Docker Compose
- Make (optional, for using Makefile commands)

### Running the Services

1. Clone the repository
2. Navigate to the backend directory
3. Start all services using Docker Compose:

```bash
docker-compose up -d
```

To stop all services:

```bash
docker-compose down
```

## API Documentation

Each service provides OpenAPI documentation:

- API Gateway: http://localhost:8080/api/v1/docs/
- Search API: http://localhost:8081/swagger/index.html
- Content Classifier: http://localhost:8082/swagger
- Content Indexer: http://localhost:8083/swagger/index.html
- Crawler Service: http://localhost:8084/docs (FastAPI automatic docs)

## Service Endpoints

### API Gateway (Port 8080)

- `GET /api/v1/health`: Health check
- `GET /api/v1/search`: Search for content
- `POST /api/v1/classify`: Classify content
- `POST /api/v1/index`: Index content
- `POST /api/v1/crawl`: Crawl URLs

### Search API (Port 8081)

- `GET /health`: Health check
- `GET /search`: Search for content

### Content Classifier (Port 8082)

- `GET /health`: Health check
- `POST /classify`: Classify content

### Content Indexer (Port 8083)

- `GET /health`: Health check
- `POST /index`: Index content
- `GET /index/:id`: Get indexed content
- `DELETE /index/:id`: Delete indexed content

### Crawler Service (Port 8084)

- `GET /health`: Health check
- `POST /crawl`: Crawl URLs
- `GET /crawl/status/:id`: Get crawl status

## Development

### Prerequisites

- Go 1.21+
- Python 3.11+
- Docker and Docker Compose

### Local Development

Each service can be run independently. Refer to the README in each service directory for specific instructions.

#### Using the Makefile

The Makefile provides shortcuts for common tasks:

```bash
# Start all services
make up

# Stop all services
make down

# View logs
make logs

# Build all services
make build

# Run tests
make test
```

## License

MIT
