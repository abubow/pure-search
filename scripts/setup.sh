#!/bin/bash

# Setup script for PureSearch backend

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}Starting PureSearch backend setup...${NC}"

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo -e "${RED}Error: Docker is not installed. Please install Docker first.${NC}"
    exit 1
fi

# Check if Docker Compose is installed
if ! command -v docker-compose &> /dev/null; then
    echo -e "${RED}Error: Docker Compose is not installed. Please install Docker Compose first.${NC}"
    exit 1
fi

# Directory of this script
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
# Parent directory (backend root)
PARENT_DIR="$(dirname "$DIR")"

cd "$PARENT_DIR"

echo -e "${YELLOW}Setting up Go module dependencies...${NC}"
# Initialize Go modules for each Go service
for service in api-gateway search-api content-indexer; do
    cd "$PARENT_DIR/services/$service"
    echo -e "${YELLOW}Initializing Go module for $service...${NC}"
    go mod tidy
    if [ $? -ne 0 ]; then
        echo -e "${RED}Error initializing Go module for $service${NC}"
        exit 1
    fi
done

# Return to backend root directory
cd "$PARENT_DIR"

# Setup Python dependencies for the crawler service
echo -e "${YELLOW}Setting up Python dependencies for crawler service...${NC}"
cd "$PARENT_DIR/services/crawler-service"
if [ ! -f "requirements.txt" ]; then
    echo -e "${RED}Error: requirements.txt not found for crawler service${NC}"
    exit 1
fi

# Return to backend root directory
cd "$PARENT_DIR"

echo -e "${YELLOW}Building Docker images...${NC}"
docker-compose build

if [ $? -ne 0 ]; then
    echo -e "${RED}Error building Docker images${NC}"
    exit 1
fi

echo -e "${GREEN}Setup completed successfully!${NC}"
echo -e "${YELLOW}To start the backend services, run:${NC}"
echo -e "${YELLOW}cd $PARENT_DIR && docker-compose up -d${NC}"