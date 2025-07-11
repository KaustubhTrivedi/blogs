#!/bin/bash

# Production deployment script for Astro blog
set -e

echo "🚀 Starting deployment..."

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
COMPOSE_FILE="../docker-compose.prod.yml"
SERVICE_NAME="blog"

# Functions
log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if Docker is running
if ! docker info >/dev/null 2>&1; then
    log_error "Docker is not running. Please start Docker and try again."
    exit 1
fi

# Check if docker-compose is installed
if ! command -v docker compose >/dev/null 2>&1; then
    log_error "docker compose is not installed. Please install it and try again."
    exit 1
fi

# Pull latest changes (if in git repo)
if [ -d ".git" ]; then
    log_info "Pulling latest changes from git..."
    git pull origin main || log_warning "Could not pull from git. Continuing with local changes."
fi

# Build and deploy
log_info "Building Docker image..."
docker compose -f $COMPOSE_FILE build --no-cache

log_info "Starting services..."
docker compose -f $COMPOSE_FILE up -d

# Wait for health check
log_info "Waiting for service to be healthy..."
timeout 60s bash -c 'until docker-compose -f '$COMPOSE_FILE' ps '$SERVICE_NAME' | grep -q "healthy"; do sleep 2; done'

if [ $? -eq 0 ]; then
    log_info "✅ Deployment successful! Service is healthy."
    log_info "Blog is accessible at http://localhost"
else
    log_error "❌ Deployment failed. Service is not healthy."
    log_info "Checking logs..."
    docker-compose -f $COMPOSE_FILE logs $SERVICE_NAME
    exit 1
fi

# Clean up old images
log_info "Cleaning up old Docker images..."
docker image prune -f

log_info "🎉 Deployment completed successfully!" 