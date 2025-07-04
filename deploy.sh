#!/bin/bash

# Sepsis Sentinel Deployment Script
# Automated deployment for production environment

set -e  # Exit on any error

echo "🚀 Starting Sepsis Sentinel Deployment..."

# Configuration
PROJECT_NAME="sepsis-sentinel"
DOCKER_REGISTRY=${DOCKER_REGISTRY:-"localhost:5000"}
VERSION=${VERSION:-"1.0.0"}
ENVIRONMENT=${ENVIRONMENT:-"production"}

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check prerequisites
check_prerequisites() {
    log_info "Checking prerequisites..."
    
    # Check Docker
    if ! command -v docker &> /dev/null; then
        log_error "Docker is not installed"
        exit 1
    fi
    
    # Check Docker Compose
    if ! command -v docker-compose &> /dev/null; then
        log_error "Docker Compose is not installed"
        exit 1
    fi
    
    # Check NVIDIA Docker (for GPU support)
    if ! docker run --rm --gpus all nvidia/cuda:11.8-base-ubuntu20.04 nvidia-smi &> /dev/null; then
        log_warning "NVIDIA Docker not available - GPU support disabled"
        export GPU_SUPPORT=false
    else
        log_success "NVIDIA Docker available - GPU support enabled"
        export GPU_SUPPORT=true
    fi
    
    log_success "Prerequisites check completed"
}

# Build Docker images
build_images() {
    log_info "Building Docker images..."
    
    # Build API image
    log_info "Building API image..."
    docker build -t ${DOCKER_REGISTRY}/${PROJECT_NAME}-api:${VERSION} \
        -f deploy/api/Dockerfile deploy/api/
    
    # Build Dashboard image
    log_info "Building Dashboard image..."
    docker build -t ${DOCKER_REGISTRY}/${PROJECT_NAME}-dashboard:${VERSION} \
        -f deploy/dashboard/Dockerfile deploy/dashboard/
    
    # Tag as latest
    docker tag ${DOCKER_REGISTRY}/${PROJECT_NAME}-api:${VERSION} \
        ${DOCKER_REGISTRY}/${PROJECT_NAME}-api:latest
    docker tag ${DOCKER_REGISTRY}/${PROJECT_NAME}-dashboard:${VERSION} \
        ${DOCKER_REGISTRY}/${PROJECT_NAME}-dashboard:latest
    
    log_success "Docker images built successfully"
}

# Setup data directories
setup_data_directories() {
    log_info "Setting up data directories..."
    
    mkdir -p data/raw
    mkdir -p data/processed
    mkdir -p data/models
    mkdir -p logs
    mkdir -p deploy/triton/models
    
    # Set permissions
    chmod 755 data logs deploy/triton/models
    
    log_success "Data directories created"
}

# Validate configuration files
validate_configs() {
    log_info "Validating configuration files..."
    
    # Check required config files
    required_configs=(
        "configs/schema.yaml"
        "configs/train_tft_gnn.yaml"
        "configs/infer.yaml"
        "deploy/docker-compose.yml"
        "deploy/nginx/nginx.conf"
    )
    
    for config in "${required_configs[@]}"; do
        if [[ ! -f "$config" ]]; then
            log_error "Required config file missing: $config"
            exit 1
        fi
    done
    
    # Validate YAML syntax
    if command -v python &> /dev/null; then
        python -c "
import yaml
import sys
try:
    with open('configs/schema.yaml', 'r') as f:
        yaml.safe_load(f)
    with open('configs/train_tft_gnn.yaml', 'r') as f:
        yaml.safe_load(f)
    print('YAML validation passed')
except Exception as e:
    print(f'YAML validation failed: {e}')
    sys.exit(1)
"
    fi
    
    log_success "Configuration validation completed"
}

# Initialize database and storage
init_storage() {
    log_info "Initializing storage systems..."
    
    # Start Redis for caching
    docker run -d --name sepsis-redis-temp \
        -p 6379:6379 redis:7-alpine \
        redis-server --appendonly yes || true
    
    # Wait for Redis to be ready
    sleep 5
    
    # Test Redis connection
    if docker exec sepsis-redis-temp redis-cli ping | grep -q PONG; then
        log_success "Redis initialized successfully"
    else
        log_warning "Redis initialization failed - continuing without cache"
    fi
    
    # Clean up temp container
    docker stop sepsis-redis-temp && docker rm sepsis-redis-temp || true
}

# Deploy model artifacts
deploy_models() {
    log_info "Deploying model artifacts..."
    
    # Create Triton model repository structure
    mkdir -p deploy/triton/models/sepsis_sentinel/1
    
    # Check if ONNX model exists
    if [[ ! -f "models/sepsis_sentinel.onnx" ]]; then
        log_warning "ONNX model not found - using placeholder"
        # Create placeholder model file
        touch deploy/triton/models/sepsis_sentinel/1/model.onnx
    else
        cp models/sepsis_sentinel.onnx deploy/triton/models/sepsis_sentinel/1/model.onnx
        log_success "ONNX model deployed"
    fi
    
    # Deploy model configuration
    if [[ -f "deploy/triton/config.pbtxt" ]]; then
        cp deploy/triton/config.pbtxt deploy/triton/models/sepsis_sentinel/config.pbtxt
    fi
    
    log_success "Model artifacts deployed"
}

# Start services
start_services() {
    log_info "Starting Sepsis Sentinel services..."
    
    cd deploy/
    
    # Set environment variables
    export VERSION=$VERSION
    export ENVIRONMENT=$ENVIRONMENT
    export GPU_SUPPORT=$GPU_SUPPORT
    
    # Stop any existing services
    docker-compose down --remove-orphans 2>/dev/null || true
    
    # Start services based on environment
    if [[ "$ENVIRONMENT" == "production" ]]; then
        log_info "Starting production services..."
        docker-compose -f docker-compose.yml up -d
    else
        log_info "Starting development services..."
        docker-compose -f docker-compose.yml up -d \
            --scale triton=1 \
            --scale api=1 \
            --scale dashboard=1
    fi
    
    cd ..
    
    log_success "Services started"
}

# Health checks
perform_health_checks() {
    log_info "Performing health checks..."
    
    # Wait for services to start
    sleep 30
    
    # Check API health
    max_attempts=30
    attempt=1
    
    while [[ $attempt -le $max_attempts ]]; do
        if curl -f http://localhost:8000/health &> /dev/null; then
            log_success "API health check passed"
            break
        else
            log_info "Waiting for API... (attempt $attempt/$max_attempts)"
            sleep 5
            ((attempt++))
        fi
    done
    
    if [[ $attempt -gt $max_attempts ]]; then
        log_error "API health check failed"
        return 1
    fi
    
    # Check Dashboard health
    if curl -f http://localhost:5000 &> /dev/null; then
        log_success "Dashboard health check passed"
    else
        log_warning "Dashboard health check failed"
    fi
    
    # Check Triton health
    if curl -f http://localhost:8000/v2/health/ready &> /dev/null; then
        log_success "Triton health check passed"
    else
        log_warning "Triton health check failed"
    fi
    
    log_success "Health checks completed"
}

# Setup monitoring
setup_monitoring() {
    log_info "Setting up monitoring and logging..."
    
    # Create log directories
    mkdir -p logs/api logs/dashboard logs/triton
    
    # Setup log rotation
    cat > logs/logrotate.conf << EOF
logs/*.log {
    daily
    missingok
    rotate 7
    compress
    notifempty
    create 0644 root root
}
EOF
    
    log_success "Monitoring setup completed"
}

# Display deployment summary
show_deployment_summary() {
    log_success "🎉 Sepsis Sentinel Deployment Completed!"
    
    echo ""
    echo "📋 Deployment Summary:"
    echo "====================="
    echo "Environment: $ENVIRONMENT"
    echo "Version: $VERSION"
    echo "GPU Support: $GPU_SUPPORT"
    echo ""
    echo "🌐 Service URLs:"
    echo "Dashboard: http://localhost (main interface)"
    echo "API: http://localhost/api (REST endpoints)"
    echo "API Docs: http://localhost/docs (Swagger UI)"
    echo "Metrics: http://localhost/metrics (monitoring)"
    echo ""
    echo "🔧 Management Commands:"
    echo "View logs: docker-compose -f deploy/docker-compose.yml logs -f"
    echo "Stop services: docker-compose -f deploy/docker-compose.yml down"
    echo "Restart: docker-compose -f deploy/docker-compose.yml restart"
    echo ""
    echo "📊 Quick Health Check:"
    echo "curl http://localhost/health"
    echo ""
    echo "🚨 For production use, ensure:"
    echo "- SSL certificates are configured"
    echo "- Environment variables are set securely"
    echo "- Monitoring and alerting is configured"
    echo "- Regular backups are scheduled"
    echo ""
}

# Cleanup on failure
cleanup_on_failure() {
    log_error "Deployment failed - cleaning up..."
    
    cd deploy/ 2>/dev/null || true
    docker-compose down --remove-orphans 2>/dev/null || true
    cd .. 2>/dev/null || true
    
    # Remove temporary containers
    docker stop sepsis-redis-temp 2>/dev/null || true
    docker rm sepsis-redis-temp 2>/dev/null || true
    
    log_info "Cleanup completed"
    exit 1
}

# Set trap for cleanup on failure
trap cleanup_on_failure ERR

# Main deployment flow
main() {
    echo "🚀 Sepsis Sentinel Deployment Script v${VERSION}"
    echo "================================================"
    
    check_prerequisites
    validate_configs
    setup_data_directories
    build_images
    init_storage
    deploy_models
    start_services
    perform_health_checks
    setup_monitoring
    show_deployment_summary
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --version)
            VERSION="$2"
            shift 2
            ;;
        --environment)
            ENVIRONMENT="$2"
            shift 2
            ;;
        --registry)
            DOCKER_REGISTRY="$2"
            shift 2
            ;;
        --skip-build)
            SKIP_BUILD=true
            shift
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --version VERSION      Set deployment version (default: 1.0.0)"
            echo "  --environment ENV      Set environment (default: production)"
            echo "  --registry REGISTRY    Set Docker registry (default: localhost:5000)"
            echo "  --skip-build          Skip Docker image building"
            echo "  --help                Show this help message"
            exit 0
            ;;
        *)
            log_error "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Skip build if requested
if [[ "$SKIP_BUILD" == "true" ]]; then
    build_images() {
        log_info "Skipping Docker image build"
    }
fi

# Run main deployment
main
