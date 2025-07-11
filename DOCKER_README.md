# Docker Production Setup for Astro Blog

This directory contains a production-ready Docker setup for the Astro blog project, optimized for simplicity and ease of deployment.

## 🏗️ Architecture

The setup uses a multi-stage Docker build:

1. **Builder Stage**: Uses Node.js to build the Astro static site
2. **Production Stage**: Uses a lightweight Node.js static file server to serve the files

## 📁 Files Overview

- `Dockerfile` - Multi-stage production Docker build
- `.dockerignore` - Excludes unnecessary files from Docker build context
- `docker-compose.yml` - Development/staging deployment
- `docker-compose.prod.yml` - Production deployment with resource limits
- `scripts/deploy.sh` - Automated deployment script

## 🚀 Quick Start

### Development/Staging Deployment

```bash
# Build and run (accessible on http://localhost:3001)
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

### Production Deployment

```bash
# Option 1: Use the deployment script (recommended)
./scripts/deploy.sh

# Option 2: Manual deployment
docker-compose -f docker-compose.prod.yml up -d

# Option 3: With resource monitoring
docker-compose -f docker-compose.prod.yml up -d --remove-orphans
```

## 🔧 Configuration

### Environment Variables

The following environment variables can be configured:

- `NODE_ENV=production` - Sets the Node.js environment

### Port Configuration

- **Development**: `localhost:3001`
- **Production**: `localhost:80` (mapped to container port 3001)
- **Internal container port**: `3001`

### Resource Limits (Production)

- **CPU**: 0.5 cores (limit), 0.1 cores (reserved)
- **Memory**: 512MB (limit), 128MB (reserved)

## 🛡️ Security Features

- **Non-root user**: Container runs as user `astro` (UID 1001)
- **Lightweight server**: Uses `serve` package with minimal attack surface

## 🚀 Performance Optimizations

- **Static file serving**: Efficient static file serving with `serve`
- **Optimized Docker layers**: Multi-stage build for smaller image size
- **Minimal dependencies**: Only includes what's needed for serving static files

## 📊 Monitoring & Health Checks

### Docker Health Checks

Built-in health checks verify service availability:

```bash
# Check service health
docker-compose ps

# View health check logs
docker inspect <container_id> | grep -A 10 Health
```

### Log Management

- **Log rotation**: Automatic log rotation (max 10MB, 3 files)
- **Structured logging**: JSON format for easy parsing
- **Access logs**: Server access logs for traffic analysis

## 🔄 Deployment Workflow

### Automated Deployment (Recommended)

```bash
# Make script executable (first time only)
chmod +x scripts/deploy.sh

# Deploy
./scripts/deploy.sh
```

The script will:
1. ✅ Check Docker and docker-compose availability
2. 🔄 Pull latest git changes (if in repo)
3. 🏗️ Build Docker image with no cache
4. 🚀 Start services
5. 🏥 Wait for health check to pass
6. 🧹 Clean up old Docker images

### Manual Deployment

```bash
# Build image
docker-compose -f docker-compose.prod.yml build --no-cache

# Start services
docker-compose -f docker-compose.prod.yml up -d

# Check status
docker-compose -f docker-compose.prod.yml ps
```

## 🐛 Troubleshooting

### Common Issues

1. **Port already in use**:
   ```bash
   # Change port in docker-compose.yml
   ports:
     - "8080:3001"  # Use different host port
   ```

2. **Permission denied**:
   ```bash
   # Fix script permissions
   chmod +x scripts/deploy.sh
   ```

3. **Service not healthy**:
   ```bash
   # Check logs
   docker-compose logs blog
   
   # Check if serve is running
   docker-compose exec blog ps aux
   ```

4. **Build failures**:
   ```bash
   # Clear Docker cache
   docker system prune -a
   
   # Rebuild without cache
   docker-compose build --no-cache
   ```

### Useful Commands

```bash
# View real-time logs
docker-compose logs -f blog

# Access container shell
docker-compose exec blog sh

# Check running processes
docker-compose exec blog ps aux

# View container resource usage
docker stats

# Clean up unused Docker resources
docker system prune

# Remove all stopped containers and unused images
docker system prune -a
```

## 📈 Scaling & Load Balancing

For high-traffic scenarios, consider:

1. **Multiple instances**:
   ```bash
   docker-compose -f docker-compose.prod.yml up -d --scale blog=3
   ```

2. **Add reverse proxy**: Add nginx or traefik in front for load balancing

3. **CDN integration**: Configure CDN for static assets

## 🎯 Best Practices

1. **Regular updates**: Keep base images updated
2. **Security scanning**: Regularly scan images for vulnerabilities
3. **Backup strategy**: Implement regular backups
4. **Monitoring**: Set up proper monitoring and alerting
5. **Resource monitoring**: Monitor CPU, memory, and disk usage

## 📝 Customization

### Docker Configuration

Edit `Dockerfile` to customize:
- Node.js version
- Static server package (could use `http-server` instead of `serve`)
- Security settings

### Static Server Options

The `serve` package supports various options. You can modify the CMD in Dockerfile:

```dockerfile
# Examples of serve options
CMD ["serve", "-s", "dist", "-l", "3001", "--no-clipboard"]  # Disable clipboard
CMD ["serve", "-s", "dist", "-l", "3001", "--cors"]         # Enable CORS
```

## 📋 Production Checklist

Before deploying to production:

- [ ] Update `astro.config.mjs` with correct site URL
- [ ] Configure proper SSL certificates (via reverse proxy)
- [ ] Set up monitoring and alerting
- [ ] Configure backup strategy
- [ ] Test health check endpoints
- [ ] Verify resource limits are appropriate
- [ ] Configure log aggregation
- [ ] Set up CI/CD pipeline

## 🤝 Contributing

To improve this Docker setup:

1. Test changes thoroughly
2. Update documentation
3. Maintain security best practices
4. Consider performance implications 