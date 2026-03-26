#!/bin/bash
BACKUP_DIR="./backups"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
mkdir -p $BACKUP_DIR

# PostgreSQL dump
docker exec trading_db pg_dump -U trader trading > $BACKUP_DIR/trading_$TIMESTAMP.sql

# Model volume backup
docker run --rm -v model_data:/data -v $BACKUP_DIR:/backup alpine tar czf /backup/models_$TIMESTAMP.tar.gz -C /data .

# Optional: remote sync (uncomment and adjust)
# rclone copy $BACKUP_DIR remote:backups/

# Keep last 30 days
find $BACKUP_DIR -name "*.sql" -mtime +30 -delete
find $BACKUP_DIR -name "*.tar.gz" -mtime +30 -delete
