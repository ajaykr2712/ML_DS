#!/bin/bash

# Monitor system resources
while true; do
    echo "=== System Monitor ===" >> logs/system.log
    date >> logs/system.log
    echo "CPU Usage:" >> logs/system.log
    top -bn1 | head -n 20 >> logs/system.log
    echo "Memory Usage:" >> logs/system.log
    free -h >> logs/system.log
    echo "GPU Usage:" >> logs/system.log
    nvidia-smi >> logs/system.log
    sleep 60
done 