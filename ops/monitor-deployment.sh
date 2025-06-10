#!/bin/bash

echo "🚀 Monitoring Clarity Backend deployment on AWS ECS..."
echo "=================================================="

while true; do
    # Get task status
    TASK_STATUS=$(aws ecs describe-tasks \
        --cluster clarity-backend-cluster \
        --tasks arn:aws:ecs:us-east-1:124355672559:task/clarity-backend-cluster/fb6512c1f07e440e8c493f580c9fa35a \
        --query "tasks[0].lastStatus" \
        --output text 2>/dev/null)
    
    CONTAINER_STATUS=$(aws ecs describe-tasks \
        --cluster clarity-backend-cluster \
        --tasks arn:aws:ecs:us-east-1:124355672559:task/clarity-backend-cluster/fb6512c1f07e440e8c493f580c9fa35a \
        --query "tasks[0].containers[0].lastStatus" \
        --output text 2>/dev/null)
    
    echo "$(date '+%Y-%m-%d %H:%M:%S') - Task: $TASK_STATUS, Container: $CONTAINER_STATUS"
    
    if [ "$TASK_STATUS" == "RUNNING" ] && [ "$CONTAINER_STATUS" == "RUNNING" ]; then
        echo "✅ Container is running! Getting public IP..."
        
        # Get public IP
        PUBLIC_IP=$(aws ec2 describe-network-interfaces \
            --network-interface-ids eni-08c2aa957efc11b99 \
            --query "NetworkInterfaces[0].Association.PublicIp" \
            --output text)
        
        echo "🌐 Public IP: $PUBLIC_IP"
        echo "📍 Health endpoint: http://$PUBLIC_IP:8000/health"
        
        # Test health endpoint
        echo "Testing health endpoint..."
        curl -s http://$PUBLIC_IP:8000/health && echo ""
        
        break
    elif [ "$TASK_STATUS" == "STOPPED" ]; then
        echo "❌ Task stopped. Checking reason..."
        STOPPED_REASON=$(aws ecs describe-tasks \
            --cluster clarity-backend-cluster \
            --tasks arn:aws:ecs:us-east-1:124355672559:task/clarity-backend-cluster/fb6512c1f07e440e8c493f580c9fa35a \
            --query "tasks[0].stoppedReason" \
            --output text)
        echo "Stopped reason: $STOPPED_REASON"
        break
    fi
    
    sleep 10
done