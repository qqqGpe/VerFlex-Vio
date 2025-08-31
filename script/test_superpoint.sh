#!/bin/bash

# SuperPoint Feature Extraction Test Script
# This script demonstrates how to use the SuperPoint ROS service

echo "=== SuperPoint Feature Extraction Test ==="

# Check if image path is provided
if [ $# -eq 0 ]; then
    echo "Usage: $0 <image_path>"
    echo "Example: $0 /path/to/your/image.jpg"
    exit 1
fi

IMAGE_PATH=$1

# Check if image file exists
if [ ! -f "$IMAGE_PATH" ]; then
    echo "Error: Image file does not exist: $IMAGE_PATH"
    exit 1
fi

echo "Testing with image: $IMAGE_PATH"

# Method 1: Start server and client separately
echo ""
echo "Method 1: Starting server and client separately"
echo "1. Starting SuperPoint server..."
roslaunch vio superpoint_server.launch &
SERVER_PID=$!

# Wait for server to start
sleep 3

echo "2. Running C++ client..."
rosrun vio superpoint_client "$IMAGE_PATH"

# Kill server
kill $SERVER_PID

echo ""
echo "Method 2: Using combined launch file"
echo "Launching both server and client together..."
roslaunch vio superpoint_test.launch image_path:="$IMAGE_PATH"

echo ""
echo "Test completed!"
echo ""
echo "To use the service in your own code:"
echo "1. Start the server: roslaunch vio superpoint_server.launch"
echo "2. Call the service: /extract_features"
echo "3. Service type: vio/nnFeatures"
echo ""
echo "Service request: sensor_msgs/Image"
echo "Service response:"
echo "  - keypoints: float32[] (Nx2 array)"
echo "  - descriptors: float32[] (NxD array)"
echo "  - scores: float32[] (N array)"
echo "  - num_keypoints: int32"
echo "  - computation_time: float32"
