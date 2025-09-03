#!/home/gao/anaconda3/envs/nn/bin/python
"""
SuperPoint Feature Extraction ROS Service
Receives images and returns feature points and descriptors
"""

import rospy
import torch
import cv2
import numpy as np
import time
import sys
import os
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from vio.srv import nnFeatures, nnFeaturesResponse

# Add SuperPoint to Python path
workspace_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
superpoint_path = os.path.join(workspace_path, "third_party", "SuperPoint")
sys.path.insert(0, superpoint_path)

try:
    from superpoint_pytorch import SuperPoint
except ImportError as e:
    rospy.logerr(f"Failed to import SuperPoint: {e}")
    rospy.logerr(f"Please ensure SuperPoint is properly installed at: {superpoint_path}")
    sys.exit(1)

class SuperPointServer:
    """SuperPoint Feature Extraction ROS Service"""

    def __init__(self):
        """Initialize SuperPoint server"""
        # Initialize ROS node
        rospy.init_node('nnFeatureServer', anonymous=True)

        # Get parameters
        self.max_keypoints = rospy.get_param('~max_keypoints', 1024)
        self.use_gpu = rospy.get_param('~use_gpu', True)
        self.confidence_threshold = rospy.get_param('~confidence_threshold', 0.015)

        # Initialize SuperPoint model
        self._init_model()

        # Create OpenCV bridge
        self.bridge = CvBridge()

        # Create ROS service
        self.service = rospy.Service('extract_features', nnFeatures, self.handle_extract_features)

        rospy.loginfo("SuperPoint feature extraction service started")
        rospy.loginfo(f"Service name: /extract_features")
        rospy.loginfo(f"Maximum keypoints: {self.max_keypoints}")
        rospy.loginfo(f"Use GPU: {self.use_gpu}")
        rospy.loginfo(f"Confidence threshold: {self.confidence_threshold}")

    def _init_model(self):
        """Initialize SuperPoint model"""
        try:
            # Check if weights file exists
            weights_path = os.path.join(superpoint_path, 'weights', 'superpoint_v6_from_tf.pth')
            if not os.path.exists(weights_path):
                rospy.logerr(f"Weights file does not exist: {weights_path}")
                rospy.logerr("Please ensure SuperPoint weights file is properly downloaded")
                sys.exit(1)

            # Initialize model
            self.model = SuperPoint(max_num_keypoints=self.max_keypoints)

            # Load weights
            device = torch.device('cuda' if self.use_gpu and torch.cuda.is_available() else 'cpu')
            self.model.load_state_dict(torch.load(weights_path, map_location=device))
            self.model.eval()

            if self.use_gpu and torch.cuda.is_available():
                self.model = self.model.cuda()
                rospy.loginfo("Using GPU for inference")
            else:
                rospy.loginfo("Using CPU for inference")

        except Exception as e:
            rospy.logerr(f"Failed to initialize SuperPoint model: {e}")
            sys.exit(1)

    def preprocess_image(self, cv_image):
        """Preprocess image for SuperPoint"""
        try:
            # Convert to RGB
            image = cv2.cvtColor(cv_image, cv2.COLOR_GRAY2RGB)

            # Convert to torch tensor
            image_tensor = torch.from_numpy(image).float()
            image_tensor = image_tensor.permute(2, 0, 1)  # HWC -> CHW
            image_tensor = image_tensor.unsqueeze(0)      # Add batch dimension
            image_tensor = image_tensor / 255.0           # Normalize to [0,1]

            # Move to correct device
            device = torch.device('cuda' if self.use_gpu and torch.cuda.is_available() else 'cpu')
            return image_tensor.to(device)

        except Exception as e:
            rospy.logerr(f"Image preprocessing failed: {e}")
            raise

    def handle_extract_features(self, req):
        """Handle feature extraction request"""
        start_time = time.time()

        try:
            # Convert ROS image to OpenCV format
            cv_image  = self.bridge.imgmsg_to_cv2(req.image, "mono8")

            # Preprocess image
            image_tensor = self.preprocess_image(cv_image.copy())

            print("tensor shape: ", image_tensor.shape)

            # Extract features
            with torch.no_grad():
                features = self.model({"image": image_tensor})

            # Get results
            keypoints = features["keypoints"][0].cpu().numpy()
            descriptors = features["descriptors"][0].cpu().numpy()
            scores = features["keypoint_scores"][0].cpu().numpy()

            # Filter low confidence keypoints
            valid_mask = scores > self.confidence_threshold
            keypoints = keypoints[valid_mask]
            descriptors = descriptors[valid_mask]
            scores = scores[valid_mask]

            # Calculate processing time
            computation_time = time.time() - start_time

            # Create response
            response = nnFeaturesResponse()
            response.keypoints = keypoints.flatten().astype(np.float32).tolist()
            response.descriptors = descriptors.flatten().astype(np.float32).tolist()
            response.scores = scores.astype(np.float32).tolist()
            response.num_keypoints = len(keypoints)
            response.computation_time = computation_time

            rospy.loginfo(f"Successfully extracted {len(keypoints)} feature points, time: {computation_time:.3f}s")

            return response

        except Exception as e:
            rospy.logerr(f"Feature extraction failed: {e}")
            # Return empty response
            response = nnFeaturesResponse()
            response.keypoints = []
            response.descriptors = []
            response.scores = []
            response.num_keypoints = 0
            response.computation_time = time.time() - start_time
            return response

    def spin(self):
        rospy.spin()


if __name__ == '__main__':
    try:
        server = SuperPointServer()
        server.spin()
    except rospy.ROSInterruptException:
        rospy.loginfo("SuperPoint server stopped")
    except Exception as e:
        rospy.logerr(f"SuperPoint server failed to run: {e}")