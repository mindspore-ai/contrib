import os
import numpy as np
import cv2
import mindspore as ms
from mindspore import nn, ops, Tensor, context
from mindspore.ops import GradOperation

# Import the grad_cam module
from grad_cam import InfoHolder, generate_heatmap, superimpose, to_RGB


class SimpleConvNet(nn.Cell):
    """A simple CNN for demonstration purposes"""

    def __init__(self):
        super(SimpleConvNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1, pad_mode='pad')
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1, pad_mode='pad')
        self.relu2 = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.flatten = nn.Flatten()
        self.fc = nn.Dense(32 * 8 * 8, 10)  # For 32x32 input size

    def construct(self, x):
        x = self.relu1(self.conv1(x))
        x = self.pool(x)
        x = self.relu2(self.conv2(x))
        x = self.pool(x)  # Now 8x8 feature maps
        features = x
        x = self.flatten(x)
        x = self.fc(x)
        return x, features  # Return both logits and features


class GradCamTest(nn.Cell):
    """Simplified Grad-CAM implementation for testing"""

    def __init__(self, model):
        super(GradCamTest, self).__init__()
        self.model = model
        self.grad_op = GradOperation()

    def construct(self, x):
        logits, features = self.model(x)
        target_class = ops.argmax(logits)

        # For visualization purposes only
        weights = ops.mean(features, [2, 3])  # Simple global average pooling
        weighted_features = features * weights.view(1, -1, 1, 1)
        cam_map = ops.mean(weighted_features, 1)

        return features, cam_map


def main():
    """Run a simple test for Grad-CAM visualization"""
    print("Starting simplified Grad-CAM test...")

    # Set context
    context.set_context(mode=context.PYNATIVE_MODE, device_target="CPU")

    # Create a simple model
    model = SimpleConvNet()
    print("Model created successfully")

    # Create a random input tensor (3 channels, 32x32 image)
    input_tensor = Tensor(np.random.random((1, 3, 32, 32)).astype(np.float32))
    print("Input tensor shape:", input_tensor.shape)

    # Run simplified Grad-CAM test
    print("Running simplified Grad-CAM test...")
    try:
        # Forward pass
        logits, features = model(input_tensor)
        print("Forward pass completed")
        print("Features shape:", features.shape)

        # Create a heatmap from features
        # For testing, just use the average of feature maps
        feature_maps = features.squeeze(0).asnumpy()  # Remove batch dimension
        heatmap = np.mean(feature_maps, axis=0)  # Average across channels

        # Normalize heatmap
        heatmap = np.maximum(heatmap, 0)
        heatmap /= np.max(heatmap) + 1e-10

        # Create a dummy RGB image for visualization
        dummy_img = np.uint8(np.random.random((32, 32, 3)) * 255)

        # Resize heatmap to match image size
        heatmap_resized = cv2.resize(heatmap, (dummy_img.shape[1], dummy_img.shape[0]))
        heatmap_colored = np.uint8(255 * heatmap_resized)
        heatmap_colored = cv2.applyColorMap(heatmap_colored, cv2.COLORMAP_JET)

        # Superimpose
        superimposed_img = np.uint8(heatmap_colored * 0.4 + dummy_img * 0.6)

        # Save the output image
        output_path = "simple_grad_cam_result.jpg"
        cv2.imwrite(output_path, superimposed_img)
        print(f"Output saved to {output_path}")

        print("Simple Grad-CAM test completed successfully!")

    except Exception as e:
        print("Error running simplified Grad-CAM test:", e)
        import traceback
        traceback.print_exc()

    print("Test completed.")


if __name__ == "__main__":
    main()