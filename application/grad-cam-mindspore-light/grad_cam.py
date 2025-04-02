import cv2
import numpy as np
import mindspore as ms
from mindspore import nn, ops


class InfoHolder():
    def __init__(self, heatmap_layer):
        self.gradient = None
        self.activation = None
        self.heatmap_layer = heatmap_layer
        self.grad_handle = None

    def set_gradients(self, grad):
        self.gradient = grad

    def set_activation(self, activation):
        self.activation = activation


def generate_heatmap(weighted_activation):
    raw_heatmap = ops.mean(weighted_activation, 0)
    heatmap_np = raw_heatmap.asnumpy()
    heatmap = np.maximum(heatmap_np, 0)
    heatmap /= np.max(heatmap) + 1e-10
    return heatmap


def superimpose(input_img, heatmap):
    img = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)
    heatmap = cv2.resize(heatmap, (img.shape[0], img.shape[1]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    superimposed_img = np.uint8(heatmap * 0.6 + img * 0.4)
    pil_img = cv2.cvtColor(superimposed_img, cv2.COLOR_BGR2RGB)
    return pil_img


def to_RGB(tensor):
    tensor_np = tensor.asnumpy()
    tensor_np = (tensor_np - np.min(tensor_np))
    tensor_np = tensor_np / (np.max(tensor_np) + 1e-10)
    image_binary = np.transpose(tensor_np, (1, 2, 0))
    image = np.uint8(255 * image_binary)
    return image


def grad_cam(model, input_tensor, heatmap_layer, truelabel=None):
    # Create info holder
    info = InfoHolder(heatmap_layer)

    # MindSpore approach to get activations
    def forward_hook(cell, inputs, outputs):
        info.set_activation(outputs)
        return outputs

    # Register hook for activation
    hook_handle = info.heatmap_layer.register_forward_hook(forward_hook)

    # Prepare the input
    input_expanded = ops.expand_dims(input_tensor, 0)

    # Set the model to gradient calculation mode
    model.set_train(False)
    model.set_grad(True)

    # Define grad function
    grad_fn = ms.value_and_grad(lambda x: model(x)[0][truelabel], grad_position=0, weights=None)

    # Forward and backward pass
    output, grads = grad_fn(input_expanded)
    if truelabel is None:
        truelabel = ops.argmax(output[0])
        # Recalculate gradients with the correct label
        output, grads = grad_fn(input_expanded)

    # Get gradients for the target layer
    for name, cell in model.cells_and_names():
        if cell is info.heatmap_layer:
            # Get gradients for this layer
            info.set_gradients(grads)
            break

    # Remove the hook
    hook_handle.remove()

    # Process gradients and activations
    activation = info.activation.squeeze(0)
    weights = ops.mean(info.gradient[0], [0, 2, 3])

    # Calculate weighted activation
    weighted_activation = ms.Tensor(np.zeros(activation.shape), ms.float32)
    for idx in range(len(weights)):
        weighted_activation[idx] = weights[idx] * activation[idx]

    # Generate heatmap
    heatmap = generate_heatmap(weighted_activation)
    input_image = to_RGB(input_tensor)

    return superimpose(input_image, heatmap)