import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops


def svd_orthonormal(shape):
    if len(shape) < 2:
        raise RuntimeError("Only shapes of length 2 or more are supported.")
    flat_shape = (shape[0], ops.prod(ms.Tensor(shape[1:])).asnumpy().item())
    a = ops.function.random_func.standard_normal(flat_shape)
    _, u, v = ops.function.linalg_func.svd(a, full_matrices=False)
    q = u if u.shape == flat_shape else v
    q = q.reshape(shape)
    print(f"input shape:{shape}, out shape:{q.shape}")
    return q


def get_activations(model:nn.Cell, layer:nn.Cell, input):
    def activation_hook_handler(model, input, output):
        global buf
        buf = output
    
    handle = layer.register_forward_hook(activation_hook_handler)
    model(input)
    handle.remove()
    return buf


def get_output_shape(model, layer, input):
    def output_hook_handler(model, input, output):
        global buf1
        buf1 = ms.Tensor(output.shape)
    
    handle = layer.register_forward_hook(output_hook_handler)
    model(input)
    handle.remove()
    return buf1

def get_weights(layer):
    weights_and_biases = []
    for param in layer.get_parameters():
        weights_and_biases.append(param)
    return weights_and_biases


def set_weights(layer, weights_and_biases):
    for param,data in zip(layer.get_parameters(),weights_and_biases):
        param.set_data(data)


def LSUVinit(model:nn.Cell, input, verbose=True, margin=0.1, max_iter=10):
    class_to_consider = (nn.Dense, nn.Conv2d)

    needed_variance = 1.0

    layers_initialized = 0
    for name, layer in model.cells_and_names():
        if verbose:
            print(name, layer)
        if not isinstance(layer, class_to_consider):
            continue
        
        if ops.prod(get_output_shape(model, layer, input)[1:]) < 32:
            if verbose:
                print(name, 'too small')
            continue
        if verbose:
            print('LSUV initializing', name)
        
        layers_initialized += 1
        weights_and_biases = get_weights(layer)
        weights_and_biases[0] = svd_orthonormal(weights_and_biases[0].shape)
        set_weights(layer,weights_and_biases)
        activations = get_activations(model, layer, input)
        variance = ops.var(activations)
        iteration = 0
        if verbose:
            print(variance)
        while abs(needed_variance - variance) > margin:
            if ops.abs(ops.sqrt(variance)) < 1e-7:
                break

            weights_and_biases = get_weights(layer)
            weights_and_biases[0] /= ops.sqrt(variance)
            set_weights(layer, weights_and_biases)
            activations = get_activations(model, layer, input)
            variance = ops.var(activations)

            iteration += 1
            if verbose:
                print(variance)
            if iteration >= max_iter:
                break
    if verbose:
        print(f"LSUV: total layers initialized {layers_initialized}")
    return model

class test_model(nn.Cell):
    def __init__(self, auto_prefix=True, flags=None):
        super().__init__(auto_prefix, flags)
        self.network=nn.SequentialCell([
            nn.Conv2d(1,1,3),
            nn.Flatten(),
            nn.Dense(100,32),
            nn.Dense(32,10)
        ])
    def construct(self, input):
        return self.network(input)

if __name__ == "__main__":
    model = test_model()    
    LSUVinit(model,ops.rand([1,1,10,10]))
