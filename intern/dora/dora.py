import mindspore
from mindspore import Tensor,context
import mindspore.nn as nn
import mindspore.ops as ops
import mindspore.dataset as ds


# This layer is dropped into your pre-trained PyTorch model where nn.Linear is used
class DoRALayer(nn.Cell):
    def __init__(self, d_in, d_out, rank=4, weight=None, bias=None):
        super().__init__()

        if weight is not None:
            self.weight = mindspore.Parameter(weight, requires_grad=False)
        else:
            self.weight = mindspore.Parameter(Tensor(d_out, d_in), requires_grad=False)

        if bias is not None:
            self.bias = mindspore.Parameter(bias, requires_grad=False)
        else:
            self.bias = mindspore.Parameter(Tensor(d_out), requires_grad=False)

        # m = Magnitude column-wise across output dimension
        self.m = mindspore.Parameter(self.weight.norm(ord=2, dim=0, keepdim=True))
        
        std_dev = 1 / ops.sqrt(Tensor(rank,mindspore.float32))
        self.lora_A = mindspore.Parameter(ops.randn(d_out, rank)*std_dev)
        self.lora_B = mindspore.Parameter(ops.zeros((rank, d_in)))

    def construct(self, x):
        lora = ops.matmul(self.lora_A, self.lora_B)
        adapted = self.weight + lora
        column_norm = adapted.norm(ord=2, dim=0, keepdim=True)
        norm_adapted = adapted / column_norm
        calc_weights = self.m * norm_adapted
        return ops.dense(x, calc_weights, self.bias)


class SimpleModel(nn.Cell):
    def __init__(self, input_dim, output_dim):
        super(SimpleModel, self).__init__()
        self.layer1 = nn.Dense(input_dim, output_dim)

    def construct(self, x):
        x = self.layer1(x)
        return x

# Generating synthetic data
def generate_data(num_samples=100, input_dim=10):
    X = ops.randn(num_samples, input_dim)
    y = ops.sum(X, dim=1, keepdim=True).asnumpy()  # Simple relationship for demonstration
    X = X.asnumpy()
    return X, y




# Training function
def train(model, criterion, optimizer, data_loader, epochs=5):
    #def loss function
    def get_loss(inputs,targets):
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        return loss 
    grad_fn = mindspore.value_and_grad(get_loss, None, optimizer.parameters)
    for epoch in range(epochs):
        for inputs, targets in data_loader:
            loss,grads = grad_fn(inputs,targets)
            optimizer(grads)
        print(f"Epoch {epoch+1}, Loss: {loss.item()}")



def replace_linear_with_dora(model):
    """
    This function is intended exclusively for replacing layer in SimpleModel.
    If you need to switch to a different model, please modify this function.
    For reference, see the implementation at:
    https://github.com/catid/dora/blob/main/dora.py
    """
    for name, cell in model.cells_and_names():
        _,(name,cell) = cell.cells_and_names()
        d_in = cell.in_channels
        d_out = cell.out_channels
        setattr(model, name, DoRALayer(d_out=d_out, d_in=d_in, weight=cell.weight.data.clone(), bias=cell.bias.data.clone()))
        break


def print_model_parameters(model):
    total_params = sum(p.size for p in model.get_parameters())
    trainable_params = sum(p.size for p in model.trainable_params())
    
    print(f"Total Parameters: {total_params}")
    print(f"Trainable Parameters: {trainable_params}")

# Main script
if __name__ == "__main__":
    context.set_context(mode=context.GRAPH_MODE, device_target="CPU")
    mindspore.set_seed(100)
    input_dim, output_dim = 10, 1
    model = SimpleModel(input_dim, output_dim)
    criterion = nn.MSELoss()
    optimizer = nn.AdamWeightDecay(model.trainable_params(), learning_rate=6e-4,eps = 1e-8,weight_decay=0.01)

    X, y = generate_data(num_samples=1000, input_dim=input_dim)

    data_loader = ds.NumpySlicesDataset((X, y), column_names=["inputs", "outputs"])
    data_loader = data_loader.batch(batch_size=64, drop_remainder=True)  
    data_loader = data_loader.shuffle(buffer_size=len(X)) 

    print_model_parameters(model)

    train(model, criterion, optimizer, data_loader, epochs=100)

    #Evaluate the model
    inputs, targets = next(iter(data_loader))

    predictions = model(inputs)
    loss = criterion(predictions, targets)
    print(f"Final Evaluation Loss: {loss.item()}")

    replace_linear_with_dora(model)
    
    print_model_parameters(model)
    # Continue training with the Dora model
    optimizer = nn.AdamWeightDecay(model.trainable_params(), learning_rate=6e-4,eps = 1e-8,weight_decay=0.01)
    print("Continuing training with DoRA layers...")
    train(model, criterion, optimizer, data_loader, epochs=5)  # Continue training

    # Evaluate the model
    
    inputs, targets = next(iter(data_loader))
    predictions = model.construct(inputs)
    loss = criterion(predictions, targets)
    print(f"Final (DoRA) Evaluation Loss: {loss.item()}")
