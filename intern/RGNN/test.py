import mindspore
import mindspore.nn as nn
import mindspore.ops as ops
import mindspore.common.dtype as mstype
from model_simple import *

def test_sym_sim_gcn_net():
    # Hyperparameters
    num_nodes = 10
    num_features = 16
    num_hiddens = (32,)
    num_classes = 3
    K = 2
    dropout = 0.5
    learn_edge_weight = True
    domain_adaptation = "RevGrad"
    alpha = 0.1  # Coefficient for RevGrad
    lr = 0.01
    epochs = 5

    # Generate synthetic graph data
    x = ops.rand((num_nodes, num_features))  # Node features
    edge_index = ops.randint(0, num_nodes, (2, num_nodes * num_nodes))  # Random edges
    y = ops.randint(0, num_classes, (1,)).astype(mstype.int32)  # Random label

    # PyG Data object

    data = {
        'x': x,
        'edge_index': edge_index,
        'y': y,
        'batch': ops.zeros(num_nodes,dtype = mindspore.int32)
    }

    # Initial edge weight (upper triangular matrix)
    edge_weight = ops.rand(num_nodes * num_nodes)
    model = SymSimGCNNet(
        num_nodes=num_nodes,
        learn_edge_weight=learn_edge_weight,
        edge_weight=edge_weight,
        num_features=num_features,
        num_hiddens=num_hiddens,
        alpha=alpha,
        num_classes=num_classes,
        K=K,
        dropout=dropout,
        domain_adaptation=domain_adaptation
    )

    # Define loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = nn.Adam(model.trainable_params(), learning_rate=lr)

    def get_loss(data,alpha,domain_adaptation):
        out, domain_out = model(data, alpha=alpha)
        loss = criterion(out, data['y'])
        if domain_adaptation == "RevGrad":
            domain_labels = ops.randint(0, 2, (domain_out.shape[0],)).astype(mstype.int32) 
            domain_loss = criterion(domain_out, domain_labels)
            loss += domain_loss
        return loss


    grad_fn = mindspore.value_and_grad(get_loss, None, optimizer.parameters)

    # Training loop
    for epoch in range(epochs):
        loss,grads = grad_fn(data,alpha,domain_adaptation)
        optimizer(grads)
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss}")

    # Evaluation
    pred, _ = model(data)
    print("Prediction value:",pred)
    print("Prediction:", pred.argmax(axis=1))

# Run the test
if __name__ == "__main__":
    test_sym_sim_gcn_net()
