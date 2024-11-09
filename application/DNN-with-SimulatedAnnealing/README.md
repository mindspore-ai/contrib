# Output Range Analysis for Deep Neural Networks with Simulated Annealing

## ✨ Introduction
> This code is a mindspore implementation of DNN-with-SimulatedAnnealing which is avaliable at https://github.com/Nicerova7/output-range-analysis-for-deep-neural-networks-with-simulated-annealing.

Our approach addresses the lack of local geometric information and high non-linearity in DNNs, making it versatile across various architectures, especially Residual Neural Networks (ResNets). We present a straightforward, implementation-friendly algorithm that avoids restrictive assumptions about network architecture. Through theoretical analysis and experimental evaluations, including tests on the Ackley function, we demonstrate our algorithm’s effectiveness in navigating complex, non-convex surfaces and accurately estimating DNN output ranges.

## 📈 Reproduction results

```python

## Model

model = DeeperResidualNN()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

## You need to set up your data here

num_epochs = 1000
for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(x)
    loss = criterion(outputs, y)
    loss.backward()
    optimizer.step()
    if epoch % 100 == 0:
        print(f'Epoch [{epoch}/{num_epochs}], Loss: {loss.item():.4f}')



## Use function

initial_solution = torch.tensor([-3.8, -3.8], dtype=torch.float32).unsqueeze(0)  # Starting point
max_temperature = 1000     # Initial Temperature
min_temperature = 1
cooling_rate = 0.99            # Cooling Rate
num_iterations = 1000          # Number of iterations

l = torch.Tensor([-4, -4])
u = torch.Tensor([4, 4])
interval = (l, u)
sigma = 0.1

best_solution, best_value = simulated_annealing(model, initial_solution, max_temperature, min_temperature, cooling_rate, num_iterations, interval, sigma)
print(f"Optimal Solution: {best_value:.4f}, Prediction = {best_solution}")

```

## 📝 Citation

[WIP]


## ✒️ Authors:

* Helder Rojas (h.rojas-molina23@imperial.ac.uk)
* Nilton Rojas-Vales (nrojasv@uni.pe)


## 📃 License

No license yet.