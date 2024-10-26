import mindspore
import mindspore.nn as nn
from tqdm import tqdm
import time
from math import pi

class Lqa(nn.Cell):
    """
    param couplings: square symmetric numpy or torch array encoding the
                     the problem Hamiltonian
    """

    def __init__(self, couplings):
        super(Lqa, self).__init__()

        self.couplings = couplings
        self.n = couplings.shape[0]
        self.energy = 0.
        self.config = mindspore.ops.zeros((self.n, 1))
        self.min_en = 9999.
        self.min_config = mindspore.ops.zeros((self.n, 1))
        self.weights = mindspore.Parameter(mindspore.ops.zeros(self.n),requires_grad=True)

    def schedule(self, i, N):
        #annealing schedule
        return i / N

    def energy_ising(self, config):
        # ising energy of a configuration
        return (mindspore.ops.tensor_dot(mindspore.ops.matmul(self.couplings, config), config, axes = 1)) / 2
    

    def energy_full(self, t, g):
        # cost function value
        config = mindspore.ops.tanh(self.weights)*pi/2
        ez = self.energy_ising(mindspore.ops.sin(config))
        ex = mindspore.ops.cos(config).sum()

        return (t*ez*g- (1-t)*ex)


    def get_energy(self,i,N,g):
        t = self.schedule(i, N)
        energy = self.energy_full(t,g)
        return energy

    def minimise(self,
                 step=1,  # learning rate
                 N=200,  # no of iterations
                 g=1.,
                 f=1.):
        self.weights = mindspore.Parameter( (2 * mindspore.ops.rand([self.n]) - 1) * f,requires_grad=True)
        time0 = time.time()
        optimizer = nn.Adam([self.weights], learning_rate=step)


        grad_fn = mindspore.value_and_grad(self.get_energy, None, optimizer.parameters)

        for i in tqdm(range(N), desc="Processing"):
            _,grads = grad_fn(i,N,g)
            optimizer(grads)

        self.opt_time = time.time() - time0
        self.config = mindspore.ops.sign(self.weights)
        self.energy = float(self.energy_ising(self.config))

        return self.energy


