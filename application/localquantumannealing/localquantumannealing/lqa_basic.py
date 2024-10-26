import mindspore    
import time
from math import pi
from tqdm import tqdm


class Lqa_basic():
    """
    param couplings: square symmetric numpy or torch array encoding the
                     the problem Hamiltonian
    """
    def __init__(self, couplings):
        super(Lqa_basic, self).__init__()
        self.couplings = couplings
        self.n = couplings.shape[0]
        self.energy = 0.
        self.config = mindspore.ops.zeros((self.n, 1), dtype=mindspore.float32)
        self.weights = (2 * mindspore.ops.rand(self.n) - 1) * 0.1
        self.velocity = mindspore.ops.zeros(self.n)
        self.grad = mindspore.ops.zeros(self.n)

    def forward(self, t, step, mom, g):
        # Implements momentum assisted gradient descent update
        w = mindspore.ops.tanh(self.weights)
        a = 1 - mindspore.ops.tanh(self.weights) ** 2
        # spin x,z values
        z = mindspore.ops.sin(w * pi / 2)
        x = mindspore.ops.cos(w * pi / 2)

        # gradient
        self.grad = ((1 - t) * z + 2 * t * g * mindspore.ops.matmul(self.couplings, z) * x) * a * pi / 2
        # weight update
        self.velocity = mom * self.velocity - step * self.grad
        self.weights = self.weights + self.velocity

    def schedule(self, i, N):
        return i / N

    def energy_ising(self, config):
        # energy of a configuration
        return (mindspore.ops.tensor_dot(mindspore.ops.matmul(self.couplings, config), config, axes = 1)) / 2

    def minimise(self,
                 step=2,  # step size
                 g=1,  # gamma in the article
                 N=200,  # no of iterations
                 mom=0.99,  # momentum
                 f=0.1  # multiplies the weight initialisation
                 ):

        self.weights = (2 * mindspore.ops.rand(self.n) - 1) * f
        self.velocity = mindspore.ops.zeros(self.n)
        self.grad = mindspore.ops.zeros(self.n)

        time0 = time.time()

        for i in tqdm(range(N), desc="Processing"):
            t = self.schedule(i, N)
            self.forward(t, step, mom, g)

        self.opt_time = time.time() - time0
        self.config = mindspore.ops.sign(self.weights)
        self.energy = float(self.energy_ising(self.config))

        print('min energy ' + str(self.energy))
