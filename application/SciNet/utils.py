import numpy as np
import mindspore.ops 



def pendulum(t, A0, delta0, k, b, m):
	"""
	Solution x(t) for pendulum differential equation
		mx'' = -kx + bx'
	Returns position at time t

	Parameters:
		- t: time
		- A0: starting amplitude
		- delta0: phase
		- k: spring constant
		- b: damping factor
	"""
	A = 1 - b**2 / (4 * m * k)
	if A < 0:
		return None
	w = np.sqrt(k/m)* np.sqrt(A)
	result = A0 * np.exp( - t * b / (2. * m) ) * np.cos(w * t + delta0)
	return result


def target_loss(pred, answer):
    """
    Computes the mean squared error loss between predicted values and true answers.

    Parameters:
        - pred: tensor of predicted values
        - answer: tensor of true values

    Returns:
        - The mean squared error loss, a scalar value representing the average 
          squared difference between predicted and true values.
    """
    pred = pred[:, 0]
    return mindspore.ops.mean(mindspore.ops.sum(mindspore.ops.square((pred - answer))), axis=0)
