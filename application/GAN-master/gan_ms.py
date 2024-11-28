import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
import numpy as np
import matplotlib.pyplot as plt



BATCH_SIZE = 64
LR_G = 0.0001  # learning rate for generator
LR_D = 0.0001  # learning rate for discriminator
N_IDEAS = 5  # think of this as number of ideas for generating an art work(Generator)
ART_COMPONENTS = 15  # it could be total point G can drew in the canvas
PAINT_POINTS = np.vstack([np.linspace(-1, 1, ART_COMPONENTS) for _ in range(BATCH_SIZE)])


def artist_works():  # painting from the famous artist (real target)
    r = 0.02 * np.random.randn(1, ART_COMPONENTS)
    paintings = np.sin(PAINT_POINTS * np.pi) + r
    paintings = ms.Tensor(paintings,dtype=ms.float32)
    return paintings

G = nn.SequentialCell([
    nn.Dense(N_IDEAS,128),
    nn.ReLU(),
    nn.Dense(128,ART_COMPONENTS),
])

D=nn.SequentialCell([
    nn.Dense(ART_COMPONENTS,128),
    nn.ReLU(),
    nn.Dense(128,1),
    nn.Sigmoid(),
])

opt_G=nn.Adam(G.trainable_params(),learning_rate=LR_G)
opt_D=nn.Adam(D.trainable_params(),learning_rate=LR_D)

plt.ion()  # something about continuous plotting

D_loss_history = []
G_loss_history = []

# 定义梯度操作
grad_op_D = ops.value_and_grad(D, None, D.trainable_params())
grad_op_G = ops.value_and_grad(G, None, G.trainable_params())


criterion = nn.BCEWithLogitsLoss()

for step in range(1000):
    artist_paintings = artist_works()  # real painting from artist
    G_ideas = ops.randn(BATCH_SIZE, N_IDEAS)  # random ideas
    G_paintings = G(G_ideas)  # fake painting from G (random ideas)

    prob_artist0 = D(artist_paintings)  # D try to increase this prob
    prob_artist1 = D(G_paintings)  # D try to reduce this prob
    real_labels=ops.ones_like(prob_artist0)
    fake_labels=ops.zeros_like(prob_artist1)


    loss_d_real=criterion(prob_artist0,real_labels)
    loss_d_fake=criterion(prob_artist1,fake_labels)


    # 计算判别器梯度
    opt_D.requires_grad=False
    (loss, grads_d) = grad_op_D(artist_paintings)
    opt_D(grads_d)


    # 计算生成器梯度
    opt_G.requires_grad=False
    (loss, grads_g) = grad_op_G(G_ideas)
    opt_G(grads_g)

    D_loss = - ops.mean(ops.log(prob_artist0) + ops.log(1. - prob_artist1))
    G_loss = ops.mean(ops.log(1. - prob_artist1))

    D_loss_history.append(D_loss)
    G_loss_history.append(G_loss)


    if step % 50 == 0:  # plotting
        plt.cla()
        plt.plot(PAINT_POINTS[0], G_paintings.asnumpy()[0], c='#4AD631', lw=3, label='Generated painting', )
        plt.plot(PAINT_POINTS[0], np.sin(PAINT_POINTS[0] * np.pi), c='#74BCFF', lw=3, label='upper bound')
        plt.text(-1, 0.75, 'D accuracy=%.2f (0.5 for D to converge)' % prob_artist1.asnumpy().mean(),
                 fontdict={'size': 13})
        plt.text(-1, 0.5, 'D score= %.2f (-1.38 for G to converge)' % -D_loss.asnumpy(), fontdict={'size': 13})
        plt.ylim((-1, 1));
        plt.legend(loc='upper right', fontsize=10);
        plt.draw();
        plt.pause(0.01)

plt.ioff()
plt.show()