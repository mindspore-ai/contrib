"""
This data implement paper's algorithm

:paper author: hxq
:code author: hxq
:code convert: shy
"""

import random
import os
import sys
import numpy as np
from tqdm import tqdm
from scipy import sparse
import mindspore.nn.probability.distribution as msd
import mindspore.numpy as mnp
from mindspore import dtype as mstype
from mindspore.train.serialization import save_checkpoint as save_model
from mindspore import ParameterTuple, Parameter, context, load_checkpoint,\
load_param_into_net, Tensor, nn, ops
from utils import l2_regularizer, set_rng_seed
from dataloader import load_data, sampler
from min_norm_solvers_numpy import MinNormSolver
from metrics import hit_precision_recall_ndcg_k


context.set_context(mode=context.PYNATIVE_MODE)

ARG = {
    'data': './netflix',
    'mode': 'trn',
    'MOO': True,
    'lagrangian_method': True,
    'logdir': './runs/',
    'seed': 98765,
    'epoch': 30,
    'batch': 500,
    'learning rate': 1e-3,
    'lr_lagrange_factor': 1e-4,
    'rg': 0.0,
    'keep': 0.5,
    'beta': 1.0,
    'tau': 0.1,
    'std': 0.075,
    'kfac': 1,
    'dfac': 5,
    'nogb': False,
    'normalization_type': 'none',
    'constant': 40
}


BATCH_SIZE_VAD = BATCH_SIZE_TEST = 3*ARG['batch']


class Helper(nn.Cell):
    """
    This class is a solve grad class
    :author: shy
    """
    def __init__(self, group_ph, input_ph, items, cores):
        """
        :param group_ph:
        :param input_ph:
        :param items:
        :param cores:
        """
        super(Helper, self).__init__()
        self.group_ph = group_ph
        self.keep_prob_ph = 1.0
        self.mul = ops.MatMul(transpose_b=True)
        self.input_ph = input_ph
        self.items = items
        self.cores = cores
        self.anneal_ph = 1

    def construct(self, weights_1, bias_1, weights_2, bias_2):
        """
        :param weights_1:
        :param bias_1:
        :param weights_2:
        :param bias_2:
        :return:
        """
        recon_loss_users, kl_users = self.forward_pass(weights_1, bias_1, weights_2, bias_2)
        recon_loss_group_list = self.get_multi_group_loss_helper(recon_loss_users)
        kl_group_list = self.get_multi_group_loss_helper(kl_users)
        multi_loss_list = [
            recon_loss_group_list[i] + self.anneal_ph * kl_group_list[i] for i in range(GROUP_NUM)
        ]
        return multi_loss_list

    def forward_pass(self, weights_1, bias_1, weights_2, bias_2,):
        """
        :param weights_1:
        :param bias_1:
        :param weights_2:
        :param bias_2:
        :return:
        """
        is_training_ph = 1
        items = l2_regularizer(self.items)
        cores = l2_regularizer(self.cores)
        cates_logits = l2_regularizer(
            ops.MatMul(transpose_b=True)(items, cores) / ARG['tau'], axis=0
        )
        # cates_logits = ops.MatMul(transpose_b=True)(self.items, self.cores) / ARG['tau']
        if ARG['nogb']:
            cates = nn.Softmax(axis=1)(cates_logits)
        else:
            # cates_dist = msd.Categorical(cates_logits)
            # cates_sample = cates_dist.sample()
            cates_sample = Tensor(np.ones(cates_logits.shape), dtype=mstype.float32)
            cates_mode = nn.Softmax(axis=1)(cates_logits)
            cates = (is_training_ph * cates_sample +
                     (1 - is_training_ph) * cates_mode)
        return self.forward_pass_two(
            (weights_1, bias_1), (weights_2, bias_2), cates, items)

    def forward_pass_two(self, w_b_1, w_b_2, cates, items):
        """
        :param w_b_1:
        :param w_b_2:
        :param cates:
        :param items:
        :return:
        """
        probs, kl_users = None, None
        for k in range(ARG['kfac']):
            # q-network
            mu_k, std_k, kl_k_users = self.q_graph_k(
                w_b_1, w_b_2, self.input_ph * mnp.reshape(cates[:, k], (1, -1)))
            epsilon = msd.Normal(mean=np.random.random(), sd=np.random.random()).prob(std_k)
            if k == 0:
                kl_users = kl_k_users
            else:
                kl_users += kl_k_users
            # p-network
            # z_k = tf.nn.l2_normalize(z_k, axis=1)
            # logits_k = tf.matmul(z_k, items, transpose_b=True)
            logits_k = ops.MatMul(transpose_b=True)(
                l2_regularizer(mu_k + epsilon * std_k), items)
            # probs_k = tf.exp(logits_k / ARG.tau)
            probs_k = mnp.exp(logits_k / ARG['tau']) * mnp.reshape(cates[:, k], (1, -1))
            probs = (probs_k if (probs is None) else (probs + probs_k))
        recon_loss_users = mnp.sum(- mnp.log(nn.Softmax()(mnp.log(probs))) * self.input_ph, axis=1)
        return recon_loss_users, kl_users

    def get_multi_group_loss_helper(self, loss_users):
        """
        :param loss_users:
        :return:
        """
        loss_group_list = []
        for i in range(GROUP_NUM):
            loss_i = mnp.mean(ops.Gather()(
                loss_users, Tensor(np.where(mnp.equal(self.group_ph, i))[0]), 0))
            loss_group_list.append(loss_i)
        return loss_group_list

    def q_graph_k(self, w_b_1, w_b_2, x_input):
        """
        :param w_b_1:
        :param w_b_2:
        :param x_input:
        :return:
        """
        weights_1, bias_1 = w_b_1
        weights_2, bias_2 = w_b_2
        hidden_layer = l2_regularizer(x_input, 1)

        hidden_layer = nn.Dropout(keep_prob=self.keep_prob_ph)(hidden_layer)

        hidden_layer = self.mul(hidden_layer, weights_1) + bias_1

        hidden_layer = mnp.tanh(hidden_layer)

        hidden_layer = self.mul(hidden_layer, weights_2) + bias_2
        mu_q = hidden_layer[:, :ARG['dfac']]  # a^k_u
        # mu_q = tf.nn.l2_normalize(mu_q, axis=1)
        mu_q = l2_regularizer(mu_q)
        lnvarq_sub_lnvar0 = -hidden_layer[:, ARG['dfac']:]  # b^k_u
        std0 = ARG['std']
        # std_q = tf.exp(0.5 * lnvarq_sub_lnvar0) * std0
        std_q = mnp.exp(0.5 * lnvarq_sub_lnvar0) * std0
        kl_users = mnp.sum(
            0.5 * (-lnvarq_sub_lnvar0 + mnp.exp(lnvarq_sub_lnvar0) - 1.), axis=1
        )

        return mu_q, std_q, kl_users


class MOO(nn.Cell):
    """
    This class is a main model
    """
    def __init__(self, num_items):
        """
        :param num_items:
        """
        super(MOO, self).__init__()

        kfac, dfac = ARG['kfac'], ARG['dfac']

        self.lam = ARG['rg']
        self.constant = ARG['constant']
        self.lagrange_factor = Parameter(Tensor([np.random.random()], dtype=mstype.float32))
        self.weights_q_1 = nn.Dense(num_items, dfac)
        self.weights_q_2 = nn.Dense(dfac, 2 * dfac)

        self.items = Parameter(
            Tensor(np.random.random((num_items, dfac)), dtype=mstype.float32)
        )
        self.cores = Parameter(
            Tensor(np.random.random((kfac, dfac)), dtype=mstype.float32)
        )
        self.keep_prob_ph = ARG['keep']
        self.anneal_ph = 0
        self.is_training_ph = 1
        self.tsk_weights_ph = Tensor(np.ones(4), mstype.float32)
        self.counter = 0
    def get_grad_loss(self, x_input, x_group):
        """
        :param x_input:
        :param x_group:
        :return:
        """
        logits, recon_loss_users, kl_users = self.forward_pass(x_input)

        recon_loss_group_list = self.get_multi_group_loss(recon_loss_users, x_group)
        kl_group_list = self.get_multi_group_loss(kl_users, x_group)
        # reg_var = np.random.random()
        multi_loss_list = [
            recon_loss_group_list[i] +
            self.anneal_ph * kl_group_list[i] for i in range(GROUP_NUM)
        ]
        return logits, multi_loss_list

    def construct(self, x_input, x_group):
        """
        :param x_input:
        :param x_group:
        :return:
        """
        logits, recon_loss_users, kl_users = self.forward_pass(x_input)
        print(logits)
        recon_loss_group_list = self.get_multi_group_loss(recon_loss_users, x_group)
        kl_group_list = self.get_multi_group_loss(kl_users, x_group)

        multi_loss_list = [
            recon_loss_group_list[i] +
            self.anneal_ph * kl_group_list[i] for i in range(GROUP_NUM)
        ]
        fairness_violation = ops.ReLU()(
            self.fair_loss(multi_loss_list) - self.constant
        )
        specific_loss = mnp.mean(recon_loss_users) + \
                        self.anneal_ph * mnp.mean(kl_users) + np.random.random()

        proxy_loss = np.random.random() + \
                     mnp.sum(mnp.stack(multi_loss_list, axis=0) * self.tsk_weights_ph)

        train_op_lagrange = -(self.lagrange_factor * fairness_violation)

        train_op_share = proxy_loss + self.lagrange_factor * fairness_violation

        train_op_specific = specific_loss + \
                            mnp.sum(mnp.stack(multi_loss_list, axis=0))

        return train_op_specific, train_op_share, train_op_lagrange

    def forward_pass(self, input_ph):
        """
        :param input_ph:
        :return:
        """
        items = l2_regularizer(self.items)
        cores = l2_regularizer(self.cores)
        cates_logits = l2_regularizer(
            ops.MatMul(transpose_b=True)(items, cores) / ARG['tau'], axis=0
        )
        if ARG['nogb']:
            cates = nn.Softmax(axis=1)(cates_logits)
        else:
            # cates_dist = msd.Categorical(cates_logits)
            # cates_sample = cates_dist.sample()
            cates_sample = Tensor(np.ones(cates_logits.shape), dtype=mstype.float32)
            cates_mode = nn.Softmax(axis=1)(cates_logits)
            cates = (self.is_training_ph * cates_sample +
                     (1 - self.is_training_ph) * cates_mode)
        probs, kl_users = None, None
        for k in range(ARG['kfac']):
            # cates_k = tf.reshape(cates[:, k], (1, -1))
            cates_k = mnp.reshape(cates[:, k], (1, -1))
            # q-network
            x_k = input_ph * cates_k
            mu_k, std_k, kl_k_users = self.q_graph_k(x_k)
            # epsilon = tf.random_normal(tf.shape(std_k))
            epsilon = msd.Normal(
                mean=np.random.random(), sd=np.random.random()
            ).prob(std_k)
            z_k = mu_k + self.is_training_ph * epsilon * std_k
            if k == 0:
                kl_users = kl_k_users
            else:
                kl_users += kl_k_users
            # p-network
            # z_k = tf.nn.l2_normalize(z_k, axis=1)
            z_k = l2_regularizer(z_k)
            # logits_k = tf.matmul(z_k, items, transpose_b=True)
            logits_k = ops.MatMul(transpose_b=True)(z_k, items)
            # probs_k = tf.exp(logits_k / ARG.tau)
            probs_k = mnp.exp(logits_k / ARG['tau'])
            probs_k = probs_k * cates_k  # (num_users * num_items)*(1, num_items)
            probs = (probs_k if (probs is None) else (probs + probs_k))
        logits = mnp.log(probs)
        softmax_logits = nn.Softmax()(logits)
        recon_loss_users = mnp.sum(- mnp.log(softmax_logits) * input_ph, axis=1)
        return logits, recon_loss_users, kl_users

    def q_graph_k(self, x_input):
        """
        :param x_input:
        :return:
        """

        hidden_layer = l2_regularizer(x_input, 1)

        hidden_layer = nn.Dropout(keep_prob=self.keep_prob_ph)(hidden_layer)

        hidden_layer = self.weights_q_1(hidden_layer)

        hidden_layer = mnp.tanh(hidden_layer)

        hidden_layer = self.weights_q_2(hidden_layer)

        mu_q = hidden_layer[:, :ARG['dfac']]  # a^k_u
        # mu_q = tf.nn.l2_normalize(mu_q, axis=1)
        mu_q = l2_regularizer(mu_q)
        lnvarq_sub_lnvar0 = -hidden_layer[:, ARG['dfac']:]  # b^k_u
        std0 = ARG['std']
        # std_q = tf.exp(0.5 * lnvarq_sub_lnvar0) * std0
        std_q = mnp.exp(0.5 * lnvarq_sub_lnvar0) * std0
        # Trick: KL is constant w.r.category. to mu_q after we normalize mu_q.
        kl_users = mnp.sum(
            0.5 * (-lnvarq_sub_lnvar0 + mnp.exp(lnvarq_sub_lnvar0) - 1.), axis=1
        )

        return mu_q, std_q, kl_users

    def get_multi_group_loss(self, loss_users, group_ph):
        """
        :param loss_users:
        :param group_ph:
        :return:
        """
        self.counter += 1
        loss_group_list = []
        for i in range(GROUP_NUM):
            loss_i = mnp.mean(ops.Gather()(
                loss_users, Tensor(np.where(mnp.equal(group_ph, i))[0]), 0))
            loss_group_list.append(loss_i)
        return loss_group_list

    def fair_loss(self, loss_group_list):
        """
        :param loss_group_list:
        :return:
        """
        self.counter += 1
        loss_mean = mnp.mean(mnp.stack(loss_group_list, axis=0))
        fair_constraints = None
        for i, loss in enumerate(loss_group_list):
            if i == 0:
                fair_constraints = ops.ReLU()(loss - loss_mean)
            else:
                fair_constraints = ops.ReLU()(loss - loss_mean)
        return fair_constraints


class WithLossCell(nn.Cell):
    """
    This class is a loss class
    """
    def __init__(self, backbone, loss_fn):
        """
        :param backbone:
        :param loss_fn:
        """
        super(WithLossCell, self).__init__(auto_prefix=False)
        self._backbone = backbone
        self._loss_fn = loss_fn

    def construct(self, data, label, op_=0):
        """
        :param data:
        :param label:
        :param op:
        :return:
        """
        out = self._backbone(data[0], data[1])
        return self._loss_fn(out[op_], label)

    @property
    def backbone_network(self):
        """
        :return:
        """
        return self._backbone


class TrainOneStepCell(nn.Cell):
    """
    This class is a train class
    """
    def __init__(self, network, optimizer):
        """参数初始化"""
        super(TrainOneStepCell, self).__init__(auto_prefix=False)
        self.network = network
        # 使用tuple包装weight
        self.weights = ParameterTuple(network.trainable_params())
        self.optimizer = optimizer
        # 定义梯度函数
        self.grad = ops.GradOperation(get_by_list=True)

    def construct(self, data):
        """构建训练过程"""
        loss = self.network.construct(data[0], data[1])
        grads = self.grad(self.network, self.weights)(data[0], data[1])

        # _grads = [Tensor(np.zeros(i.shape), dtype=mstype.float32) for i in grads]
        # _grads[op] = grads[op]
        # 为反向传播设定系数
        # sens = ops.Fill()(ops.DType()(loss), ops.Shape()(loss), self.sens)
        return loss, self.optimizer(grads)

def validation_fun(n_vad, vae, train_net, train_r30_list, r30_train_group_list):
    """[summary]

    Args:
        n_vad ([type]): [description]
        vae ([type]): [description]
        train_net ([type]): [description]
        train_r30_list ([type]): [description]
        r30_train_group_list ([type]): [description]
    """
    r30_list = []
    x_group_list = []
    recon_vad_loss_list = []
    r30_vad_group_list = []

    st_idx_dict = dict((group, 0) for group in range(GROUP_NUM))
    for bnum, st_idx in enumerate(range(0, n_vad, BATCH_SIZE_VAD)):
        print(bnum, st_idx)
        x_group, x_input, x_te, st_idx_dict = \
            sampler(TRAIN_DATA, st_idx_dict,
                    BATCH_SIZE_VAD, VAD_GROUP_DICT, VAD_DATA)

        train_set = sparse.lil_matrix(x_input).rows
        max_train_count = np.max([len(category) for category in train_set])
        vad_item = sparse.lil_matrix(x_te).rows

        if sparse.isspmatrix(x_input):
            x_input = x_input.toarray()
        x_input = x_input.astype('float32')

        # vae.input_ph = x_input
        vae.is_training_ph = 0
        # vae.group_ph = Tensor(x_group, dtype=mstype.float32)
        x_input = Tensor(x_input, dtype=mstype.float32)
        x_group = Tensor(x_group, dtype=mstype.float32)
        label = Tensor(0, dtype=mstype.float32)
        train_net((x_input, x_group), label)
        logits_var, multi_loss_list = vae.get_grad_loss(x_input, x_group)

        pred_val = logits_var
        recon_vad_loss_list.append(multi_loss_list[:GROUP_NUM])

        pred_val[x_input.nonzero()] = -np.inf
        _, _, recall_k, _ = \
            hit_precision_recall_ndcg_k(train_set, vad_item,
                                        np.squeeze(np.array(pred_val)),
                                        max_train_count, k=30, ranked_tag=False)

        r30_list.extend(recall_k)
        x_group_list.extend(x_group)

    r30_list = np.array(r30_list)
    # recon_vad_loss_list = np.array(recon_vad_loss_list)
    recall = r30_list.mean()
    x_group_list = np.array(x_group_list)

    for i in range(GROUP_NUM):
        train_r30_tmp = (train_r30_list[x_group_list == i]).mean()
        if not np.isnan(train_r30_tmp):
            r30_train_group_list.append(train_r30_tmp)

        r30_tmp = (r30_list[x_group_list == i]).mean()
        if not np.isnan(r30_tmp):
            r30_vad_group_list.append(r30_tmp)
    return recall
def preprocess(validation, best_epoch=None):
    """[summary]

    Args:
        validation ([type]): [description]
        best_epoch ([type], optional): [description]. Defaults to None.

    Returns:
        [type]: [description]
    """
    set_rng_seed(ARG['seed'])
    if not VALIDATION:
        train_data = TRAIN_DATA + VAD_DATA
        for key in TRAIN_GROUP_DICT:
            TRAIN_GROUP_DICT[key] = list(
                set(TRAIN_GROUP_DICT[key]).union(set(VAD_GROUP_DICT[key]))
            )

    num = train_data.shape[0]  # train-users, train_data是一个csr_matrix
    n_items = train_data.shape[1]
    # idxlist = list(range(num))
    n_vad = VAD_DATA.shape[0]

    num_batches = int(np.ceil(float(num) / ARG['batch']))
    total_anneal_steps = num_batches

    vae = MOO(n_items)

    if validation:
        epochs = ARG['epoch']
    else:
        epochs = int(1.2 * best_epoch)

    best_recall = 0.0
    # best_epoch = 0
    # best_grad_norm = np.inf
    update_count = 0.0
    # scale = np.array([1 / NUM_TASK] * NUM_TASK)

    loss = nn.L1Loss()
    optimizer = nn.Adam(params=vae.trainable_params(), learning_rate=0.01)
    net_with_criterion = WithLossCell(vae, loss)
    train_net = TrainOneStepCell(vae, optimizer)
    label = Tensor(-5000, dtype=mstype.float32)
    cal_grads = ops.GradOperation(get_all=True)
    data = {
        'num': num,
        'epochs': epochs,
        'n_vad': n_vad,
        'total_anneal_steps': total_anneal_steps,
        'best_recall': best_recall,
        'update_count': update_count,
        'net_with_criterion': net_with_criterion,
        'train_net': train_net,
        'label': label,
        'cal_grads': cal_grads,
        'vae': vae,
    }
    return data
def get_loss(vae, x_group, x_input, train_net, cal_grads):
    """[summary]

    Args:
        vae ([type]): [description]
        x_group ([type]): [description]
        x_input ([type]): [description]
        train_net ([type]): [description]
        cal_grads ([type]): [description]

    Returns:
        [type]: [description]
    """

    for i in range(1):
        print(i)
        train_net((x_input, x_group))
    # print(net_with_criterion((x_input, x_group), label, op=0))
    helper_net = Helper(x_group, x_input, vae.cores, vae.items)
    grads_net = cal_grads(helper_net, helper_net.trainable_params())(
        Tensor(vae.weights_q_1.weight.data),
        Tensor(vae.weights_q_1.bias.data),
        Tensor(vae.weights_q_2.weight.data),
        Tensor(vae.weights_q_2.bias.data)
    )

    grads_net = [[j.asnumpy() for j in grads_net]for i in range(4)]
    # recon_train_loss_list.append(results[NUM_TASK:(NUM_TASK+GROUP_NUM)])
    multi_loss_list = helper_net.construct(
        Tensor(vae.weights_q_1.weight.data),
        Tensor(vae.weights_q_1.bias.data),
        Tensor(vae.weights_q_2.weight.data),
        Tensor(vae.weights_q_2.bias.data)
    )
    return multi_loss_list, grads_net
def main_train_vad(validation=True, best_epoch=None):  # , Nu_list):
    """
    :param validation:
    :param best_epoch:
    :return:
    """
    data = preprocess(validation, best_epoch)

    epochs, num, n_vad = data['epochs'], data['num'], data['n_vad']
    total_anneal_steps = data['total_anneal_steps']
    best_recall, update_count = data['best_recall'], data['update_count']
    net_with_criterion = data['net_with_criterion']
    train_net, label = data['train_net'], data['label']
    cal_grads, vae = data['cal_grads'], data['vae']
    min_norm_solver = MinNormSolver()
    for epoch in tqdm(range(epochs)):
        for group in TRAIN_GROUP_DICT:
            random.shuffle(TRAIN_GROUP_DICT[group])
        st_idx_dict = dict((group, 0) for group in range(GROUP_NUM))

        recon_train_loss_list, train_r30_list = [], []
        r30_train_group_list = []
        x_group_list = []

        for bnum, st_idx in enumerate(range(0, num, ARG['batch'])):
            print(bnum, st_idx)
            x_group, x_input, st_idx_dict = \
                sampler(TRAIN_DATA, st_idx_dict, ARG['batch'], TRAIN_GROUP_DICT)

            train_set = sparse.lil_matrix(x_input).rows
            max_train_count = 0
            vad_item = [[] for i in range(len(train_set))]

            if sparse.isspmatrix(x_input):
                x_input = x_input.toarray()
            x_input = x_input.astype('float32')

            if total_anneal_steps > 0:
                anneal = min(ARG['beta'], 1. * update_count / total_anneal_steps)
            else:
                anneal = ARG['beta']
            vae.anneal_ph = anneal
            x_input = Tensor(x_input, dtype=mstype.float32)
            x_group = Tensor(x_group, dtype=mstype.float32)
            multi_loss_list, grads = get_loss(vae, x_group, x_input, train_net, cal_grads)
            sum_loss = sum(multi_loss_list)
            recon_train_loss_list.append(multi_loss_list)
            if ARG['MOO']:
                for category, grad_t in enumerate(grads):
                    grads[category] = np.hstack(
                        [g.reshape(-1)*multi_loss_list[category]/sum_loss for g in grad_t]
                    )
                scale, min_norm = min_norm_solver.find_min_norm_element(grads)
                scale = np.minimum(1.0, scale + 0.2)
                print(min_norm)
                if min_norm < 5:
                    save_model(vae, 'model.ckpt')
                    # saver.save(sess, '{}/chkpt'.format(LOG_DIR))
                #     return best_recall

            vae.tsk_weights_ph = Tensor(scale, mstype.float32)
            for i in range(1):
                train_net((x_input, x_group))
            # print(net_with_criterion((x_input, x_group), label, op=1))

            pred_val = vae.get_grad_loss(x_input, x_group)[0]

            if ARG['lagrangian_method']:
                for i in range(1):
                    train_net((x_input, x_group))
                print(net_with_criterion((x_input, x_group), label, op_=2))
                # results = sess.run([train_op_list[2]] + lagrange_list, feed_dict=feed_dict)
                # print('lagrange_list', results[1:])
                # print(train_op_list[2], lagrange_list)

            _, _, recall_k, _ = hit_precision_recall_ndcg_k(
                vad_item, train_set, np.squeeze(np.array(pred_val)),
                max_train_count, k=30, ranked_tag=False)
            train_r30_list.extend(recall_k)
            x_group_list.extend(x_group)

            # if bnum % 50 == 0:
            #     summary_train = sess.run(merged_var, feed_dict=feed_dict)
            #     summary_writer.add_summary(
            #         summary_train,
            #         global_step=epoch * num_batches + bnum)
            update_count += 1

        x_group_list = np.array(x_group_list)
        train_r30_list = np.array(train_r30_list)
        recon_train_loss_list = np.array(recon_train_loss_list)

        if validation:
            recall = validation_fun(n_vad, vae, train_net, train_r30_list, r30_train_group_list)
        else:
            recall = train_r30_list.mean()
            for i in range(GROUP_NUM):
                train_r30_tmp = (train_r30_list[x_group_list == i]).mean()
                if not np.isnan(train_r30_tmp):
                    r30_train_group_list.append(train_r30_tmp)
            print('train_negelbo: ', recon_train_loss_list.mean(0),
                  '\ntrain_recall: ', r30_train_group_list)

        if recall > best_recall:
            best_epoch = epoch
            # saver.save(sess, '{}/chkpt'.format(LOG_DIR))
            best_recall = recall
    save_model(vae, 'model.ckpt')
    return best_epoch

def output(recon_tst_loss_list, h30_list, r30_list, h20_list, r20_list,
           tst_cnt, x_group_list, tst_cnt_array):
    """[summary]

    Args:
        recon_tst_loss_list ([type]): [description]
        h30_list ([type]): [description]
        r30_list ([type]): [description]
        h20_list ([type]): [description]
        r20_list ([type]): [description]
        tst_cnt ([type]): [description]
        x_group_list ([type]): [description]
        tst_cnt_array ([type]): [description]
    """
    recon_tst_loss_list = np.array(recon_tst_loss_list).mean(0)
    h30_list = np.array(h30_list)
    r30_list = np.array(r30_list)
    h20_list = np.array(h20_list)
    r20_list = np.array(r20_list)

    x_group_list = np.array(x_group_list)

    r20_group_list, r30_group_list = [], []
    hit20_group_list, hit30_group_list = [], []

    for i in range(GROUP_NUM):
        r20_tmp = (r20_list[x_group_list == i]).mean()
        if not np.isnan(r20_tmp):
            r20_group_list.append(r20_tmp)

        hit20_tmp = (h20_list[x_group_list == i]).sum() / tst_cnt_array[i]
        if not np.isnan(hit20_tmp):
            hit20_group_list.append(hit20_tmp)

        r30_tmp = (r30_list[x_group_list == i]).mean()
        if not np.isnan(r30_tmp):
            r30_group_list.append(r30_tmp)

        h30_tmp = (h30_list[x_group_list == i]).sum() / tst_cnt_array[i]
        if not np.isnan(h30_tmp):
            hit30_group_list.append(h30_tmp)
    recall20_diff = np.std(np.array(r20_group_list))
    hit20_diff = np.std(np.array(hit20_group_list))
    recall30_diff = np.std(np.array(r30_group_list))
    hit30_diff = np.std(np.array(hit30_group_list))
    print('=================================================')
    print('recon_tst_loss_list', recon_tst_loss_list)
    print('r30_group_list', r30_group_list, '\nhit30_group_list', hit30_group_list)
    print('r20_group_list', r20_group_list, '\nhit20_group_list', hit20_group_list)

    print("Test HR@20=%.5f" % (h20_list.sum() / tst_cnt),
          file=sys.stderr)
    print("Test Recall@20={} ({})".format(
        r20_list.mean(), np.std(r20_list) / np.sqrt(len(r20_list))),
          file=sys.stderr)
    print("Test HR@30=%.5f " % (h30_list.sum() / tst_cnt),
          file=sys.stderr)
    print("Test Recall@30=%{} ({})".format(
        r30_list.mean(), np.std(r30_list) / np.sqrt(len(r30_list))),
          file=sys.stderr)

    print("Test difference between groups are: \num Recall@20=%.5f, "
          "hit@20=%.5f, Recall@30=%.5f, hit@30=%.5f," %
          (recall20_diff, hit20_diff, recall30_diff, hit30_diff))

    file = open(ARG['data'] + '/hyper_search.txt', 'a')
    file.write('Recall20: %.5f' % r20_list.mean() + '\t')
    file.write('HitRate20: %.5f' % (h20_list.sum() / tst_cnt) + '\t')
    file.write('Recall30: %.5f' % r30_list.mean() + '\t')
    file.write('HitRate30: %.5f' % (h30_list.sum() / tst_cnt) + '\t')

    file.write('Recall20-std: %.5f' % recall20_diff + '\t')
    file.write('HitRate20-std: %.5f' % hit20_diff + '\t')
    file.write('Recall30-std: %.5f' % recall30_diff + '\t')
    file.write('HitRate30-std: %.5f' % hit30_diff + '\t')

    file.write('beta:' + str(ARG['beta']) + '\tdfac:' + str(ARG['dfac']) +
               '\tkfac:' + str(ARG['kfac']) + '\tkeep:' + str(ARG['keep']))
    file.write('\n')
    file.close()

def main_tst():
    """
    :param report_r20:
    :return:
    """
    set_rng_seed(ARG['seed'])
    n_test = TEST_DATA.shape[0]
    n_items = TEST_DATA.shape[1]
    # idxlist_test = list(range(n_test))

    vae = MOO(n_items)
    param_dict = load_checkpoint("model.ckpt")
    # 将参数加载到网络中
    load_param_into_net(vae, param_dict)

    # saver, logits_var, _, _, multi_loss_list, _, lagrange_list = vae.build_graph()

    h30_list, r30_list = [], []
    h20_list, r20_list = [], []

    x_group_list = []
    tst_cnt_array = np.zeros(GROUP_NUM)

    # TRAIN_DATA = sp.lil_matrix(TRAIN_DATA)
    # TEST_DATA = sp.lil_matrix(TEST_DATA)
    st_idx_dict = dict((group, 0) for group in range(GROUP_NUM))
    tst_cnt = 0
    recon_tst_loss_list = []
    # loss = nn.MSELoss()
    #
    # optimizer = nn.SGD(params=vae.trainable_params())
    # net_with_criterion = WithLossCell(vae, loss)
    # train_net = TrainOneStepCell(vae, optimizer)
    # label = Tensor(0, dtype=mstype.float32)

    for bnum, st_idx in enumerate(range(0, n_test, BATCH_SIZE_TEST)):
        print(bnum, st_idx)
        x_group, x_input, x_te, x_vad, st_idx_dict = \
            sampler(TRAIN_DATA, st_idx_dict, BATCH_SIZE_TEST,
                    TEST_GROUP_DICT, TEST_DATA, VAD_DATA)

        train_set = sparse.lil_matrix(x_input).rows
        if 'book-crossing' in ARG['data']:
            x_vad = None
        else:
            x_vad = sparse.lil_matrix(x_vad).rows
        max_train_count = np.max([len(category) for category in train_set])
        tst_item = sparse.lil_matrix(x_te).rows

        if sparse.isspmatrix(x_input):
            x_input = x_input.toarray()
        x_input = x_input.astype('float32')
        vae.anneal_ph = 0
        x_input = Tensor(x_input, dtype=mstype.float32)
        x_group = Tensor(x_group, dtype=mstype.float32)

        # for i in range(10):
        #     train_net((x_input, x_group), label, op=0)
        #     print(net_with_criterion((x_input, x_group), label, op=0))

        logits_var, multi_loss_list = vae.get_grad_loss(x_input, x_group)

        pred_val = logits_var.asnumpy()
        # print('pred_val', np.max(pred_val), np.min(pred_val))
        pred_val[(x_input.asnumpy()).nonzero()] = -np.inf
        recon_tst_loss_list.append(multi_loss_list)

        hits_tmp, _, recall_tmp, _ = \
            hit_precision_recall_ndcg_k(train_set, tst_item,
                                        np.squeeze(np.array(pred_val)),
                                        max_train_count, k=30, ranked_tag=False,
                                        vad_set_batch=x_vad)
        h30_list.extend(hits_tmp)
        r30_list.extend(recall_tmp)

        hits_tmp, _, recall_tmp, _ = \
            hit_precision_recall_ndcg_k(train_set, tst_item,
                                        np.squeeze(np.array(pred_val)),
                                        max_train_count, k=20, ranked_tag=False,
                                        vad_set_batch=x_vad)
        h20_list.extend(hits_tmp)
        r20_list.extend(recall_tmp)

        tst_cnt += x_te.count_nonzero()

        for i in range(GROUP_NUM):
            tst_cnt_array[i] += np.sum(
                [len(l) for l in tst_item[np.array(x_group) == i]]
            )

        x_group_list.extend(x_group)
    output(recon_tst_loss_list, h30_list, r30_list, h20_list, r20_list,
           tst_cnt, x_group_list, tst_cnt_array)
    return sum(r20_list)/len(r20_list)


if __name__ == '__main__':

    (N_ITEMS, N_USERS, TRAIN_DATA, TRAIN_GROUP_DICT, VAD_DATA, VAD_GROUP_DICT,
     TEST_DATA, TEST_GROUP_DICT) = load_data(ARG['data'])
    print('finishing loading data', '%d users and %d items' % (N_USERS, N_ITEMS))

    GROUP_NUM = len(TRAIN_GROUP_DICT)
    NUM_TASK = GROUP_NUM

    VALIDATION, TEST = 0, 0
    BEST_EPOCH = int(ARG['epoch'] / 1.2)

    LOG_DIR = 'log/'

    if not os.path.exists(LOG_DIR):
        os.makedirs(LOG_DIR)

    if ARG['mode'] in ('vad',):
        BEST_EPOCH = main_train_vad(validation=True)  # , Nu_list)
        print('======= VALIDATION finished, the best epoch is {} ======='.format(BEST_EPOCH))
    if ARG['mode'] in ('trn',):
        BEST_EPOCH = main_train_vad(validation=False, best_epoch=BEST_EPOCH)
        print('======= training finished, the best epoch is {} ======='.format(BEST_EPOCH))

    if ARG['mode'] in ('trn', 'vad', 'TEST'):
        TEST = main_tst()  # 其实不用vad�?把train和vad合并成train就可以了
