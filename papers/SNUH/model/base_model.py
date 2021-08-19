
"""Base model framework."""
import math
import random
import pickle
import argparse
import datetime
from datetime import timedelta
from timeit import default_timer as timer
import numpy as np
from mindspore import context, save_checkpoint, load_checkpoint, load_param_into_net
import mindspore.nn as nn

from utils.mindspore_helper import GradWrap
from utils.logger import Logger
from utils.data import LabeledDocuments
from utils.evaluation import compute_retrieval_precision
from utils.mindspore_helper import gen_checkpoints_list

context.set_context(mode=context.PYNATIVE_MODE, device_target="CPU")

class BaseModel(nn.Cell):
    """Base model"""
    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams
        self.load_data()

    def load_data(self):
        self.data = LabeledDocuments(self.hparams.data_path, self.hparams.num_neighbors)

    def run_training_sessions(self):
        """run outer training session"""
        logger = Logger(self.hparams.model_path + '.log', on=True)
        val_perfs = []
        best_val_perf = float('-inf')
        start = timer()
        random.seed(self.hparams.seed)  # For reproducible random runs

        for run_num in range(1, self.hparams.num_runs + 1):
            state_dict, val_perf = self.run_training_session(run_num, logger)
            val_perfs.append(val_perf)

            if val_perf > best_val_perf:
                best_val_perf = val_perf
                logger.log('----New best {:8.2f}, saving'.format(val_perf))
                save_checkpoint(gen_checkpoints_list(state_dict), self.hparams.model_path+'.ckpt')
                pickle.dump(self.hparams, open(self.hparams.model_path+'.hpar', 'wb'))

        logger.log('Time: %s' % str(timedelta(seconds=round(timer() - start))))
        self.load()
        if self.hparams.num_runs > 1:
            logger.log_perfs(val_perfs)
            logger.log('best hparams: ' + self.flag_hparams())

        val_perf, test_perf = self.run_test()
        logger.log('Val:  {:8.2f}'.format(val_perf))
        logger.log('Test: {:8.2f}'.format(test_perf))

    def run_training_session(self, run_num, logger):
        """run inner training session"""
        if self.hparams.num_runs > 1:
            logger.log('RANDOM RUN: %d/%d' % (run_num, self.hparams.num_runs))
            for hparam, values in self.get_hparams_grid().items():
                assert hasattr(self.hparams, hparam)
                self.hparams.__dict__[hparam] = random.choice(values)

        np.random.seed(self.hparams.seed)
        random.seed(self.hparams.seed)

        train_loader, database_loader, val_loader, _ = self.data.get_loaders(
            self.hparams.num_trees, self.hparams.alpha, self.hparams.batch_size, self.hparams.num_workers,
            shuffle_train=True, get_test=False)
        self.define_parameters(self.data.num_nodes, self.data.num_edges)
        opt = nn.Adam(params=self.trainable_params(), learning_rate=self.hparams.lr)
        train_network = GradWrap(self)
        train_network.set_train()
        best_val_perf = float('-inf')
        loss_sum = 0
        num_steps = 0
        bad_epochs = 0

        times = []
        try:
            for epoch in range(1, self.hparams.epochs + 1):
                starttime = datetime.datetime.now()
                for _, batch in enumerate(train_loader):
                    x, label, edge1, edge2, weight = batch[0], batch[1], batch[2], batch[3], batch[4]
                    grads = train_network(x, label, edge1, edge2, weight)
                    opt(grads)

                    loss = self.construct(x, edge1, edge2, weight)
                    loss_sum += loss.asnumpy()
                    num_steps += 1

                endtime = datetime.datetime.now()
                times.append(endtime - starttime)

                if math.isnan(loss_sum):
                    logger.log('Stopping epoch because loss is NaN')
                    break

                val_perf = self.evaluate(database_loader, val_loader)
                logger.log('End of epoch {:3d}'.format(epoch), False)
                logger.log(' Loss: {:8.2f} | val perf {:8.2f}'.format(loss_sum / num_steps, val_perf), False)

                if val_perf > best_val_perf:
                    best_val_perf = val_perf
                    bad_epochs = 0
                    logger.log('\t\t*Best model so far, deep copying*')
                    state_dict = self.parameters_and_names()
                else:
                    bad_epochs += 1
                    logger.log('\t\tBad epoch %d' % bad_epochs)

                if bad_epochs > self.hparams.num_bad_epochs:
                    break
        except KeyboardInterrupt:
            logger.log('-' * 89)
            logger.log('Exiting from training early')

        logger.log("time per training epoch: " + str(np.mean(times)))
        return state_dict, best_val_perf

    def evaluate(self, database_loader, eval_loader):
        perf = compute_retrieval_precision(database_loader, eval_loader, self.encode_discrete,\
            self.hparams.distance_metric, self.hparams.num_retrieve, self.hparams.num_features)
        return perf

    def load(self):
        checkpoint = load_checkpoint(self.hparams.model_path+'.ckpt')
        hparams = pickle.load(open(self.hparams.model_path+'.hpar', 'rb'))
        self.hparams = hparams
        self.define_parameters(self.data.num_nodes, self.data.num_edges)
        load_param_into_net(self, checkpoint, strict_load=True)

    def run_test(self):
        _, database_loader, val_loader, test_loader = self.data.get_loaders(
            self.hparams.num_trees, self.hparams.alpha, self.hparams.batch_size, self.hparams.num_workers,
            shuffle_train=False, get_test=True)
        val_perf = self.evaluate(database_loader, val_loader)
        test_perf = self.evaluate(database_loader, test_loader)
        return val_perf, test_perf

    def flag_hparams(self):
        """flag hyperparameters"""
        flags = '%s %s' % (self.hparams.model_path, self.hparams.data_path)
        for hparam in vars(self.hparams):
            val = getattr(self.hparams, hparam)
            if str(val) == 'False':
                continue
            elif str(val) == 'True':
                flags += ' --%s' % (hparam)
            elif str(hparam) in {'model_path', 'data_path', 'num_runs',
                                 'num_workers'}:
                continue
            else:
                flags += ' --%s %s' % (hparam, val)
        return flags

    @staticmethod
    def get_general_argparser():
        """command parser"""
        parser = argparse.ArgumentParser()

        parser.add_argument('model_path', type=str)
        parser.add_argument('data_path', type=str)
        parser.add_argument('--train', action='store_true',
                            help='train a model?')

        parser.add_argument('--num_features', type=int, default=64,
                            help='num discrete features [%(default)d]')
        parser.add_argument('--dim_hidden', type=int, default=500,
                            help='dimension of hidden state [%(default)d]')
        parser.add_argument('--num_layers', type=int, default=0,
                            help='num layers [%(default)d]')
        parser.add_argument('--num_neighbors', type=int, default=10,
                            help='num neighbors [%(default)d]')

        parser.add_argument('--batch_size', type=int, default=128,
                            help='batch size [%(default)d]')
        parser.add_argument('--lr', type=float, default=0.001,
                            help='initial learning rate [%(default)g]')
        parser.add_argument('--init', type=float, default=0.05,
                            help='unif init range (default if 0) [%(default)g]')
        parser.add_argument('--clip', type=float, default=10,
                            help='gradient clipping [%(default)g]')
        parser.add_argument('--epochs', type=int, default=100,
                            help='max number of epochs [%(default)d]')
        parser.add_argument('--num_runs', type=int, default=1,
                            help='num random runs (not random if 1) '
                            '[%(default)d]')

        parser.add_argument('--num_retrieve', type=int, default=100,
                            help='num neighbors to retrieve [%(default)d]')
        parser.add_argument('--num_bad_epochs', type=int, default=6,
                            help='num indulged bad epochs [%(default)d]')
        parser.add_argument('--num_workers', type=int, default=0,
                            help='num dataloader workers [%(default)d]')
        parser.add_argument('--distance_metric', default='hamming',
                            choices=['hamming', 'cosine'])
        parser.add_argument('--no_tfidf', action='store_true',
                            help='raw bag-of-words as input instead of tf-idf?')
        parser.add_argument('--seed', type=int, default=50971,
                            help='random seed [%(default)d]')
        parser.add_argument('--cuda', action='store_true',
                            help='use CUDA?')

        return parser
