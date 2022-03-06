import numpy as np
import torch
import torch.nn.functional as F
import time
from graphpas.build_gnn.gnn_net import GnnNet
from collections import Counter

class GnnManager(object):

    def __init__(self,
                 drop_out=0.6,
                 learning_rate=0.005,
                 learning_rate_decay=0.0005,
                 train_epoch=300,
                 model_select="min_loss",
                 one_layer_component_num=5,
                 early_stop=True,
                 early_stop_mode="val_loss",
                 early_stop_patience=10):

        self.drop_out = drop_out
        self.learning_rate = learning_rate
        self.learning_rate_decay = learning_rate_decay
        self.train_epoch = train_epoch
        self.model_select = model_select
        self.retrain_epoch = self.train_epoch
        self.one_layer_component_num = one_layer_component_num
        self.early_stop = early_stop
        self.early_stop_mode = early_stop_mode
        self.early_stop_patience = early_stop_patience
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def build_gnn(self, architecture, data, training=True):
        print("build architecture:", architecture)

        self.architecture = architecture

        if training:
            self.architecture_check(self.architecture)
            self.data_check(data)

        self.data = data
        self.in_feats = data.num_features
        self.num_class = data.y.max().item() + 1
        self.architecture[-1] = self.num_class

        layer_num = int(len(self.architecture) / self.one_layer_component_num)

        self.model = GnnNet(self.architecture,
                            self.in_feats,
                            layer_num,
                            self.one_layer_component_num,
                            dropout=self.drop_out)

        self.model.build_architecture()

        self.optimizer = torch.optim.Adam(self.model.parameters(),
                                          lr=self.learning_rate,
                                          weight_decay=self.learning_rate_decay)

        self.loss_func = torch.nn.functional.nll_loss

    def train(self, test_mode=False, show_info=False):

        print("train architecture:", self.architecture)

        self.model.to(self.device)
        val_model, val_acc = self.run_model(self.model,
                                            self.optimizer,
                                            self.loss_func,
                                            self.data,
                                            self.train_epoch,
                                            test_mode=test_mode,
                                            show_info=show_info)

        return val_model, val_acc

    def evaluate(self, output, labels, mask):
        _, indices = torch.max(output, dim=1)
        correct = torch.sum(indices[mask] == labels[mask])
        return correct.item() * 1.0 / mask.sum().item()

    def test_evaluate(self, test_mode=True, show_info=False):
        print("test architecture:", self.architecture)
        self.model.to(self.device)
        val_model, max_val_acc, max_val_acc_test_acc, min_loss_val_acc, min_val_loss_test_acc = self.run_model(self.model,
                                                                                                                self.optimizer,
                                                                                                                self.loss_func,
                                                                                                                self.data,
                                                                                                                self.retrain_epoch,
                                                                                                                test_mode=test_mode,
                                                                                                                show_info=show_info)
        return val_model, max_val_acc, max_val_acc_test_acc, min_loss_val_acc, min_val_loss_test_acc

    def retrian(self):

        print("train architecture:", self.architecture)
        self.model.to(self.device)
        val_model, _, _, _, _ = self.run_model(self.model,
                                               self.optimizer,
                                               self.loss_func,
                                               self.data,
                                               self.train_epoch,
                                               test_mode=True)
        return val_model

    def ensemble_test(self, model_list):
        indices_list = []
        for model in model_list:
            model.eval()
            logits = model(self.data.x, self.data.edge_index)
            logits = F.log_softmax(logits, 1)
            _, indices = torch.max(logits, dim=1)
            indices_list.append(indices)

        ensemble_pre_y = []
        for index in range(len(indices)):
            pre_y_list = []
            for pre in indices_list:
                pre_y_list.append(pre[index].item())
            label = Counter(pre_y_list).most_common(1)[0][0]
            ensemble_pre_y.append(label)
        ensemble_indices = torch.tensor(ensemble_pre_y).to(self.device)
        test_acc = self.ensemble_test_accuracy(ensemble_indices, self.data.y, self.data.test_mask)
        return test_acc

    def ensemble_test_accuracy(self, indices, labels, mask):
        correct = torch.sum(indices[mask] == labels[mask])
        return correct.item() * 1.0 / mask.sum().item()

    def early_stopping(self, val_acc_list, val_loss_list, stop_mode, stop_patience):

        if stop_mode == "val_acc":
            if len(val_acc_list) < stop_patience:
                return False

            if val_acc_list[-stop_patience:][0] > val_acc_list[-1]:
                return True
            else:
                return False

        elif stop_mode == "val_loss":
            if len(val_loss_list) < stop_patience:
                return False

            if val_loss_list[-stop_patience:][0] > val_loss_list[-1]:
                return True
            else:
                return False

    def run_model(self, model, optimizer, loss_fn, data, epochs, test_mode=False, show_info=False):

        dur = []
        begin_time = time.time()
        max_val_acc_test_acc = 0
        min_val_loss_test_acc = 0
        max_val_acc = 0
        min_val_loss = float("inf")
        min_loss_val_acc = 0
        min_loss_val_epoch = None
        min_loss_model = None
        max_acc_model = None
        val_epoch = 0

        val_acc_list = []
        val_loss_list = []
        early_stop_flag = False

        print("Number of train data:", data.train_mask.sum())
        # 输出训练样本规模
        for epoch in range(1, epochs + 1):

            t0 = time.time()
            model.train()
            logits = model(data.x, data.edge_index)
            logits = F.log_softmax(logits, 1)

            loss = loss_fn(logits[data.train_mask], data.y[data.train_mask])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            model.eval()
            logits = model(data.x, data.edge_index)
            logits = F.log_softmax(logits, 1)
            train_acc = self.evaluate(logits, data.y, data.train_mask)
            val_acc = self.evaluate(logits, data.y, data.val_mask)
            test_acc = self.evaluate(logits, data.y, data.test_mask)
            loss = loss_fn(logits[data.val_mask], data.y[data.val_mask])
            val_loss = loss.item()
            val_loss_list.append(val_acc)
            val_loss_list.append(val_loss)
            if self.early_stop:
                early_stop_flag = self.early_stopping(val_acc_list,
                                                      val_loss_list,
                                                      self.early_stop_mode,
                                                      self.early_stop_patience)

            dur.append(time.time() - t0)

            if val_acc > max_val_acc:
                max_val_acc = val_acc
                max_val_acc_test_acc = test_acc
                max_acc_model = model
                val_epoch = epoch

            if val_loss < min_val_loss:
                min_val_loss = val_loss
                min_loss_val_acc = val_acc
                min_val_loss_test_acc = test_acc
                min_loss_model = model
                min_loss_val_epoch = epoch

            if show_info:
                print(
                    "Epoch {:05d} | val Loss {:.4f} | Time(s) {:.4f} | train acc {:.4f} | max_val_acc {:.4f} | max_val_acc_test_acc {:.4f}".format(
                        epoch, val_loss, np.mean(dur), train_acc, max_val_acc, max_val_acc_test_acc))
                print(
                    "Epoch {:05d} | val Loss {:.4f} | Time(s) {:.4f} | train acc {:.4f} | min_loss_val_acc {:.4f} | min_loss_test_acc {:.4f}".format(
                        epoch, val_loss, np.mean(dur), train_acc, min_loss_val_acc, min_val_loss_test_acc))

                end_time = time.time()
                print("Each Epoch Cost Time: %f " % ((end_time - begin_time) / epoch))

            if early_stop_flag:
                print("early stopping epoch:", epoch, "\n")
                break

        print(f"the_best_val_accuracy:{max_val_acc}\n"
              f"the_best_val_acc_test_accuracy:{max_val_acc_test_acc}\n"
              f"the_result_epoch:{val_epoch}\n")

        print(f"min_loss_the_best_val_accuracy:{min_loss_val_acc}\n"
              f"min_val_loss_test_accuracy:{min_val_loss_test_acc}\n"
              f"the_result_epoch:{min_loss_val_epoch}\n")

        if test_mode:
            if self.model_select == "max_acc":
                return max_acc_model,  max_val_acc, max_val_acc_test_acc, min_loss_val_acc, min_val_loss_test_acc
            elif self.model_select == "min_loss":
                return min_loss_model, max_val_acc, max_val_acc_test_acc, min_loss_val_acc, min_val_loss_test_acc

        if self.model_select == "max_acc":
            return max_acc_model, max_val_acc
        elif self.model_select == "min_loss":
            return min_loss_model, min_loss_val_acc

    def architecture_check(self, architecture):

        search_space = [
            ["gat", "gcn", "cos", "const", "gat_sym", 'linear', 'generalized_linear'],
            ["sum", "mean", "max"],
            ["sigmoid", "tanh", "relu", "linear", "softplus", "leaky_relu", "relu6", "elu"],
            [1, 2, 4, 6, 8],
            [4, 8, 16, 32, 64, 128, 256]
        ]

        if len(architecture) % 5 != 0:
            raise Exception("wrong architecture sizes：", len(architecture))

        index = 0
        for component in architecture:

            if index == 5:
                index = 0

            if component not in search_space[index]:
                raise Exception("wrong architecture component:", str(component))

            index += 1

    def data_check(self, data):
        pass
