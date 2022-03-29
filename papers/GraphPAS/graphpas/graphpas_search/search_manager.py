from graphpas.graphpas_search import search_algorithm
from graphpas.graphpas_search import utils
from graphpas import estimation
from data_utils.util_cite_network import CiteNetwork  # for unit test
from parallel_config import ParallelConfig
import os
import re

class GraphPAS(object):

    def __init__(self, data, search_parameter_, gnn_parameter_):
        self.data = data
        self.search_parameter = search_parameter_
        self.gnn_parameter = gnn_parameter_

    def search(self):
        gnn_search = search_algorithm.Search(self.data, self.search_parameter, self.gnn_parameter)
        gnn_search.search_operator()

    def derive(self):

        path = os.path.split(os.path.realpath(__file__))[0][:-24] + "data_save/graphpas_data_save"
        file_name = os.listdir(path)
        index = 0
        temp = 0
        file_index = 0
        for file_ in file_name:
            if "search_epoch_" in file_:
                epoch_num = int(re.findall(r"\d+\.?\d*", file_)[0][0])
                if epoch_num > temp:
                    temp = epoch_num
                    file_index = index
                index += 1
            else:
                index += 1
        file_name = file_name[file_index]
        path = os.path.split(os.path.realpath(__file__))[0][:-24] + "data_save/graphpas_data_save/" + file_name
        gnn, acc = utils.experiment_graphpas_data_load(path)

        if not self.search_parameter["ensemble"]:

            gnn, _ = utils.top_population_select(gnn, acc, self.search_parameter["derive_val_num"])
            acc = []
            for gnn_architecture in gnn:
                val_acc = estimation.val_data_etimatamtion(gnn_architecture, self.data, self.gnn_parameter, self.search_parameter)
                acc.append(val_acc)
            gnn, acc = utils.top_population_select(gnn, acc, 1)
            test_acc = estimation.test_data_estimation(gnn[0], self.data, self.gnn_parameter, self.search_parameter)
            print("best gnn architecture: ", gnn[0])
            print("test accuracy:", test_acc)

        elif self.search_parameter["ensemble"]:

            ensemble_num_list = [i for i in range(self.search_parameter["ensemble_num"]+1)]
            ensemble_num_list = ensemble_num_list[1::2]

            for ensemble in ensemble_num_list:

                gnn_, _ = utils.top_population_select(gnn, acc, ensemble)
                test_acc = estimation.test_data_estimation(gnn_, self.data, self.gnn_parameter,
                                                        self.search_parameter)
                print("ensemble_num : ", ensemble)
                print("gnn architecture list : ", gnn_)
                print("test accuracy:", test_acc)

                path = os.path.split(os.path.realpath(__file__))[0][:-24] + "data_save/graphpas_data_save"
                if not os.path.exists(path):
                    os.makedirs(path)
                utils.experiment_graphpas_final_data_save(path,
                                                          "ensemble_num_"+str(ensemble)+"_final_gnn_test_acc.txt",
                                                          gnn_,
                                                          test_acc)


if __name__ =="__main__":
    ParallelConfig(False)
    graph = CiteNetwork("cora")
    search_parameter = {"searcher_name": ["ev1", "ev2"],
                        "mutation_num": [1, 2],
                        "single_searcher_initial_num": 5,
                        "sharing_num": 5,
                        "search_epoch": 2,
                        "es_mode": "transductive",
                        "derive_val_num": 3,
                        "ensemble": True,
                        "ensemble_num": 3}

    gnn_parameter = {"drop_out": 0.5,
                     "learning_rate": 0.05,
                     "learning_rate_decay": 0.0005,
                     "train_epoch": 5,
                     "model_select": "min_loss",
                     "retrain_epoch": 5,
                     "early_stop": False,
                     "early_stop_mode": "val_acc",
                     "early_stop_patience": 10}

    graphpas_instance = GraphPAS(graph, search_parameter, gnn_parameter)
    graphpas_instance.search()
    graphpas_instance.derive()