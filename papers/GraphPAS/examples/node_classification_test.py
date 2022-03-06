import graphpas.graphpas_search.search_manager as search_algorithm
from data_utils.util_cite_network import CiteNetwork
from parallel_config import ParallelConfig


ParallelConfig(True)
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

searcher = search_algorithm.GraphPAS(graph, search_parameter, gnn_parameter)
searcher.search()
searcher.derive()