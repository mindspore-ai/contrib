from hyperopt import fmin, tpe
from autognas.hp_search.hp_search_space import HP_SEARCH_SPACE, HP_SEARCH_SPACE_Mapping
from data_utils.util_cite_network import CiteNetwork # for unit test
from autognas.build_gnn.gnn_manager import GnnManager

class HpSearchObj():

    def __init__(self, gnn_architecture, data, gnn_parameter, search_parameter):
        self.gnn_architecture = gnn_architecture
        self.data = data
        self.gnn_parameter = gnn_parameter
        self.search_parameter = search_parameter

    def tuining_obj(self, hp_space):

        # gnn train 默认配置
        drop_out = 0.60
        learning_rate = 0.005
        learning_rate_decay = 0.0005
        train_epoch = 300
        model_select = "min_loss"
        one_layer_component_num = 5
        early_stop = True
        early_stop_mode = "val_loss"
        early_stop_patience = 10

        # 训练超参搜索参数
        # =======================================
        if "drop_out" in hp_space:
            drop_out = hp_space["drop_out"]
        if "learning_rate" in hp_space:
            learning_rate = hp_space["learning_rate"]
        if "learning_rate_decay" in hp_space:
            learning_rate_decay = hp_space["learning_rate_decay"]

        print(32 * "#" + " 本轮训练超参 " + 32 * "#")
        print("drop_out: %f / learning_rate: %f / learning_rate_decay: %f" % (
        drop_out, learning_rate, learning_rate_decay))
        print(32 * "#" + " 本轮训练超参 " + 32 * "#")

        # ========================================
        if "train_epoch" in self.gnn_parameter:
            train_epoch = self.gnn_parameter["train_epoch"]
        if "model_select" in self.gnn_parameter:
            model_select = self.gnn_parameter["model_select"]
        if "one_layer_component_num" in self.gnn_parameter:
            one_layer_component_num = self.gnn_parameter["one_layer_component_num"]
        if "early_stop" in self.gnn_parameter:
            early_stop = self.gnn_parameter["early_stop"]
        if "early_mode" in self.gnn_parameter:
            early_stop_mode = self.gnn_parameter["early_stop_mode"]
        if "early_num" in self.gnn_parameter:
            early_stop_patience = self.gnn_parameter["early_stop_patience"]

        # search 默认验证配置
        es_mode = "transductive"

        # 读取search验证模式配置
        if "es_mode" in self.search_parameter:
            es_mode = self.search_parameter["es_mode"]

        if es_mode == "transductive":
            model = GnnManager(drop_out,
                               learning_rate,
                               learning_rate_decay,
                               train_epoch,
                               model_select,
                               one_layer_component_num,
                               early_stop,
                               early_stop_mode,
                               early_stop_patience)

            model.build_gnn(self.gnn_architecture, self.data, training=False)
            val_model, val_acc, val_loss = model.train()

        return val_loss


    def hp_tuning(self, search_epoch, search_algorithm):
        print("target_gnn_architecture:", self.gnn_architecture)
        best_hp = fmin(fn=self.tuining_obj,
                       space=HP_SEARCH_SPACE,
                       algo=search_algorithm,
                       max_evals=search_epoch)
        drop_out_index = best_hp["drop_out"]
        learning_rate_index = best_hp["learning_rate"]
        learning_rate_decay_index = best_hp["learning_rate_decay"]

        drop_out = HP_SEARCH_SPACE_Mapping["drop_out"][drop_out_index]
        learning_rate = HP_SEARCH_SPACE_Mapping["learning_rate"][learning_rate_index]
        learning_rate_decay = HP_SEARCH_SPACE_Mapping["learning_rate_decay"][learning_rate_decay_index]

        print("optimal_hyper_parameter: drop_out %f / learning_rate: %f / learning_rate_decay: %f"%
              (drop_out, learning_rate, learning_rate_decay))

        return drop_out, learning_rate, learning_rate_decay

if __name__=="__main__":
    # 获取图数据
    graph = CiteNetwork("cora")
    # 搜索配置
    search_parameter = {"parallel_num": 10,  # graphpas　并行度;
                        "mutation_num": [1, 1, 2, 2, 3, 3, 4, 4, 5, 5],  # 每个搜索器变异组件数,与 searcher_name 对应;
                        "initial_num": 16,  # 每个搜索器初始化 gnn 个数;
                        "sharing_num": 16,  # 初始化共享 population 与共享 parent 规模
                        "search_epoch": 1,  # 搜索轮次;
                        "es_mode": "transductive",  # gnn 验证模式, 目前只支持模式: transductive
                        "derive_val_num": 3,
                        "test_gnn_num": 5,
                        "exp_mode": "acc_hp"}  # 基于 top fitness 验证最终 model　的数量;

    # GNN 训练配置
    gnn_parameter = {"drop_out": 0.5,  # 节点特征矩阵　drop_out 操作随机赋0率
                     "learning_rate": 0.05,  # 学习率
                     "learning_rate_decay": 0.0005,  # 学习率衰减
                     "train_epoch": 100,  # model 训练轮次
                     "model_select": "min_loss",  # inference模型选择方式:1.min_loss,验证loss最小对应model 2.max_acc:验证acc最大对应model
                     "retrain_epoch": 1,  # test模型选取重新训练 GNN 轮次;
                     "early_stop": False,  # 是否开启 GNN 训练时 early stop 模式
                     "early_stop_mode": "val_acc",  # early stop模式:1.基于val_loss判断, 2.基于val_acc判断
                     "early_stop_patience": 10}  # early stop　基于 search epoch 判断终止训练窗口

    gnn_architecture = ['const', 'sum', 'relu', 6, 128, 'gcn', 'sum', 'tanh', 6, 7]

    HP_tuining = HpSearchObj(gnn_architecture, graph, gnn_parameter, search_parameter)
    HP_tuining.hp_tuning(search_epoch=20, search_algorithm=tpe.suggest)