from graphpas.graphpas_search import search_algorithm
from graphpas.graphpas_search import utils
from graphpas import estimation
from data_utils.util_cite_network import CiteNetwork  # for unit test
from parallel_config import ParallelConfig
import os
import re

class GraphPAS(object):

    def __init__(self, data, search_parameter, gnn_parameter):
        self.data = data
        self.search_parameter = search_parameter
        self.gnn_parameter = gnn_parameter

    def search(self):
        gnn_search = search_algorithm.Search(self.data, self.search_parameter, self.gnn_parameter)
        gnn_search.search_operator()

    def derive(self):

        # 获取最后一次搜索结果的sharing_population与对应的acc
        path = os.path.split(os.path.realpath(__file__))[0][:-24] + "data_save/graphpas_data_save"
        file_name = os.listdir(path)
        index = 0
        temp = 0
        file_index = 0
        for file in file_name:
            if "search_epoch_" in file:
                epoch_num = int(re.findall(r"\d+\.?\d*", file)[0][0])
                if epoch_num > temp:
                    temp = epoch_num
                    file_index = index
                index += 1
            else:
                index += 1
        file_name = file_name[file_index]
        path = os.path.split(os.path.realpath(__file__))[0][:-24] + "data_save/graphpas_data_save/" + file_name
        gnn, acc = utils.experiment_graphpas_data_load(path)

        # test 开始
        if not self.search_parameter["ensemble"]:
            # 挑选 top k gnn 重新训练
            gnn, _ = utils.top_population_select(gnn, acc, self.search_parameter["derive_val_num"])
            acc = []
            for gnn_architecture in gnn:
                val_acc = estimation.val_data_etimatamtion(gnn_architecture, self.data, self.gnn_parameter, self.search_parameter)
                acc.append(val_acc)
            # 挑选出 top 1 作为最后的输出模型
            gnn, acc = utils.top_population_select(gnn, acc, 1)
            test_acc = estimation.test_data_estimation(gnn[0], self.data, self.gnn_parameter, self.search_parameter)
            print("best gnn architecture: ", gnn[0])
            print("test accuracy:", test_acc)

        elif self.search_parameter["ensemble"]:
            # 集成度验证
            if self.search_parameter["ensemble_num"] % 2 == 0:
                raise Exception("wrong ensemble_num：", self.search_parameter["ensemble_num"])

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
    # 并行配置
    ParallelConfig(False)

    # 图数据获取
    graph = CiteNetwork("cora")

    # 搜索配置
    search_parameter = {"searcher_name": ["ev1", "ev2"],   # 并行搜索器名称, 确定　graphpas　并行度;
                        "mutation_num": [1, 2],            # 每个搜索器变异组件数,与 searcher_name 对应;
                        "single_searcher_initial_num": 5,  # 每个搜索器初始化 gnn 个数;
                        "sharing_num": 5,                  # 基于 top fitness 选择的初始化共享 population 与共享 child 规模;
                        "search_epoch": 2,                 # 搜索轮次;
                        "es_mode": "transductive",         # gnn 验证模式, 目前只支持模式:transductive;
                        "derive_val_num": 3,               # 基于 top fitness 验证最终 model　的数量;
                        "ensemble": True,                  # 是否开启　inference　集成策略;
                        "ensemble_num": 3}                 # inference　集成度只能时奇数,集成度不能超过搜索结束后共享 parent size;　

    # GNN 训练配置
    gnn_parameter = {"drop_out": 0.5,                # 节点特征矩阵 drop_out 操作随机赋0率
                     "learning_rate": 0.05,          # 学习率
                     "learning_rate_decay": 0.0005,  # 学习率衰减
                     "train_epoch": 5,               # model 训练轮次
                     "model_select": "min_loss",     # 训练模型选择方式:1.min_loss,验证loss最小对应model 2.max_acc:验证acc最大对应model
                     "retrain_epoch": 5,             # derive 模型重新训练 model 轮次;
                     "early_stop": False,            # 是否开启 model 训练 early stop 模式
                     "early_stop_mode": "val_acc",   # early stop 模式: 1.基于val_loss判断, 2.基于val_acc判断
                     "early_stop_patience": 10}      # early stop 基于 search epoch 判断终止训练窗口

    graphpas_instance = GraphPAS(graph, search_parameter, gnn_parameter)
    graphpas_instance.search()  # graphpas 并行搜索
    graphpas_instance.derive()  # 搜索结果获取






