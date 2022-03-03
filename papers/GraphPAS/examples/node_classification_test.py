import graphpas.graphpas_search.search_manager as search_algorithm
from data_utils.util_cite_network import CiteNetwork
from parallel_config import ParallelConfig

# 并行配置
ParallelConfig(True)  # True 并行; False 串行;

# 获取图数据
graph = CiteNetwork("cora")

# 搜索配置
search_parameter = {"searcher_name": ["ev1", "ev2"],    # 并行搜索器名称, 确定　graphpas　并行度;
                    "mutation_num": [1, 2],             # 每个搜索器变异组件数,与 searcher_name 对应;
                    "single_searcher_initial_num": 5,   # 每个搜索器初始化 gnn 个数;
                    "sharing_num": 5,                   # 初始化共享 population 与共享 child 规模
                    "search_epoch": 2,                  # 搜索轮次;
                    "es_mode": "transductive",          # gnn 验证模式, 目前只支持模式: transductive
                    "derive_val_num": 3,                # 基于 top fitness 验证最终 model　的数量;
                    "ensemble": True,                   # 是否开启　inference　集成策略
                    "ensemble_num": 3}                  # inference　集成度只能奇数,集成度不能超过搜索结束后共享 parent size　

# GNN 训练配置
gnn_parameter = {"drop_out": 0.5,                # 节点特征矩阵　drop_out 操作随机赋0率
                 "learning_rate": 0.05,          # 学习率
                 "learning_rate_decay": 0.0005,  # 学习率衰减
                 "train_epoch": 5,               # model 训练轮次
                 "model_select": "min_loss",     # 训练模型选择方式:1.min_loss,验证loss最小对应model 2.max_acc:验证acc最大对应model
                 "retrain_epoch": 5,             # test模型选取重新训练 GNN 轮次;
                 "early_stop": False,            # 是否开启 GNN 训练时 early stop 模式
                 "early_stop_mode": "val_acc",   # early stop模式:1.基于val_loss判断, 2.基于val_acc判断
                 "early_stop_patience": 10}      # early stop　基于 search epoch 判断终止训练窗口

searcher = search_algorithm.GraphPAS(graph, search_parameter, gnn_parameter)
searcher.search()  # graphpas 并行搜索
searcher.derive()  # gnn 获取