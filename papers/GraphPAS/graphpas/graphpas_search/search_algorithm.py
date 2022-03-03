import numpy as np
import os
import ray
import time
from graphpas.search_space import search_space_node_classification, gnn_architecture_flow
from graphpas.graphpas_search import utils
from graphpas.estimation import val_data_etimatamtion
from data_utils.util_cite_network import CiteNetwork # for unit test
from parallel_config import ParallelConfig, ray_gpu_num

@ray.remote(num_gpus=ray_gpu_num)
class ParallelOperator(object):

    def __init__(self, data, search_parameter, searcher_name, gnn_parameter):
        self.data = data
        self.search_parameter = search_parameter
        self.searcher_name = searcher_name
        self.gnn_parameter = gnn_parameter

    @ray.method(num_returns=2)
    def random_initialize_population(self):

        print(35*"=", "随机初始化种群", 35*"=")  # 随机初始化种群

        gnn_architecture_list = []
        gnn_architecture_embedding_list = []
        while len(gnn_architecture_list) < self.search_parameter["single_searcher_initial_num"]:
            gnn_architecture_embedding = utils.random_generate_gnn_architecture_embedding()
            gnn_architecture = utils.gnn_architecture_embedding_decoder(gnn_architecture_embedding)
            # 将gnn结构数字编码解码为字符串名称
            gnn_architecture_embedding_list.append(gnn_architecture_embedding)
            gnn_architecture_list.append(gnn_architecture)
        val_acc_list = self.fitness_computation(gnn_architecture_list)

        path = os.path.split(os.path.realpath(__file__))[0][:-24] + "data_save/graphpas_data_save"
        if not os.path.exists(path):
            os.makedirs(path)
        utils.experiment_initial_data_save(path,
                                           self.searcher_name + "_initialization.txt",
                                           gnn_architecture_list,
                                           val_acc_list)

        return gnn_architecture_embedding_list, val_acc_list

    @ray.method(num_returns=1)
    def fitness_computation(self, gnn_architecture_list):
        acc_list = []
        for gnn_architecture in gnn_architecture_list:
            val_acc = val_data_etimatamtion(gnn_architecture,
                                            self.data,
                                            self.gnn_parameter,
                                            self.search_parameter)
            acc_list.append(val_acc)
            print("gnn_architecture: ", gnn_architecture)
            print("gnn_val_acc: ", val_acc)
        return acc_list

class EvolutionSearch(object):

    def __init__(self, searcher_name, sharing_num, mutation_num):
        self.search_name = searcher_name
        self.sharing_num = sharing_num
        self.mutation_num = mutation_num

    def search(self,
               history_pop,
               sharing_population,
               sharing_accuracy,
               mutation_selection_probability):

        print(35*"=", "进化搜索", 35*"=")
        print("sharing population:\n", sharing_population)
        print("sharing accuracy:\n", sharing_accuracy)
        print("[sharing population accuracy] Mean/Median/Best:\n",
                np.mean(sharing_accuracy),
                np.median(sharing_accuracy),
                np.max(sharing_accuracy))

        # 选择 获取parents :使用wheel策略
        parents = self.selection(sharing_population, sharing_accuracy)
        print(self.search_name + "选择后parents:\n", parents)

        # 变异 基于mutation_select_probability选择变异组件
        children = self.mutation(parents, mutation_selection_probability, history_pop)
        print(self.search_name+"变异后children:\n", children)

        # 为了增加计算效率,先不计算child_list中的个体fitness,去重后再计算child_list中fitness,
        # 按照顺序往history_accuracy列表中添加
        # 已经给不同的searcher产生的child进行了去重处理
        history_pop = history_pop + children

        return children, history_pop

    def selection(self,
                  population,
                  accuracies):
        print(35*"=", self.search_name + " 选择(wheel select strategy)", 35*"=")
        # 基于fitness计算采样概率:
        fitness = np.array(accuracies)
        fitness_probility = fitness / sum(fitness)
        fitness_probility = fitness_probility.tolist()
        # 基于fitness概率采样parent
        index_list = [index for index in range(len(fitness))]
        parents = []
        parent_index = np.random.choice(index_list, self.sharing_num, replace=False, p=fitness_probility)
        for index in parent_index:
            parents.append(population[index].copy())
        return parents

    def mutation(self,
                 parents,
                 mutation_selection_probability,
                 history_pop):

        print(35 * "=", self.search_name + " 变异(基于信息熵选择变异的component)", 35 * "=")
        for index in range(len(parents)):

            # 直到出现history_pop中没有的gnn结构为止
            while parents[index] in history_pop:
                # 基于information_entropy_probability确定变异基因的位点，基于mutation_num确定变异基因的个数
                position_to_mutate_list = np.random.choice([gene for gene in range(len(parents[index]))],
                                                           self.mutation_num,
                                                           replace=False,
                                                           p=mutation_selection_probability)

                for mutation_index in position_to_mutate_list:
                    mutation_space = search_space_node_classification[gnn_architecture_flow[mutation_index]]
                    parents[index][mutation_index] = np.random.randint(0, len(mutation_space))
        return parents

    def updating(self,
                 sharing_children,
                 sharing_children_val_acc_list,
                 sharing_population,
                 sharing_acc
                 ):

        print(35*"=", "更新", 35*"=")
        print("更新前sharing_acc:\n", sharing_acc)

        # 计算sharing_popualtion中top k的acc均值
        _, top_acc = utils.top_population_select(sharing_population,
                                                 sharing_acc,
                                                 top_k=self.sharing_num)
        avg_accuracy = np.mean(top_acc)

        index = 0
        for acc in sharing_children_val_acc_list:
            if acc > avg_accuracy:
                # 添加到sharing_population中
                sharing_acc.append(acc)
                sharing_population.append(sharing_children[index])
                index += 1
            else:
                index += 1
        print("更新后sharing_acc:\n", sharing_acc)
        return sharing_population, sharing_acc

class Search(object):

    def __init__(self, data, search_parameter, gnn_parameter):

        self.data = data
        self.search_parameter = search_parameter
        self.gnn_parameter = gnn_parameter

    def search_operator(self):

        paralleloperator_list = []
        # 构造ray并行类
        for searcher_name in self.search_parameter["searcher_name"]:
            paralleloperator = ParallelOperator.remote(self.data,
                                                       self.search_parameter,
                                                       searcher_name,
                                                       self.gnn_parameter)
            paralleloperator_list.append(paralleloperator)

        print(35 * "=", "并行随机初始化", 35 * "=")

        time_initial = time.time()
        task = []
        for index in range(len(self.search_parameter["searcher_name"])):
            initial_pop_list, val_acc_list = paralleloperator_list[index].random_initialize_population.remote()
            task.append(initial_pop_list)
            task.append(val_acc_list)
        result = ray.get(task)
        time_initial = time.time() - time_initial

        # 种群随机初始化时间记录
        path = os.path.split(os.path.realpath(__file__))[0][:-24] + "data_save/graphpas_time_save"
        if not os.path.exists(path):
            os.makedirs(path)
        utils.experiment_time_save_initial(path, self.data.data_name + "_initial_time.txt", time_initial)

        total_initial_pop_list = []
        for pop in result[::2]:
            total_initial_pop_list = total_initial_pop_list + pop
        total_val_acc_list = []
        for acc in result[1::2]:
            total_val_acc_list = total_val_acc_list + acc

        # 初始化种群个体去重
        history_pop = []
        history_acc = []
        index = 0
        for gnn in total_initial_pop_list:
            if gnn not in history_pop:
                history_pop.append(gnn)
                history_acc.append(total_val_acc_list[index])
                index += 1
            else:
                index += 1

        # slave 进化节点初始化
        searcher_list = []
        for index in range(len(self.search_parameter["searcher_name"])):
            searcher = EvolutionSearch(searcher_name=self.search_parameter["searcher_name"][index],
                                       sharing_num=self.search_parameter["sharing_num"],
                                       mutation_num=self.search_parameter["mutation_num"][index])
            searcher_list.append(searcher)

        # 进化种群选择, fitness　top　选择策略
        print(35 * "=", "sharing_population 选择", 35 * "=")
        sharing_population, sharing_accuracy = utils.top_population_select(history_pop,
                                                                           history_acc,
                                                                           top_k=self.search_parameter["sharing_num"])
        # 变异选择概率计算
        sharing_population_temp = sharing_population.copy()
        mutation_selection_probability = utils.mutation_selection_probability(sharing_population_temp, gnn_architecture_flow)
        print(35 * "=", "基于信息熵确,确定选择变异的componont概率", 35 * "=")
        print(mutation_selection_probability)
        print(35 * "=", "并行进化搜索", 35 * "=")

        time_search_list = []
        epoch = []

        # 进化搜索开始
        for i in range(self.search_parameter["search_epoch"]):
            time_search = time.time()
            # 串行进行选择与变异操作
            sharing_children = []  # 为计算fitness做准备
            sharing_children_embedding = []  # 为updating操作准备

            for searcher in searcher_list:
                children, history_pop = searcher.search(history_pop,
                                                        sharing_population,
                                                        sharing_accuracy,
                                                        mutation_selection_probability)
                sharing_children.append(children)
                sharing_children_embedding = sharing_children_embedding + children

            # sharing_children 串行解码
            sharing_children_architecture = []
            for children_list in sharing_children:
                children_gnn_architecture_list = []
                for gnn_architecture_embedding in children_list:
                    gnn_architecture = utils.gnn_architecture_embedding_decoder(gnn_architecture_embedding)
                    children_gnn_architecture_list.append(gnn_architecture)
                sharing_children_architecture.append(children_gnn_architecture_list)
            print("sharing_children_architecture:\n", sharing_children_architecture)

            # 并行计算每个searcher的gnn_architecture fitness
            task = []
            for gnn_architecture_list, paralleloperator in zip(sharing_children_architecture, paralleloperator_list):
                children_val_acc = paralleloperator.fitness_computation.remote(gnn_architecture_list)
                task.append(children_val_acc)
            result = ray.get(task)
            sharing_children_val_acc_list = []
            for acc in result:
                sharing_children_val_acc_list = sharing_children_val_acc_list + acc
            # history_acc 与 history_pop 对齐
            history_acc = history_acc + sharing_children_val_acc_list

            # sharing population更新
            sharing_population, sharing_accuracy = searcher_list[0].updating(sharing_children_embedding,
                                                                             sharing_children_val_acc_list,
                                                                             sharing_population,
                                                                             sharing_accuracy)

            # 重新计算mutation_selection_probability
            sharing_population_temp = sharing_population.copy()
            mutation_selection_probability = utils.mutation_selection_probability(sharing_population_temp, gnn_architecture_flow)
            time_search_list.append(time.time()-time_search)
            epoch.append(i+1)

            # 搜索　gnn结构 val_acc　数据保存
            path = os.path.split(os.path.realpath(__file__))[0][:-24] + "data_save/graphpas_data_save"
            utils.experiment_graphpas_data_save(path,
                                                self.data.data_name + "_search_epoch_" + str(i+1) + ".txt",
                                                sharing_population,
                                                sharing_accuracy)

        index = sharing_accuracy.index(max(sharing_accuracy))
        best_val_architecture = sharing_population[index]
        best_val_architecture = utils.gnn_architecture_embedding_decoder(best_val_architecture)
        best_acc = max(sharing_accuracy)
        print("Best architecture:\n", best_val_architecture)
        print("Best val_acc:\n", best_acc)

        # 搜索时间保存
        path = os.path.split(os.path.realpath(__file__))[0][:-24] + "data_save/graphpas_time_save"
        utils.experiment_time_save(path,
                                   self.data.data_name + "_search_time.txt",
                                   epoch,
                                   time_search_list)

if __name__=="__main__":
    # 并行配置
    ParallelConfig(False)

    # 图数据获取
    graph = CiteNetwork("cora")

    # 搜索配置
    search_parameter = {"searcher_name": ["ev1", "ev2"],   # 并行搜索器名称, 确定　graphpas　并行度;
                        "mutation_num": [1, 2],            # 每个搜索器变异组件数,与 searcher_name 对应;
                        "single_searcher_initial_num": 5,  # 每个搜索器初始化 gnn 个数;
                        "sharing_num": 5,                  # 初始化共享 population 与共享 child 规模
                        "search_epoch": 2,                 # 搜索轮次;
                        "es_mode": "transductive",         # gnn 验证模式, 目前只支持模式:transductive
                        "ensemble": False,                 # 是否开启　inference　集成策略
                        "ensemble_num": 5}                 # inference　集成度只能时奇数,集成度不能超过搜索结束后共享 parent size　

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

    graphpas_instance = Search(graph, search_parameter, gnn_parameter)
    graphpas_instance.search_operator()

