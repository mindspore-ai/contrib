import numpy as np
import os
import ray
import time
from graphpas.search_space import search_space_node_classification, gnn_architecture_flow
from graphpas.graphpas_search import utils
from graphpas.estimation import val_data_etimatamtion
from data_utils.util_cite_network import CiteNetwork
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

        print(35*"=", "initializing population  based on random", 35*"=")

        gnn_architecture_list = []
        gnn_architecture_embedding_list = []
        while len(gnn_architecture_list) < self.search_parameter["single_searcher_initial_num"]:
            gnn_architecture_embedding = utils.random_generate_gnn_architecture_embedding()
            gnn_architecture = utils.gnn_architecture_embedding_decoder(gnn_architecture_embedding)
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

        print(35*"=", "search", 35*"=")
        print("sharing population:\n", sharing_population)
        print("sharing accuracy:\n", sharing_accuracy)
        print("[sharing population accuracy] Mean/Median/Best:\n",
                np.mean(sharing_accuracy),
                np.median(sharing_accuracy),
                np.max(sharing_accuracy))

        parents = self.selection(sharing_population, sharing_accuracy)
        print(self.search_name + "selected parents:\n", parents)

        children = self.mutation(parents, mutation_selection_probability, history_pop)
        print(self.search_name+"mutated children:\n", children)

        history_pop = history_pop + children

        return children, history_pop

    def selection(self,
                  population,
                  accuracies):
        print(35*"=", self.search_name + " wheel select strategy", 35*"=")
        fitness = np.array(accuracies)
        fitness_probility = fitness / sum(fitness)
        fitness_probility = fitness_probility.tolist()
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

        print(35 * "=", self.search_name + " mutate", 35 * "=")
        for index in range(len(parents)):

            while parents[index] in history_pop:
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

        print(35*"=", "update", 35*"=")
        print("sharing_acc before update:\n", sharing_acc)

        _, top_acc = utils.top_population_select(sharing_population,
                                                 sharing_acc,
                                                 top_k=self.sharing_num)
        avg_accuracy = np.mean(top_acc)

        index = 0
        for acc in sharing_children_val_acc_list:
            if acc > avg_accuracy:
                sharing_acc.append(acc)
                sharing_population.append(sharing_children[index])
                index += 1
            else:
                index += 1
        print("sharing_acc after update:\n", sharing_acc)
        return sharing_population, sharing_acc

class Search(object):

    def __init__(self, data, search_parameter, gnn_parameter):

        self.data = data
        self.search_parameter = search_parameter
        self.gnn_parameter = gnn_parameter

    def search_operator(self):

        paralleloperator_list = []
        for searcher_name in self.search_parameter["searcher_name"]:
            paralleloperator = ParallelOperator.remote(self.data,
                                                       self.search_parameter,
                                                       searcher_name,
                                                       self.gnn_parameter)
            paralleloperator_list.append(paralleloperator)

        print(35 * "=", "initial based on random", 35 * "=")

        time_initial = time.time()
        task = []
        for index in range(len(self.search_parameter["searcher_name"])):
            initial_pop_list, val_acc_list = paralleloperator_list[index].random_initialize_population.remote()
            task.append(initial_pop_list)
            task.append(val_acc_list)
        result = ray.get(task)
        time_initial = time.time() - time_initial

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

        searcher_list = []
        for index in range(len(self.search_parameter["searcher_name"])):
            searcher = EvolutionSearch(searcher_name=self.search_parameter["searcher_name"][index],
                                       sharing_num=self.search_parameter["sharing_num"],
                                       mutation_num=self.search_parameter["mutation_num"][index])
            searcher_list.append(searcher)

        print(35 * "=", "sharing_population select", 35 * "=")
        sharing_population, sharing_accuracy = utils.top_population_select(history_pop,
                                                                           history_acc,
                                                                           top_k=self.search_parameter["sharing_num"])
        sharing_population_temp = sharing_population.copy()
        mutation_selection_probability = utils.mutation_selection_probability(sharing_population_temp, gnn_architecture_flow)
        print(35 * "=", "mutate select probability", 35 * "=")
        print(mutation_selection_probability)
        print(35 * "=", "search", 35 * "=")

        time_search_list = []
        epoch = []

        for i in range(self.search_parameter["search_epoch"]):
            time_search = time.time()
            sharing_children = []
            sharing_children_embedding = []

            for searcher in searcher_list:
                children, history_pop = searcher.search(history_pop,
                                                        sharing_population,
                                                        sharing_accuracy,
                                                        mutation_selection_probability)
                sharing_children.append(children)
                sharing_children_embedding = sharing_children_embedding + children

            sharing_children_architecture = []
            for children_list in sharing_children:
                children_gnn_architecture_list = []
                for gnn_architecture_embedding in children_list:
                    gnn_architecture = utils.gnn_architecture_embedding_decoder(gnn_architecture_embedding)
                    children_gnn_architecture_list.append(gnn_architecture)
                sharing_children_architecture.append(children_gnn_architecture_list)
            print("sharing_children_architecture:\n", sharing_children_architecture)

            task = []
            for gnn_architecture_list, paralleloperator in zip(sharing_children_architecture, paralleloperator_list):
                children_val_acc = paralleloperator.fitness_computation.remote(gnn_architecture_list)
                task.append(children_val_acc)
            result = ray.get(task)
            sharing_children_val_acc_list = []
            for acc in result:
                sharing_children_val_acc_list = sharing_children_val_acc_list + acc

            history_acc = history_acc + sharing_children_val_acc_list

            sharing_population, sharing_accuracy = searcher_list[0].updating(sharing_children_embedding,
                                                                             sharing_children_val_acc_list,
                                                                             sharing_population,
                                                                             sharing_accuracy)

            sharing_population_temp = sharing_population.copy()
            mutation_selection_probability = utils.mutation_selection_probability(sharing_population_temp, gnn_architecture_flow)
            time_search_list.append(time.time()-time_search)
            epoch.append(i+1)

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

        path = os.path.split(os.path.realpath(__file__))[0][:-24] + "data_save/graphpas_time_save"
        utils.experiment_time_save(path,
                                   self.data.data_name + "_search_time.txt",
                                   epoch,
                                   time_search_list)

if __name__=="__main__":
    ParallelConfig(False)

    graph = CiteNetwork("cora")

    search_parameter = {"searcher_name": ["ev1", "ev2"],
                        "mutation_num": [1, 2],
                        "single_searcher_initial_num": 5,
                        "sharing_num": 5,
                        "search_epoch": 2,
                        "es_mode": "transductive",
                        "ensemble": False,
                        "ensemble_num": 5}

    gnn_parameter = {"drop_out": 0.5,
                     "learning_rate": 0.05,
                     "learning_rate_decay": 0.0005,
                     "train_epoch": 5,
                     "model_select": "min_loss",
                     "retrain_epoch": 5,
                     "early_stop": False,
                     "early_stop_mode": "val_acc",
                     "early_stop_patience": 10}

    graphpas_instance = Search(graph, search_parameter, gnn_parameter)
    graphpas_instance.search_operator()