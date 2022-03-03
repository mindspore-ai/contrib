from hyperopt import hp
# gnn 训练超参搜索空间
HP_SEARCH_SPACE = {
    "drop_out": hp.choice("drop_out",
                          [0.0, 0.1, 0.2, 0.3, 0.4, 0.5,
                           0.6, 0.7, 0.8, 0.9]),
    "learning_rate": hp.choice("learning_rate",
                               [1e-1, 1e-2, 1e-3,
                                1e-4, 1e-5]),
    "learning_rate_decay": hp.choice("learning_rate_decay",
                                     [0, 1e-3, 5e-4, 1e-4,
                                      5e-5, 1e-5])}
HP_SEARCH_SPACE_Mapping = \
    {
    "drop_out": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5,0.6, 0.7, 0.8, 0.9],
    "learning_rate": [1e-1, 1e-2, 1e-3,1e-4, 1e-5],
    "learning_rate_decay":[0, 1e-3, 5e-4, 1e-4, 5e-5, 1e-5]
    }