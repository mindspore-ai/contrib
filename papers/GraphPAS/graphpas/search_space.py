search_space_node_classification = \
               {'attention_type': ['gat', 'gcn', 'cos', 'const', 'gat_sym', 'linear', 'generalized_linear'],
                'aggregator_type': ['sum', 'mean', 'max'],
                'activate_function': ['sigmoid', 'tanh', 'relu', 'linear', 'softplus', 'leaky_relu', 'relu6', 'elu'],
                'number_of_heads': [1, 2, 4, 6, 8],
                'hidden_units': [4, 8, 16, 32, 64, 128, 256]}

gnn_architecture_flow \
            = ['attention_type', 'aggregator_type', 'activate_function', 'number_of_heads', 'hidden_units',
               'attention_type', 'aggregator_type', 'activate_function', 'number_of_heads', 'hidden_units']