from graphpas.build_gnn.gnn_manager import GnnManager

def val_data_etimatamtion(gnn_architecture, data, gnn_parameter, search_parameter):

    drop_out = 0.60
    learning_rate = 0.005
    learning_rate_decay = 0.0005
    train_epoch = 300
    model_select = "min_loss"
    one_layer_component_num = 5
    early_stop = True
    early_stop_mode = "val_loss"
    early_stop_patience = 10

    if "drop_out" in gnn_parameter:
        drop_out = gnn_parameter["drop_out"]
    if "learning_rate" in gnn_parameter:
        learning_rate = gnn_parameter["learning_rate"]
    if "weight_decay" in gnn_parameter:
        learning_rate_decay = gnn_parameter["learning_rate_decay"]
    if "train_epoch" in gnn_parameter:
        train_epoch = gnn_parameter["train_epoch"]
    if "model_select" in gnn_parameter:
        model_select = gnn_parameter["model_select"]
    if "one_layer_component_num" in gnn_parameter:
        one_layer_component_num = gnn_parameter["one_layer_component_num"]
    if "early_stop" in gnn_parameter:
        early_stop = gnn_parameter["early_stop"]
    if "early_mode" in gnn_parameter:
        early_stop_mode = gnn_parameter["early_stop_mode"]
    if "early_num" in gnn_parameter:
        early_stop_patience = gnn_parameter["early_stop_patience"]

    es_mode = "transductive"

    if "es_mode" in search_parameter:
        es_mode = search_parameter["es_mode"]

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

        model.build_gnn(gnn_architecture, data)
        val_model, val_acc = model.train()

    return val_acc

def test_data_estimation(gnn_architecture, data, gnn_parameter, search_parameter):

    # gnn train 默认配置
    drop_out = 0.6
    learning_rate = 0.005
    learning_rate_decay = 0.0005
    train_epoch = 300
    model_select = "min_loss"
    one_layer_component_num = 5
    early_stop = True
    early_stop_mode = "val_loss"
    early_stop_patience = 10

    if "drop_out" in gnn_parameter:
        drop_out = gnn_parameter["drop_out"]
    if "learning_rate" in gnn_parameter:
        learning_rate = gnn_parameter["learning_rate"]
    if "weight_decay" in gnn_parameter:
        learning_rate_decay = gnn_parameter["learning_rate_decay"]
    if "train_epoch" in gnn_parameter:
        train_epoch = gnn_parameter["train_epoch"]
    if "model_select" in gnn_parameter:
        model_select = gnn_parameter["model_select"]
    if "one_layer_component_num" in gnn_parameter:
        one_layer_component_num = gnn_parameter["one_layer_component_num"]
    if "early_stop" in gnn_parameter:
        early_stop = gnn_parameter["early_stop"]
    if "early_stop_mode" in gnn_parameter:
        early_stop_mode = gnn_parameter["early_stop_mode"]
    if "early_stop_num" in gnn_parameter:
        early_stop_patience = gnn_parameter["early_stop_patience"]

    es_mode = "transductive"

    if "es_mode" in search_parameter:
        es_mode = search_parameter["es_mode"]

    if es_mode == "transductive":
        if not search_parameter["ensemble"]:
            model = GnnManager(drop_out,
                               learning_rate,
                               learning_rate_decay,
                               train_epoch,
                               model_select,
                               one_layer_component_num,
                               early_stop,
                               early_stop_mode,
                               early_stop_patience)

            model.build_gnn(gnn_architecture, data, training=False)
            val_model, max_val_acc, max_val_acc_test_acc, min_loss_val_acc, min_val_loss_test_acc = model.test_evaluate()

            if model_select == "max_acc":
                test_acc = max_val_acc_test_acc
            elif model_select == "min_loss":
                test_acc = min_val_loss_test_acc
        else:
            model_num = 1
            model_list = []
            for gnn in gnn_architecture:
                print("retrain model ", model_num)

                model = GnnManager(drop_out,
                                   learning_rate,
                                   learning_rate_decay,
                                   train_epoch,
                                   model_select,
                                   one_layer_component_num,
                                   early_stop,
                                   early_stop_mode,
                                   early_stop_patience)

                model.build_gnn(gnn, data, training=False)

                val_model = model.retrian()
                model_num += 1
                model_list.append(val_model)
            test_acc = model.ensemble_test(model_list)

    return test_acc