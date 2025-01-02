import argparse
import os
from transformers import *
import mindspore as ms
from mindspore import Tensor, nn, context
from model import LangModelWithDense
from loader import load_data

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_path")
    parser.add_argument("ensemble_path")
    parser.add_argument("--output_path", default="predictions.txt")

    args = parser.parse_args()

    context.set_context(mode=context.GRAPH_MODE, device_target="CPU")
    print("Using device: CPU\n")

    models = []
    for model_path in os.listdir(args.ensemble_path):
        print("Loading model from: {}".format(os.path.join(args.ensemble_path, model_path)))
        
        model_name_path = os.path.join(args.ensemble_path, model_path)
        tokenizer = AutoTokenizer.from_pretrained(model_name_path if "scibert" not in model_name_path else "bert-base-uncased")
        model = ms.load_checkpoint(os.path.join(args.ensemble_path, model_path, "model.ckpt"))
        model.fine_tune = False
        model.set_train(False)

        models.append((model, tokenizer, model_path))

    predictions = {}
    for model, tokenizer, model_path in models:
        print("Predicting labels for model: {}...".format(model_path))
        test_loader = load_data(args.input_path, tokenizer, "CPU")

        y_pred = []
        for test_x, mask in test_loader:
            outputs = nn.Sigmoid()(model(Tensor(test_x), Tensor(mask)).reshape(-1))

            for output in outputs:
                pred = 0 if output.asnumpy() < 0.5 else 1

                y_pred.append(pred)

        predictions[model_path] = y_pred

    counter = len(list(predictions.values())[0])
    model_paths = list(predictions.keys())

    with open(args.output_path, "w") as file:
        for model_path in model_paths:
            file.write("{:<15} ".format(model_path))

        file.write("{:<15} \n".format("ensemble"))

        for i in range(counter):
            label_sum = 0

            for model_path in model_paths:
                file.write("{:<15} ".format(predictions[model_path][i]))
                label_sum += predictions[model_path][i]

            if label_sum > len(models) / 2:
                file.write("{:<15} ".format(1))
            else:
                file.write("{:<15} ".format(0))

            file.write("\n")
