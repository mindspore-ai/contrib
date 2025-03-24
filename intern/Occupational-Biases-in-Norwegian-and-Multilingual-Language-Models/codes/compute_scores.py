import logging
import pandas as pd
import mindspore.ops as ops
import argparse
from mindnlp.transformers import AutoTokenizer, AutoModelForMaskedLM, AutoModel
from mindnlp.transformers import BertForMaskedLM, BertTokenizer
import mindspore as ms

def compute_bias_scores(args):
    print("Evaluating pretrained language models for bias in correlations between gender and occupations")
    print("Input:", args.template_file)
    print("Model:", args.lm_model)
    print("=" * 100)

    logging.basicConfig(level=logging.INFO)

    with open(args.template_file) as input_file:
        data = input_file.readlines()

    print('Loading the model ...')

    if args.lm_model == "norbert":
        tokenizer = AutoTokenizer.from_pretrained("ltgoslo/norbert")
        model = AutoModelForMaskedLM.from_pretrained("ltgoslo/norbert")
        uncased = False
    elif args.lm_model == "norbert2":
        tokenizer = AutoTokenizer.from_pretrained("ltgoslo/norbert2")
        model = AutoModelForMaskedLM.from_pretrained("ltgoslo/norbert2")
        uncased = False
    elif args.lm_model == "nbbert":
        tokenizer = AutoTokenizer.from_pretrained("NbAiLab/nb-bert-base")
        model = AutoModelForMaskedLM.from_pretrained("NbAiLab/nb-bert-base")
        uncased = False
    elif args.lm_model == "mbert":
        tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
        model = BertForMaskedLM.from_pretrained('bert-base-multilingual-cased')
        uncased = False
    elif args.lm_model == "roberta":

        tokenizer = AutoTokenizer.from_pretrained('xlm-roberta-base')
        model = AutoModelForMaskedLM.from_pretrained('xlm-roberta-base')
        uncased = False

    elif args.lm_model == "nbbertLarge":
        tokenizer = AutoTokenizer.from_pretrained("NbAiLab/nb-bert-large")
        model = AutoModelForMaskedLM.from_pretrained("NbAiLab/nb-bert-large")
        uncased = True

    elif args.lm_model == "nbroberta":
        tokenizer = AutoTokenizer.from_pretrained("NbAiLab/nb-roberta-base")
        model = AutoModelForMaskedLM.from_pretrained("NbAiLab/nb-roberta-base")
        uncased = False

    ms.set_context(device_target="CPU") # 设置使用CPU

    model.set_train(False)  # 设置为推理模式

    df_score = pd.DataFrame(columns=['female', 'male',
                                     'female_score', 'male_score',
                                     'total_score', 'percentage_female', 'percentage_male', 'bias_score'])

    print("=" * 100)
    print('Processing ...')

    for sequence in data:
        sequence = sequence.replace('[MASK]', ''.join({tokenizer.mask_token}))
        sequence = sequence.strip()

        input_ids = tokenizer.encode(sequence, return_tensors="ms")
        mask_token_equal = ops.equal(input_ids, tokenizer.mask_token_id)
        mask_token_index = ops.argwhere(mask_token_equal)[:, 1]

        token_logits = model(input_ids)[0]
        # print(model(input_ids), len(model(input_ids)))
        mask_token_logits = token_logits[0, mask_token_index, :]
        mask_token_logits = ops.softmax(mask_token_logits, axis=1)

        pronouns = ['Hun', 'Han', 'hun', 'han']

        female_score = 0
        male_score = 0
        total_score = 0
        bias_score = 0

        for pronoun in pronouns:
            pronoun_id = tokenizer.encode(pronoun, add_special_tokens=False)[0]  # 928

            token_score = mask_token_logits[:, pronoun_id]

            if len(token_score) > 0:

                total_score += token_score.item()
                if pronoun in ['Hun', 'hun']:
                    female_score += token_score.item()
                elif pronoun in ['Han', 'han']:
                    male_score += token_score.item()
            else:
                total_score += 0
                if pronoun in ['Hun', 'hun']:
                    female_score += 0
                elif pronoun in ['Han', 'han']:
                    male_score += 0

        if total_score > 0:
            female_percentage = (female_score / total_score) * 100
            male_percentage = (male_score / total_score) * 100
            if female_percentage > male_percentage:
                bias_score = 1
            else:
                bias_score = 0
        else:
            bias_score = 'NaN'
            female_percentage = 0
            male_percentage = 0

        new_row = pd.DataFrame([{
            'female': sequence.replace(tokenizer.mask_token, '[Hh]un'),
            'male': sequence.replace(tokenizer.mask_token, '[Hh]an'),
            'female_score': female_score,
            'male_score': male_score,
            'total_score': total_score,
            'percentage_female': female_percentage,
            'percentage_male': male_percentage,
            'bias_score': bias_score
        }])

        df_score = pd.concat([df_score, new_row], ignore_index=True)

    df_score.to_csv(args.output_file)
    print("=" * 100)
    print('Output file saved, check "{}" for results'.format(args.output_file))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, help="the task to analyse (occupations)")
    parser.add_argument("--template_file", type=str, help="path to the template file")
    parser.add_argument("--lm_model", type=str, help="pretrained LM models to use")
    parser.add_argument("--output_file", type=str, help="path to output file with sentence scores")

    args = parser.parse_args()
    compute_bias_scores(args)

if __name__ == '__main__':
    main()