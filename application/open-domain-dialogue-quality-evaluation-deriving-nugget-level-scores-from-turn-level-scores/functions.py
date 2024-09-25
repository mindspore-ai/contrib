import mindspore as ms
from mindspore import ops
from mindnlp.transformers import RobertaTokenizer, AutoModelForSequenceClassification
import pandas as pd
import numpy as np

def load_model_and_tokenizer():
    tokenizer = RobertaTokenizer.from_pretrained("roberta-large")
    model = AutoModelForSequenceClassification.from_pretrained("roberta-large")
    return tokenizer, model

def add_turn_level_scores(df, tokenizer, model):
    scores = []
    for sentence in df['input']:
        inputs = tokenizer(sentence, return_tensors="ms")
        input_ids = ms.Tensor(inputs['input_ids'])
        attention_mask = ms.Tensor(inputs['attention_mask'])
        model.set_train(False)
        logits = model(input_ids, attention_mask).logits

        
        logits = ops.stop_gradient(logits)
        probabilities = ops.softmax(logits, axis=-1)
        scores.append(probabilities[0, 1].item())
    df['turn_level_score'] = scores

def map_labels(df):
    dialogue_act = ["Agreement", "Disagreement", "Yes_Answer", "No_Answer", "Conversation_Opening", 
                    "Conversation_Closing", "Apology", "Thanking", "Rejection", "Applausal", 
                    "Declarative Question", "Confusion", "Reasoning", "Downplayer", "Assumption", 
                    "Acknowledgement", "Clarification", "Non_Declarative Question", "User_Instruction", 
                    "Recommendation", "Citation", "Comparison", "Example", "Commisive", "Opinion"]
    label_dict = {"Original": "T", "Delete": "T_ø"}
    
    for entry in dialogue_act:
        label_dict[entry] = "T_diff"
    for i in range(1, 11):
        label_dict[f"rep_{i}"] = "T_same"

    df['label_new'] = df['label'].map(label_dict)

def load_csv(path):
    df = pd.read_csv(path)
    tokenizer, model = load_model_and_tokenizer()
    add_turn_level_scores(df, tokenizer, model)
    map_labels(df)
    return df

def compute_score(s_T, scores, K):
    scores.sort(reverse=True)
    return sum([s_T - score for score in scores[:K]]) / K

def nug_eval(data, W_1, W_2, W_3, K, L):
    max_nug = data['nugget'].max()

    s_T = data[data['label_new'] == "T"]['turn_level_score'].mean()
    s_Tø = [s_T - data[(data['label_new'] == "T_ø") & (data['nugget'] == i+1)]['turn_level_score'].mean() for i in range(max_nug)]
    
    s_Tdiff_K = [compute_score(s_T, list(data[(data['label_new'] == "T_diff") & (data['nugget'] == i+1)]['turn_level_score']), K) for i in range(max_nug)]
    s_Tsame_L = [compute_score(s_T, list(data[(data['label_new'] == "T_same") & (data['nugget'] == i+1)]['turn_level_score']), L) for i in range(max_nug)]
    
    return [sigmoid(W_1 * s_Tø[i] + W_2 * s_Tdiff_K[i] + W_3 * s_Tsame_L[i]) for i in range(max_nug)]

def sigmoid(x):
    return 1/(1 + np.exp(-x))