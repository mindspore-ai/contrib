import argparse
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import mindspore as ms
from mindspore import nn, context
from mindnlp.transformers import AutoTokenizer, GPT2LMHeadModel,AutoModelForCausalLM


context.set_context(mode=context.PYNATIVE_MODE, device_target="CPU")
model = AutoModelForCausalLM.from_pretrained('gpt2', cache_dir="path_to_cache")

def extract_example(row):
    return {
        'full_prefix': row['full_prefix'],  
        'completion': row['completion'],
        'contradictions': [
            row['contradiction_0'],
            row['contradiction_1'],
            row['contradiction_2']
        ]
    }

def read_data(path):
    df = pd.read_csv(path)
    required_columns = {
        'prefix': ['full_prefix', 'turncated_prefixes', 'truncated_prefix', 'prefix'],
        'doc': ['doc_id', 'docid', 'document_id'],
        'completion': ['completion', 'response'],
        'contradictions': ['contradiction_0', 'contradiction_1', 'contradiction_2']
    }
    

    df.columns = df.columns.str.lower()  
    col_map = {}
    
    for col_type, candidates in required_columns.items():
        # 候选列名也转为小写
        candidates_lower = [c.lower() for c in candidates]
        for candidate in candidates_lower:
            if candidate in df.columns:
                col_map[col_type] = candidate
                break
        if col_type not in col_map:
            available = [c for c in df.columns if col_type in c]
            raise ValueError(
                f"Missing required column: {col_type}\n"
                f"Available columns: {df.columns.tolist()}\n"
                f"Possible matches: {available}"
            )
    

    df = df.rename(columns={
        col_map['prefix']: 'full_prefix',
        col_map['completion']: 'completion',
        col_map['contradictions'][0]: 'contradiction_0',
        col_map['contradictions'][1]: 'contradiction_1',
        col_map['contradictions'][2]: 'contradiction_2'
    })
    
    return df.apply(lambda row: extract_example(row), axis=1).tolist()

def load_tokenizer(model_name, max_tokens):
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        padding_side='right',
        truncation_side='left',
        model_max_length=max_tokens
    )
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer

class CausalLMModel(nn.Cell):
    def __init__(self, model_name):
        super().__init__()
        self.model = GPT2LMHeadModel.from_pretrained(model_name)
        self.model.set_train(False)

    def construct(self, input_ids, attention_mask):
        return self.model(input_ids=input_ids, attention_mask=attention_mask)[0]

def format_data(ex):
    prefix = ex['full_prefix'].rstrip()
    completion = ex['completion'].lstrip()
    contradictions = [cont.lstrip() for cont in ex['contradictions']]

    batch = [f"{prefix}{completion}"] + [f"{prefix}{cont}" for cont in contradictions]
    labels_batch = [completion] + contradictions
    return batch, labels_batch

def prep_batch(ex, tokenizer):
    batch, labels_batch = format_data(ex)
    
    encoding = tokenizer(batch, 
                       padding='max_length',
                       truncation=True,
                       max_length=1024,
                       return_tensors='ms',
                       return_attention_mask=True)
    
    input_ids = encoding['input_ids']
    attention_mask = encoding['attention_mask']
    
    labels_mask = (input_ids != tokenizer.pad_token_id)
    labels = ms.ops.where(labels_mask, input_ids, -100)
    
    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask
    }, labels.astype(ms.int32)  

class CrossEntropyLoss(nn.Cell):
    def __init__(self):
        super().__init__()
        self.loss_fn = nn.CrossEntropyLoss(reduction='none')
        
    def construct(self, logits, labels):
        shift_logits = logits[:, :-1, :]
        shift_labels = labels[:, 1:]
        return self.loss_fn(shift_logits.view(-1, shift_logits.shape[-1]),
                          shift_labels.view(-1))

def run_eval(model, tokenizer, data):
    loss_fn = CrossEntropyLoss()
    all_scores = []
    
    for i, ex in tqdm(enumerate(data), total=len(data)):
        try:
            inputs, labels = prep_batch(ex, tokenizer)
            logits = model(inputs['input_ids'], inputs['attention_mask'])
            
            # 计算损失
            nll = loss_fn(logits, labels).asnumpy()
            valid_tokens = (labels != -100).sum(axis=1).asnumpy()
            scores = nll.reshape(len(labels), -1).sum(axis=1) / valid_tokens
            all_scores.append(scores)
            
            # 进度输出
            if i % 10 == 0 and i > 0:
                current_scores = np.array(all_scores)
                acc = np.mean(np.argmin(current_scores, axis=1) == 0)
                print(f"Processed {i}/{len(data)} | Current Accuracy: {acc:.4f}")
        except Exception as e:
            print(f"Error processing sample {i}: {str(e)}")
            all_scores.append([np.nan]*4)
    
    return np.array(all_scores)

def main(args):
    data = read_data(args.data_file)
    
    tokenizer = load_tokenizer(args.model_name, args.max_tokens)
    model = CausalLMModel(args.model_name)
    
    all_scores = run_eval(model, tokenizer, data)
    

    df = pd.DataFrame(data)
    df['scores'] = list(all_scores)
    valid_scores = df[df['scores'].apply(lambda x: not np.isnan(x).any())]
    
    if len(valid_scores) > 0:
        acc = np.mean(np.argmin(valid_scores['scores'].tolist(), axis=1) == 0)
        print(f"Final Accuracy ({len(valid_scores)} valid samples): {acc:.4f}")
    else:
        print("No valid samples processed")
    
    output_path = os.path.join(args.output_folder, args.model_name.split('/')[-1] + '.jsonl')
    df.to_json(output_path, orient='records', lines=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_file', required=True)
    parser.add_argument('--output_folder', required=True)
    parser.add_argument('--model_name', default='gpt2')
    parser.add_argument('--max_tokens', type=int, default=1024)
    args = parser.parse_args()
    
    main(args)