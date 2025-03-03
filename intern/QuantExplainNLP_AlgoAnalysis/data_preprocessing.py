import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import mindspore as ms
from mindspore import Tensor
from transformers import BigBirdTokenizer
from joblib import Parallel, delayed
import time
import pickle
import os
import random

def generate_sample(seq_length, medical_terms, whole_word_tokens, templates, vocab_size, tokenizer):
    """生成单个样本的文本，突出长序列依赖"""
    np.random.seed()
    words = []
    
    # 添加严重性和治疗术语
    severity_terms = ["mild", "moderate", "severe", "critical", "acute", "chronic"]
    treatment_terms = ["antibiotics", "surgery", "therapy", "medication", "intervention", "procedure"]
    outcome_terms = ["improved", "deteriorated", "stabilized", "resolved", "persisted", "worsened"]
    
    # 随机选择严重性和治疗
    severity = np.random.choice(severity_terms)
    treatment = np.random.choice(treatment_terms)
    outcome = np.random.choice(outcome_terms)
    
    while len(words) < seq_length:
        # 复杂模板，嵌入条件和长距离依赖
        template = np.random.choice(templates)
        terms = [np.random.choice(medical_terms) if np.random.random() < 0.7 else np.random.choice(whole_word_tokens) 
                 for _ in range(5)]
        
        # 添加长距离依赖 - 严重程度与治疗/结果的关联
        if "sepsis" in terms or "trauma" in terms or "stroke" in terms:
            if np.random.random() < 0.8:
                severity = np.random.choice(["severe", "critical", "acute"])
                if np.random.random() < 0.7:
                    treatment = np.random.choice(["emergency surgery", "intensive care", "ventilator support"])
        
        # 替换模板中的变量
        sentence = template.format(
            term1=terms[0], 
            term2=terms[1], 
            term3=terms[2], 
            term4=terms[3], 
            term5=terms[4],
            severity=severity,
            treatment=treatment,
            outcome=outcome
        )
        words.extend(sentence.split())
    
    text = " ".join(words[:seq_length])
    return text

def generate_medical_data(n_samples=5000, vocab_size=1407, seq_length=400, tokenizer=None, n_jobs=-1, cache_dir="data_cache"):
    """生成更真实的模拟医疗数据，优化速度与质量，支持缓存"""
    start_time = time.time()
    
    # 检查缓存
    cache_file = os.path.join(cache_dir, f"data_v2_n{n_samples}_vocab{vocab_size}_seq{seq_length}.pkl")
    if os.path.exists(cache_file):
        print(f"Loading cached data from {cache_file}")
        with open(cache_file, 'rb') as f:
            tfidf_data, texts, labels = pickle.load(f)
        print(f"Loaded tfidf_data shape: {tfidf_data.shape}")
        print(f"Load time: {time.time() - start_time:.2f} seconds")
        # 验证标签分布
        pos_ratio = np.mean(labels)
        print(f"Positive class ratio: {pos_ratio:.2%}")
        return Tensor(tfidf_data, dtype=ms.float32), texts, Tensor(labels, dtype=ms.int32)

    # 初始化 tokenizer
    if tokenizer is None:
        tokenizer = BigBirdTokenizer.from_pretrained('google/bigbird-roberta-base')
    vocab = tokenizer.get_vocab()
    whole_word_tokens = [token for token in vocab.keys() if not token.startswith('##')]
    
    # 扩展医疗术语库
    medical_terms = [
        # 疾病和症状
        "fever", "sepsis", "infection", "pain", "cough", "dyspnea", "fatigue", 
        "nausea", "vomiting", "diarrhea", "headache", "hypertension", "hypotension",
        "tachycardia", "bradycardia", "anemia", "jaundice", "edema", "rash",
        # 医疗场所和人员
        "patient", "hospital", "clinic", "emergency", "icu", "ward", "doctor", 
        "nurse", "physician", "surgeon", "specialist", "consultant", "resident",
        # 治疗和程序
        "treatment", "medication", "antibiotics", "surgery", "procedure", "therapy",
        "ventilator", "intubation", "catheter", "dialysis", "transfusion", "imaging",
        # 状态和结果
        "admitted", "discharged", "transferred", "improved", "deteriorated", "stable",
        "critical", "severe", "moderate", "mild", "acute", "chronic", "terminal",
        # 特定疾病
        "pneumonia", "diabetes", "hypertension", "stroke", "trauma", "cancer", 
        "cardiac", "renal", "hepatic", "respiratory", "neurological", "gastrointestinal"
    ]
    
    # 扩展模板以包含更多临床情境和术语关联
    templates = [
        "Patient with {severity} {term1} and {term2} was admitted for {term3}. Medical history includes {term4}. {treatment} was initiated, resulting in {outcome} {term5}.",
        "Due to {severity} {term1} and {term2}, patient required {treatment} at {term3}. Labs showed {term4} with {outcome} {term5}.",
        "After diagnosis of {severity} {term1}, {term2} occurred, necessitating {treatment} in {term3}. Patient had history of {term4} with {outcome} {term5}.",
        "{severity} {term1} symptoms caused {term2}, managed via {treatment} at {term3} due to {term4} with {outcome} {term5}.",
        "Patient with history of {term1} presented with {severity} {term2}, admitted to {term3} for {treatment} and monitoring of {term4} leading to {outcome} {term5}.",
        "Emergency admission for {severity} {term1} complicated by {term2}. Patient received {treatment} in {term3}. Prior history of {term4} with {outcome} {term5}.",
        "Consultation requested for {severity} {term1} in the setting of {term2}. Patient underwent {treatment} with {term3}. Labs significant for {term4} with {outcome} {term5}.",
        "Follow-up visit for {term1} revealed {severity} {term2} requiring {treatment}. Patient reported {term3} with {term4} leading to {outcome} {term5}."
    ]
    
    # 并行生成文本
    texts = Parallel(n_jobs=n_jobs, backend='loky')(
        delayed(generate_sample)(seq_length, medical_terms, whole_word_tokens, templates, vocab_size, tokenizer) 
        for _ in range(n_samples)
    )
    print(f"Text generation time: {time.time() - start_time:.2f} seconds")

    # TF-IDF 计算
    vectorizer = TfidfVectorizer(max_features=vocab_size, stop_words=None)
    tfidf_data = vectorizer.fit_transform(texts).toarray()
    
    actual_features = tfidf_data.shape[1]
    print(f"Actual TF-IDF features: {actual_features}")
    if actual_features < vocab_size:
        padding = np.zeros((n_samples, vocab_size - actual_features))
        tfidf_data = np.hstack((tfidf_data, padding))
        print(f"Padded TF-IDF to shape: {tfidf_data.shape}")

    # 优化标签生成
    labels = np.zeros(n_samples, dtype=np.int32)
    positive_count = 0
    target_pos_ratio = 0.4  # 目标正类比例
    
    # 第一轮：基于规则分配标签
    for i, text in enumerate(texts):
        text_lower = text.lower()
        
        # 疾病严重程度指标
        severe_terms = sum(1 for term in ["sepsis", "emergency", "icu", "stroke", "trauma", "ventilator", 
                                         "critical", "severe", "acute"] if term in text_lower)
        # 治疗强度指标
        treatment_terms = sum(1 for term in ["antibiotics", "surgery", "intubation", "ventilator", 
                                           "emergency", "intensive"] if term in text_lower)
        # 恢复指标
        recovery_terms = sum(1 for term in ["recovery", "discharged", "improved", "stable", "resolved"] 
                            if term in text_lower)
        # 并发症指标
        complication_terms = sum(1 for term in ["complication", "deteriorated", "worsened", "failure"] 
                               if term in text_lower)
        
        # 严重疾病 + 强治疗 = 高概率正类
        if severe_terms >= 2 and treatment_terms >= 2:
            labels[i] = 1
            positive_count += 1
        # 严重疾病 + 并发症 = 高概率正类
        elif severe_terms >= 2 and complication_terms >= 1:
            labels[i] = 1
            positive_count += 1
        # 严重疾病但有良好恢复 = 负类
        elif severe_terms >= 1 and recovery_terms >= 2:
            labels[i] = 0
        # 轻微疾病 + 恢复良好 = 负类
        elif severe_terms == 0 and recovery_terms >= 1:
            labels[i] = 0
        # 其他情况随机分配，但控制正类总比例
        else:
            current_ratio = positive_count / (i + 1)
            if current_ratio < target_pos_ratio:
                # 增加正类概率
                prob_pos = 0.7
            else:
                # 减少正类概率
                prob_pos = 0.3
            
            if np.random.random() < prob_pos:
                labels[i] = 1
                positive_count += 1
            else:
                labels[i] = 0
    
    # 打印并保证类别平衡
    pos_ratio = positive_count / n_samples
    print(f"Initial positive class ratio: {pos_ratio:.2%}")
    
    # 如果正类比例偏离目标过大，进行调整
    if abs(pos_ratio - target_pos_ratio) > 0.05:
        # 确定需要调整的样本数
        current_pos = np.sum(labels)
        target_pos = int(target_pos_ratio * n_samples)
        
        if current_pos < target_pos:
            # 需要增加正类
            neg_indices = np.where(labels == 0)[0]
            to_change = np.random.choice(neg_indices, target_pos - current_pos, replace=False)
            labels[to_change] = 1
        else:
            # 需要减少正类
            pos_indices = np.where(labels == 1)[0]
            to_change = np.random.choice(pos_indices, current_pos - target_pos, replace=False)
            labels[to_change] = 0
        
        pos_ratio = np.mean(labels)
        print(f"Adjusted positive class ratio: {pos_ratio:.2%}")
    
    # 增强数据：为正类样本添加特定模式
    for i in range(n_samples):
        if labels[i] == 1:
            text_lower = texts[i].lower()
            # 为正类添加更明确的特征模式
            if "sepsis" in text_lower and "treatment" in text_lower:
                texts[i] = texts[i].replace("treatment", "intensive treatment")
            if "severe" in text_lower and not "critical" in text_lower:
                texts[i] = texts[i].replace("severe", "severe and critical")
    
    # 重新计算TF-IDF以反映增强后的文本
    tfidf_data = vectorizer.fit_transform(texts).toarray()
    actual_features = tfidf_data.shape[1]
    if actual_features < vocab_size:
        padding = np.zeros((n_samples, vocab_size - actual_features))
        tfidf_data = np.hstack((tfidf_data, padding))
    
    # 缓存数据
    os.makedirs(cache_dir, exist_ok=True)
    with open(cache_file, 'wb') as f:
        pickle.dump((tfidf_data, texts, labels), f)
    print(f"Enhanced data cached to {cache_file}")

    print(f"Final tfidf_data shape: {tfidf_data.shape}")
    print(f"Final positive class ratio: {np.mean(labels):.2%}")
    print(f"Total time: {time.time() - start_time:.2f} seconds")
    return Tensor(tfidf_data, dtype=ms.float32), texts, Tensor(labels, dtype=ms.int32)

if __name__ == "__main__":
    tokenizer = BigBirdTokenizer.from_pretrained('google/bigbird-roberta-base')
    tfidf_data, texts, labels = generate_medical_data(n_samples=5000, tokenizer=tokenizer, n_jobs=-1)
    print("TF-IDF shape:", tfidf_data.shape)
    print("Sample text:", texts[0][:50], "...")
    print("Labels shape:", labels.shape)
    print("Positive class ratio:", np.mean(labels.asnumpy()))