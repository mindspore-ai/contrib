import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import Tensor, context, dataset
import numpy as np
import os
from data_preprocessing import generate_medical_data
from models import get_model
from sklearn.metrics import roc_auc_score

MAX_LENGTH = 512

def evaluate(model, val_data, val_labels, batch_size=32, input_type=ms.float32, attention_mask=None):
    """评估模型性能"""
    model.set_train(False)
    n_samples = val_data.shape[0]
    n_batches = n_samples // batch_size + (1 if n_samples % batch_size > 0 else 0)
    
    all_preds = []
    all_labels = []
    
    for i in range(n_batches):
        start = i * batch_size
        end = min(start + batch_size, n_samples)
        batch_data = val_data[start:end].astype(input_type)
        batch_labels = val_labels[start:end]
        
        # 添加对 attention_mask 的处理
        batch_mask = None
        if attention_mask is not None:
            batch_mask = attention_mask[start:end]
        
        # 使用 attention_mask 进行预测
        logits = model(batch_data, batch_mask) if batch_mask is not None else model(batch_data)
        preds = ops.sigmoid(logits).asnumpy().flatten()
        
        all_preds.extend(preds)
        all_labels.extend(batch_labels.asnumpy())
    
    # 计算准确率和AUC
    binary_preds = np.array(all_preds) > 0.5
    accuracy = np.mean(binary_preds == np.array(all_labels))
    
    try:
        auc = roc_auc_score(all_labels, all_preds)
    except:
        auc = 0.5  # 如果计算AUC失败（例如只有一个类别），则返回0.5
    
    return accuracy, auc

def tokenize_with_limit(texts, tokenizer, max_length=512, vocab_size=50304):
    """标记化文本，并处理超出词汇表范围的token"""
    encodings = tokenizer(
        texts,
        max_length=max_length,
        padding='max_length',
        truncation=True,
        return_tensors='np'
    )
    input_ids = encodings['input_ids']
    attention_mask = encodings['attention_mask']
    
    # 确保所有token ID在词汇表范围内
    input_ids = np.where(input_ids < vocab_size, input_ids, tokenizer.pad_token_id)
    
    return Tensor(input_ids, dtype=ms.int32), Tensor(attention_mask, dtype=ms.float32)

def train_lr(model, train_data, train_labels, batch_size=32, epochs=5, lr=0.0002, save_dir="models/checkpoints"):
    """训练逻辑回归模型，使用优化的学习率和正则化"""
    # 使用带权重衰减的优化器
    optimizer = nn.Adam(model.trainable_params(), learning_rate=lr, weight_decay=0.001)
    loss_fn = nn.BCEWithLogitsLoss()

    @ms.jit(compile_once=True)
    def forward_fn(data, label):
        logits = model(data)
        loss = loss_fn(logits, label.reshape(-1, 1).astype(ms.float32))
        return loss, logits

    grad_fn = ops.value_and_grad(forward_fn, None, optimizer.parameters)

    def train_step(data, label):
        (loss, _), grads = grad_fn(data, label)
        # 梯度裁剪防止梯度爆炸
        grads = ops.clip_by_norm(grads, max_norm=1.0)
        optimizer(grads)
        return loss

    model.set_train()
    n_samples = train_data.shape[0]
    train_size = int(n_samples * 0.8)
    train_data, train_labels = train_data[:train_size], train_labels[:train_size]
    
    # 创建数据集
    dataset_obj = ms.dataset.NumpySlicesDataset(
        {"data": train_data.asnumpy(), "label": train_labels.asnumpy()},
        column_names=["data", "label"],
        shuffle=True
    ).batch(batch_size=batch_size, drop_remainder=True)

    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "lr_checkpoint.ckpt")

    # 加载预训练检查点（如果存在）
    if os.path.exists(save_path):
        print(f"Loading pre-trained LR model from {save_path}")
        ms.load_checkpoint(save_path, model)

    # 初始评估
    initial_accuracy, initial_auc = evaluate(model, train_data, train_labels, batch_size, ms.float32)
    print(f"Initial LR Validation - Accuracy: {initial_accuracy:.4f}, AUC: {initial_auc:.4f}")
    best_auc = initial_auc

    print("Training Logistic Regression...")
    total_batches = len(train_data) // batch_size
    for epoch in range(epochs):
        total_loss = 0
        steps = 0
        for batch in dataset_obj.create_tuple_iterator():
            data, label = batch
            data = Tensor(data, ms.float32)
            label = Tensor(label, ms.int32)
            loss = train_step(data, label)
            if np.isnan(loss.asnumpy()):
                print(f"NaN detected at Epoch {epoch + 1}, stopping training...")
                return
            total_loss += loss.asnumpy()
            steps += 1
            progress = (steps / total_batches) * 100
            print(f"\rEpoch {epoch + 1}/{epochs}, Progress: {progress:.2f}%", end="")
        
        avg_loss = total_loss / steps
        val_accuracy, val_auc = evaluate(model, train_data, train_labels, batch_size, ms.float32)
        print(f"\nEpoch {epoch + 1}/{epochs}, Avg Loss: {avg_loss:.4f}, Val Accuracy: {val_accuracy:.4f}, AUC: {val_auc:.4f}")
        
        # 保存最佳模型
        if val_auc > best_auc:
            best_auc = val_auc
            ms.save_checkpoint(model, save_path)
            print(f"Best LR model saved to {save_path} with AUC: {best_auc:.4f}")

def train_bigbird(model, train_ids, train_labels, attention_mask=None, batch_size=8, epochs=10, lr=0.00001, save_dir="models/checkpoints"):
    """训练BigBird模型，使用固定学习率"""
    model.set_train()
    
    # 将 MindSpore 张量转换为 NumPy 数组
    train_ids_np = train_ids.asnumpy()
    train_labels_np = train_labels.asnumpy()
    attention_mask_np = attention_mask.asnumpy() if attention_mask is not None else None
    
    n_samples = train_ids_np.shape[0]
    train_size = int(n_samples * 0.8)
    val_size = n_samples - train_size

    # 划分训练集和验证集
    train_indices = np.random.choice(n_samples, train_size, replace=False)
    val_indices = np.array([i for i in range(n_samples) if i not in train_indices])
    
    # 准备训练数据 - 使用 NumPy 数组索引
    train_data_np = train_ids_np[train_indices]
    train_att_mask_np = attention_mask_np[train_indices] if attention_mask_np is not None else None
    train_label_np = train_labels_np[train_indices]
    
    # 准备验证数据 - 使用 NumPy 数组索引
    val_data_np = train_ids_np[val_indices]
    val_att_mask_np = attention_mask_np[val_indices] if attention_mask_np is not None else None
    val_label_np = train_labels_np[val_indices]
    
    # 将 NumPy 数组转换回 MindSpore 张量
    train_data = Tensor(train_data_np, dtype=ms.int32)
    train_att_mask = Tensor(train_att_mask_np, dtype=ms.float32) if train_att_mask_np is not None else None
    train_label = Tensor(train_label_np, dtype=ms.int32)
    
    val_data = Tensor(val_data_np, dtype=ms.int32)
    val_att_mask = Tensor(val_att_mask_np, dtype=ms.float32) if val_att_mask_np is not None else None
    val_label = Tensor(val_label_np, dtype=ms.int32)
    
    # 评估初始模型
    model.set_train(False)
    initial_acc, initial_auc = evaluate(model, val_data, val_label, batch_size=batch_size, 
                                       input_type=ms.int32)  # 移除 attention_mask 参数
    print(f"Initial BigBird Validation - Accuracy: {initial_acc:.4f}, AUC: {initial_auc:.4f}")
    print("Training BigBird...")
    model.set_train(True)
    
    # 使用固定学习率
    optimizer = nn.Adam(model.trainable_params(), learning_rate=lr, weight_decay=0.01)
    loss_fn = nn.BCEWithLogitsLoss()

    @ms.jit(compile_once=True)
    def forward_fn(ids, attention_mask, label):
        logits = model(ids, attention_mask)
        loss = loss_fn(logits, label.reshape(-1, 1).astype(ms.float32))
        return loss, logits

    grad_fn = ops.value_and_grad(forward_fn, None, optimizer.parameters)

    def train_step(ids, attention_mask, label, step_idx):
        (loss, _), grads = grad_fn(ids, attention_mask, label)
        # 增加梯度裁剪阈值
        grads = ops.clip_by_norm(grads, max_norm=2.0)
        optimizer(grads)
        return loss  # 只返回损失值
    
    # 创建保存目录
    os.makedirs(save_dir, exist_ok=True)
    
    # 初始化最佳验证指标
    best_auc = 0.0
    
    # 训练循环
    steps_per_epoch = train_size // batch_size
    global_step = 0
    total_steps = steps_per_epoch * epochs
    
    for epoch in range(epochs):
        # 打乱训练数据 - 使用 NumPy 数组
        # 重要修改：直接打乱训练数据的索引，范围是 [0, train_size)
        batches = np.random.permutation(train_size)
        epoch_loss = 0
        
        for step in range(steps_per_epoch):
            batch_indices = batches[step * batch_size:(step + 1) * batch_size]
            
            # 修改：直接使用 batch_indices 索引训练数据，不再重复应用 train_indices
            ids = Tensor(train_data_np[batch_indices], dtype=ms.int32)
            mask = Tensor(train_att_mask_np[batch_indices], dtype=ms.float32) if train_att_mask_np is not None else None
            label = Tensor(train_label_np[batch_indices], dtype=ms.int32)
            
            # 训练步骤
            loss = train_step(ids, mask, label, global_step)
            
            if np.isnan(loss.asnumpy()):
                print(f"NaN detected at Epoch {epoch + 1}, Step {step + 1}, stopping training...")
                return model
                
            epoch_loss += loss.asnumpy()
            global_step += 1
            
            # 显示进度
            progress = (step + 1) / steps_per_epoch * 100
            print(f"\rEpoch {epoch + 1}/{epochs}, Progress: {progress:.2f}%", end="")
        
        # 计算平均损失
        avg_loss = epoch_loss / steps_per_epoch
        
        # 在验证集上评估
        model.set_train(False)
        val_acc, val_auc = evaluate(model, val_data, val_label, batch_size=batch_size, 
                                   input_type=ms.int32)  # 移除 attention_mask 参数
        model.set_train(True)
        
        print(f"\nEpoch {epoch + 1}/{epochs}, Avg Loss: {avg_loss:.4f}, Val Accuracy: {val_acc:.4f}, AUC: {val_auc:.4f}")
        
        # 保存当前模型
        current_ckpt = os.path.join(save_dir, f"bigbird_checkpoint_epoch_{epoch + 1}.ckpt")
        ms.save_checkpoint(model, current_ckpt)
        print(f"Saved checkpoint to {current_ckpt}")
        
        # 保存最佳模型
        if val_auc > best_auc:
            best_auc = val_auc
            best_ckpt = os.path.join(save_dir, "bigbird_checkpoint_best.ckpt")
            ms.save_checkpoint(model, best_ckpt)
            print(f"New best model with AUC: {best_auc:.4f}")
        
        # 清理内存
        import gc
        gc.collect()
    
    return model












def pretrain_bigbird(model, n_samples=10000, epochs=2, lr=0.0001, batch_size=16, save_dir="models/pretrained"):
    """使用合成数据预训练BigBird模型，学习医疗文本基本模式"""
    from transformers import BigBirdTokenizer
    import random
    
    print("Generating synthetic data for pretraining...")
    
    # 医疗术语库
    medical_terms = [
        "fever", "sepsis", "infection", "pain", "cough", "dyspnea", "fatigue", 
        "nausea", "vomiting", "diarrhea", "headache", "hypertension", "hypotension",
        "tachycardia", "bradycardia", "anemia", "jaundice", "edema", "rash",
        "patient", "hospital", "clinic", "emergency", "icu", "ward", "doctor", 
        "nurse", "physician", "surgeon", "specialist", "consultant", "resident",
        "treatment", "medication", "antibiotics", "surgery", "procedure", "therapy",
        "ventilator", "intubation", "catheter", "dialysis", "transfusion", "imaging",
        "admitted", "discharged", "transferred", "improved", "deteriorated", "stable",
        "critical", "severe", "moderate", "mild", "acute", "chronic", "terminal",
        "pneumonia", "diabetes", "hypertension", "stroke", "trauma", "cancer", 
        "cardiac", "renal", "hepatic", "respiratory", "neurological", "gastrointestinal"
    ]
    
    # 严重性词汇
    severity_terms = ["mild", "moderate", "severe", "critical", "acute", "chronic"]
    
    # 治疗词汇
    treatment_terms = ["antibiotics", "surgery", "therapy", "medication", "intervention", 
                       "procedure", "ventilator", "intubation", "catheter", "dialysis"]
    
    # 结果词汇
    outcome_terms = ["improved", "deteriorated", "stabilized", "resolved", "persisted", "worsened"]
    
    # 医疗文本模板
    templates = [
        "Patient with {severity} {condition1} and {condition2} was admitted for {treatment}. Medical history includes {history}. {outcome}.",
        "Due to {severity} {condition1}, patient required {treatment} in the {location}. Labs showed {lab_result} with {outcome}.",
        "After diagnosis of {severity} {condition1}, {condition2} occurred, necessitating {treatment}. {outcome}.",
        "{severity} {condition1} symptoms caused {condition2}, managed via {treatment} due to {reason}. {outcome}.",
        "Patient with history of {history} presented with {severity} {condition1}, admitted to {location} for {treatment}. {outcome}."
    ]
    
    # 生成合成数据
    synthetic_texts = []
    synthetic_labels = []
    
    for _ in range(n_samples):
        # 随机选择模板
        template = random.choice(templates)
        
        # 随机选择术语
        severity = random.choice(severity_terms)
        condition1 = random.choice(medical_terms)
        condition2 = random.choice(medical_terms)
        treatment = random.choice(treatment_terms)
        history = random.choice(medical_terms)
        location = random.choice(["emergency room", "ICU", "ward", "clinic"])
        lab_result = random.choice(["elevated WBC", "abnormal liver enzymes", "decreased hemoglobin", "elevated creatinine"])
        reason = random.choice(["worsening symptoms", "comorbidities", "complications", "patient preference"])
        outcome = f"Patient {random.choice(outcome_terms)}"
        
        # 填充模板
        text = template.format(
            severity=severity,
            condition1=condition1,
            condition2=condition2,
            treatment=treatment,
            history=history,
            location=location,
            lab_result=lab_result,
            reason=reason,
            outcome=outcome
        )
        
        # 生成标签 - 基于关键词和严重性
        if (("severe" in text or "critical" in text) and 
            ("sepsis" in text or "respiratory" in text or "cardiac" in text or "stroke" in text) and
            not "improved" in text):
            label = 1  # 严重疾病
        elif "mild" in text and "improved" in text:
            label = 0  # 轻微疾病且改善
        elif random.random() < 0.4:  # 控制正类比例约为40%
            label = 1
        else:
            label = 0
            
        synthetic_texts.append(text)
        synthetic_labels.append(label)
    
    # 显示样本分布
    pos_ratio = sum(synthetic_labels) / len(synthetic_labels)
    print(f"Generated {n_samples} synthetic texts with {pos_ratio:.2%} positive samples")
    print(f"Sample text: {synthetic_texts[0]}")
    
    # 转换为模型输入
    tokenizer = model.tokenizer
    input_ids_list = []
    attention_mask_list = []
    
    for text in synthetic_texts:
        encodings = tokenizer(
            text,
            max_length=MAX_LENGTH,
            padding='max_length',
            truncation=True,
            return_tensors='np'
        )
        input_ids_list.append(encodings['input_ids'][0])
        attention_mask_list.append(encodings['attention_mask'][0])
    
    input_ids = Tensor(np.array(input_ids_list), dtype=ms.int32)
    attention_mask = Tensor(np.array(attention_mask_list), dtype=ms.float32)
    labels = Tensor(np.array(synthetic_labels), dtype=ms.int32)
    
    # 创建数据集
    dataset_obj = ms.dataset.NumpySlicesDataset(
        {
            "ids": input_ids.asnumpy(),
            "attention_mask": attention_mask.asnumpy(),
            "label": labels.asnumpy()
        },
        column_names=["ids", "attention_mask", "label"],
        shuffle=True
    ).batch(batch_size=batch_size, drop_remainder=True)
    
    # 设置优化器和损失函数
    optimizer = nn.Adam(model.trainable_params(), learning_rate=lr, weight_decay=0.01)
    loss_fn = nn.BCEWithLogitsLoss()
    
    @ms.jit(compile_once=True)
    def forward_fn(ids, att_mask, label):
        logits = model(ids, att_mask)
        loss = loss_fn(logits, label.reshape(-1, 1).astype(ms.float32))
        return loss, logits
    
    grad_fn = ops.value_and_grad(forward_fn, None, optimizer.parameters)
    
    def train_step(ids, att_mask, label):
        (loss, _), grads = grad_fn(ids, att_mask, label)
        grads = ops.clip_by_norm(grads, max_norm=2.0)
        optimizer(grads)
        return loss
    
    # 预训练
    model.set_train()
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "bigbird_pretrained.ckpt")
    
    print("Pretraining BigBird...")
    total_batches = len(synthetic_texts) // batch_size
    
    for epoch in range(epochs):
        total_loss = 0
        steps = 0
        for batch in dataset_obj.create_tuple_iterator():
            ids, att_mask, label = batch
            loss = train_step(ids, att_mask, label)
            
            if np.isnan(loss.asnumpy()):
                print(f"NaN detected at Epoch {epoch + 1}, stopping pretraining...")
                return model
                
            total_loss += loss.asnumpy()
            steps += 1
            progress = (steps / total_batches) * 100
            print(f"\rEpoch {epoch + 1}/{epochs}, Progress: {progress:.2f}%", end="")
        
        avg_loss = total_loss / steps
        print(f"\nEpoch {epoch + 1}/{epochs}, Avg Loss: {avg_loss:.4f}")
    
    # 保存预训练模型
    ms.save_checkpoint(model, save_path)
    print(f"Pretrained BigBird model saved to {save_path}")
    
    return model

def main():
    context.set_context(mode=context.GRAPH_MODE, device_target="CPU")

    # 加载模型
    bigbird_model = get_model("bigbird")
    tokenizer = bigbird_model.tokenizer

    # 生成医疗数据
    tfidf_data, text_data, labels = generate_medical_data(n_samples=5000, tokenizer=tokenizer)

    # 标记化文本数据
    train_ids, attention_mask = tokenize_with_limit(text_data, tokenizer, max_length=512, vocab_size=50304)

    # 查看类别分布
    pos_ratio = np.mean(labels.asnumpy())
    print(f"Dataset positive class ratio: {pos_ratio:.2%}")

    # 加载逻辑回归模型
    lr_model = get_model("lr")

    # 预训练BigBird模型（可选）
    do_pretrain = False  # 设置为True开启预训练
    if do_pretrain:
        print("\nPretraining BigBird on synthetic data...")
        bigbird_model = pretrain_bigbird(bigbird_model, n_samples=2000, epochs=2, lr=0.0001)

    # 训练逻辑回归
    print("\nTraining Logistic Regression...")
    train_lr(lr_model, tfidf_data, labels, batch_size=32, epochs=5, lr=0.0002)

    # 训练BigBird
    print("\nTraining BigBird...")
    train_bigbird(bigbird_model, train_ids, labels, attention_mask=attention_mask, batch_size=8, epochs=2, lr=0.00001)

    # 输出训练后的结果
    lr_model.set_train(False)
    bigbird_model.set_train(False)
    
    print("\nPost-training outputs:")
    # 评估LR模型
    lr_preds = ops.sigmoid(lr_model(tfidf_data[:5])).asnumpy()
    print("LR predictions:", lr_preds.flatten())
    
    # 评估BigBird模型
    bigbird_preds = ops.sigmoid(bigbird_model(train_ids[:5], attention_mask[:5])).asnumpy()
    print("BigBird predictions:", bigbird_preds.flatten())
    
    # 计算最终AUC
    _, lr_auc = evaluate(lr_model, tfidf_data, labels, batch_size=32, input_type=ms.float32)
    _, bb_auc = evaluate(bigbird_model, train_ids, labels, batch_size=8, input_type=ms.int32)
    
    print(f"\nFinal AUC scores:")
    print(f"Logistic Regression AUC: {lr_auc:.4f}")
    print(f"BigBird AUC: {bb_auc:.4f}")
    print(f"AUC Improvement: {bb_auc - lr_auc:.4f}")

if __name__ == "__main__":
    main()