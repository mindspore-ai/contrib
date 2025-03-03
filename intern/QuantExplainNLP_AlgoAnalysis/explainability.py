import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import Tensor, context
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, precision_recall_curve, average_precision_score
from sklearn.feature_extraction.text import TfidfVectorizer
import os
import pickle
import shap
from models import get_model
from data_preprocessing import generate_medical_data
from train import tokenize_with_limit
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['WenQuanYi Zen Hei']
matplotlib.rcParams['axes.unicode_minus'] = False

def compute_auc(model, data, labels, batch_size=32, input_type=None, cache_file=None, attention_mask=None):
    """计算并缓存 AUC，支持更多评估指标"""
    if cache_file and os.path.exists(cache_file):
        with open(cache_file, 'rb') as f:
            metrics = pickle.load(f)
        print(f"Loaded metrics from {cache_file}: AUC={metrics['auc']:.4f}, AP={metrics['ap']:.4f}")
        return metrics
    
    model.set_train(False)
    if input_type is None:
        input_type = ms.int32 if "BigBird" in str(type(model)) else ms.float32
    n_samples = data.shape[0]
    val_size = int(n_samples * 0.04)  # 减小验证集规模至 4%
    val_data, val_labels = data[-val_size:], labels[-val_size:]
    
    if attention_mask is not None:
        val_attention_mask = attention_mask[-val_size:]
    
    # 创建数据集
    if attention_mask is None:
        dataset_obj = ms.dataset.NumpySlicesDataset(
            {"data": val_data.asnumpy(), "label": val_labels.asnumpy()},
            column_names=["data", "label"],
            shuffle=False
        ).batch(batch_size=batch_size, drop_remainder=True)
    else:
        dataset_obj = ms.dataset.NumpySlicesDataset(
            {"data": val_data.asnumpy(), "mask": val_attention_mask.asnumpy(), "label": val_labels.asnumpy()},
            column_names=["data", "mask", "label"],
            shuffle=False
        ).batch(batch_size=batch_size, drop_remainder=True)

    probs = []
    true_labels = []
    
    for batch in dataset_obj.create_tuple_iterator():
        if attention_mask is None:
            batch_data, batch_labels = batch
            batch_data = Tensor(batch_data, input_type)
            batch_labels = batch_labels.asnumpy()
            logits = model(batch_data)
        else:
            batch_data, batch_mask, batch_labels = batch
            batch_data = Tensor(batch_data, input_type)
            batch_mask = Tensor(batch_mask, ms.float32)
            batch_labels = batch_labels.asnumpy()
            logits = model(batch_data, batch_mask)
            
        batch_probs = ops.sigmoid(logits).asnumpy().flatten()
        probs.extend(batch_probs)
        true_labels.extend(batch_labels)
    
    # 计算多种评估指标
    metrics = {}
    
    # AUC
    auc = roc_auc_score(true_labels, probs)
    metrics['auc'] = auc
    
    # 平均精度 (AP)
    ap = average_precision_score(true_labels, probs)
    metrics['ap'] = ap
    
    # 精确率-召回率数据
    precision, recall, thresholds = precision_recall_curve(true_labels, probs)
    metrics['precision'] = precision
    metrics['recall'] = recall
    metrics['thresholds'] = thresholds
    
    # 最佳阈值（F1最大值点）
    f1_scores = 2 * precision * recall / (precision + recall + 1e-10)
    best_idx = np.argmax(f1_scores)
    metrics['best_threshold'] = thresholds[best_idx] if best_idx < len(thresholds) else 0.5
    metrics['best_f1'] = f1_scores[best_idx]
    
    if cache_file:
        os.makedirs(os.path.dirname(cache_file), exist_ok=True)
        with open(cache_file, 'wb') as f:
            pickle.dump(metrics, f)
        print(f"Saved metrics to {cache_file}: AUC={auc:.4f}, AP={ap:.4f}")
    
    return metrics

def get_feature_names(tfidf_data, texts, max_features=1407):
    """获取TF-IDF特征名称"""
    vectorizer = TfidfVectorizer(max_features=max_features)
    vectorizer.fit(texts)
    feature_names = vectorizer.get_feature_names_out()
    return feature_names

def explain_lr(model, tfidf_data, texts, labels, feature_names=None, sample_idx=0):
    """解释 Logistic Regression 的特征重要性，并显示SHAP值"""
    model.set_train(False)
    sample_data = Tensor(tfidf_data.asnumpy()[sample_idx:sample_idx+1], dtype=ms.float32)
    logits = model(sample_data)
    prob = ops.sigmoid(logits).asnumpy()[0, 0]
    label = "Positive" if prob > 0.5 else "Negative"
    
    # 获取特征名称
    if feature_names is None:
        feature_names = get_feature_names(tfidf_data, texts)
        if len(feature_names) < tfidf_data.shape[1]:
            feature_names = list(feature_names) + [f"Feat_{i}" for i in range(len(feature_names), tfidf_data.shape[1])]
    
    # 获取模型权重
    weights = model.fc.weight.asnumpy().flatten()
    feature_importance = weights
    
    # 计算SHAP值
    explainer = shap.LinearExplainer((weights, 0), tfidf_data.asnumpy())
    shap_values = explainer.shap_values(tfidf_data.asnumpy()[sample_idx:sample_idx+1])
    
    # 找出最重要的特征
    indices = np.argsort(np.abs(shap_values[0]))
    top_indices = indices[-10:]  # 取绝对值最大的10个特征
    top_shap_values = shap_values[0][top_indices]
    top_feature_names = [feature_names[i] if i < len(feature_names) else f"Feat_{i}" for i in top_indices]
    
    # 计算特征权重统计信息
    total_weight_sum = np.sum(np.abs(weights))
    weight_variance = np.var(weights)
    contrib_sum = np.sum(np.abs(weights[top_indices])) / total_weight_sum
    
    # 获取并缓存AUC
    metrics = compute_auc(model, tfidf_data, labels, batch_size=32, input_type=ms.float32, 
                      cache_file="models/cache/lr_metrics.pkl")
    auc = metrics['auc']
    ap = metrics['ap']
    
    # 绘制SHAP特征重要性
    plt.figure(figsize=(12, 6))
    colors = ['red' if v < 0 else 'green' for v in top_shap_values]
    bars = plt.bar(top_feature_names, top_shap_values, color=colors)
    plt.title(f"LR SHAP Values for Top Features\n(Sample {sample_idx}, Prob: {prob:.4f}, Pred: {label}, AUC: {auc:.4f}, AP: {ap:.4f})")
    plt.xlabel("Feature")
    plt.ylabel("SHAP Value (Impact on Prediction)")
    plt.axhline(0, color='black', linewidth=0.5)
    plt.xticks(rotation=45, ha='right')
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval, f"{yval:.3f}", ha='center', va='bottom' if yval > 0 else 'top')
    plt.tight_layout()
    plt.savefig("lr_shap_values.png")
    plt.show()
    
    # 绘制模型权重
    plt.figure(figsize=(12, 6))
    top_weight_indices = np.argsort(np.abs(weights))[-10:]
    top_weights = weights[top_weight_indices]
    top_weight_features = [feature_names[i] if i < len(feature_names) else f"Feat_{i}" for i in top_weight_indices]
    
    colors = ['red' if w < 0 else 'green' for w in top_weights]
    bars = plt.bar(top_weight_features, top_weights, color=colors)
    plt.title(f"LR Model Weights for Top Features\n(Top Features Contribute: {contrib_sum:.2%})")
    plt.xlabel("Feature")
    plt.ylabel("Weight Value")
    plt.axhline(0, color='black', linewidth=0.5)
    plt.xticks(rotation=45, ha='right')
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval, f"{yval:.3f}", ha='center', va='bottom' if yval > 0 else 'top')
    plt.tight_layout()
    plt.savefig("lr_weights.png")
    plt.show()
    
    # 绘制精确率-召回率曲线
    plt.figure(figsize=(10, 6))
    plt.plot(metrics['recall'], metrics['precision'], marker='.', label=f'AP={ap:.4f}')
    plt.axhline(y=np.mean(labels.asnumpy()), color='r', linestyle='--', label=f'Baseline (y={np.mean(labels.asnumpy()):.2f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve (LR)')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig("lr_pr_curve.png")
    plt.show()
    
    print(f"--- LR Explainability Analysis ---")
    print(f"Sample {sample_idx} Text: {texts[sample_idx][:100]}...")
    print(f"True Label: {labels.asnumpy()[sample_idx]}")
    print(f"Predicted Probability: {prob:.4f}")
    print(f"Prediction: {label}")
    print(f"AUC Score: {auc:.4f}")
    print(f"Average Precision (AP): {ap:.4f}")
    print(f"Top Feature Contribution Ratio: {contrib_sum:.2%}")
    print(f"Weight Stats - Min: {np.min(weights):.4f}, Max: {np.max(weights):.4f}, Mean: {np.mean(weights):.4f}")
    print(f"Top Features by SHAP value:")
    for name, value in zip(top_feature_names, top_shap_values):
        print(f"  {name}: {value:.4f}")

def explain_bigbird(model, train_ids, texts, labels, tokenizer, attention_mask=None, sample_idx=0):
    """解释 BigBird 的输入归因，并计算梯度和注意力"""
    model.set_train(False)
    # 修改 explain_bigbird 函数中的索引操作
    sample_ids = Tensor(train_ids.asnumpy()[sample_idx:sample_idx+1], dtype=ms.int32)
    if attention_mask is not None:
        sample_mask = Tensor(attention_mask.asnumpy()[sample_idx:sample_idx+1], dtype=ms.float32)
    else:
        sample_mask = None

    # 定义嵌入和前向函数
    def embedding_fn(ids):
        embeddings = model.embedding(ids)
        # 添加位置编码
        batch_size, seq_length = ids.shape
        position_ids = ops.arange(0, seq_length, dtype=ms.int32)
        position_ids = ops.broadcast_to(position_ids, (batch_size, seq_length))
        position_embeddings = model.position_embedding(position_ids)
        embeddings = embeddings + position_embeddings
        embeddings = model.layer_norm(embeddings)
        embeddings = ops.clip_by_value(embeddings, -5, 5)
        return embeddings

    def forward_fn(embeddings, mask=None):
        embeddings = embeddings.transpose((1, 0, 2))  # [seq_len, batch_size, hidden_size]
        
        padding_length = model.max_length - 400
        src_mask = ops.zeros((model.max_length, model.max_length), dtype=ms.bool_)
        src_mask[:, 400:] = ops.ones((model.max_length, padding_length), dtype=ms.bool_)
        
        transformer_out = model.transformer(embeddings, src_mask)
        pooled = ops.mean(transformer_out, axis=0)
        
        # 使用多层分类头
        pooled = model.fc1(pooled)
        pooled = model.act(pooled)
        pooled = model.dropout(pooled)
        logits = model.fc2(pooled)
        
        return logits

    # 计算预测
    embeddings = embedding_fn(sample_ids)
    logits = forward_fn(embeddings, sample_mask)
    prob = ops.sigmoid(logits).asnumpy()[0, 0]
    label = "Positive" if prob > 0.5 else "Negative"

    # 计算梯度
    grad_fn = ops.GradOperation(get_all=True, sens_param=False)(forward_fn)
    grads = grad_fn(embeddings)[0]
    
    # 计算积分梯度 (Integrated Gradients)
    steps = 50
    baseline = ops.zeros_like(embeddings)
    integrated_grads = ops.zeros_like(grads)
    
    for i in range(steps):
        alpha = i / steps
        interpolated = baseline + alpha * (embeddings - baseline)
        step_grad = grad_fn(interpolated)[0]
        integrated_grads += step_grad
    
    integrated_grads = integrated_grads / steps
    importance = integrated_grads * (embeddings - baseline)
    importance = np.abs(importance.asnumpy()).mean(axis=2).flatten()
    
    # 获取token对应的文本
    tokens = tokenizer.convert_ids_to_tokens(sample_ids.asnumpy()[0])
    
    # 按重要性排序
    importance_scores = []
    for i, token in enumerate(tokens):
        if token == tokenizer.pad_token:
            continue
        importance_scores.append((token, importance[i], i))
    
    # 排序并获取最重要的token
    # 排序并获取最重要的token
    importance_scores.sort(key=lambda x: x[1], reverse=True)
    top_tokens = importance_scores[:10]  # 最重要的10个token
    
    # 计算AUC和其他指标
    metrics = compute_auc(model, train_ids, labels, batch_size=8, input_type=ms.int32, 
                      cache_file="models/cache/bigbird_metrics.pkl", attention_mask=attention_mask)
    auc = metrics['auc']
    ap = metrics['ap']
    
    # 可视化token重要性
    plt.figure(figsize=(14, 7))
    tokens_text = [t[0] for t in top_tokens]
    importance_values = [t[1] for t in top_tokens]
    positions = [t[2] for t in top_tokens]
    
    bars = plt.bar(tokens_text, importance_values, color='blue')
    plt.title(f"BigBird Token Importance (Integrated Gradients)\n(Sample {sample_idx}, Prob: {prob:.4f}, Pred: {label}, AUC: {auc:.4f}, AP: {ap:.4f})")
    plt.xlabel("Token")
    plt.ylabel("Importance Score")
    plt.xticks(rotation=45, ha='right')
    
    # 添加位置标签
    for i, bar in enumerate(bars):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, f"Pos: {positions[i]}", 
                 ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.savefig("bigbird_token_importance.png")
    plt.show()
    
    # 可视化精确率-召回率曲线
    plt.figure(figsize=(10, 6))
    plt.plot(metrics['recall'], metrics['precision'], marker='.', label=f'AP={ap:.4f}')
    plt.axhline(y=np.mean(labels.asnumpy()), color='r', linestyle='--', label=f'Baseline (y={np.mean(labels.asnumpy()):.2f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve (BigBird)')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig("bigbird_pr_curve.png")
    plt.show()
    
    # 可视化文本高亮
    if len(tokens) <= 100:  # 对于较短的文本，显示完整高亮
        text_importance = [(tokens[i], importance[i]) for i in range(len(tokens)) if tokens[i] != tokenizer.pad_token]
        sorted_by_pos = sorted(text_importance, key=lambda x: tokens.index(x[0]))
        
        plt.figure(figsize=(15, 3))
        token_text = [t[0] for t in sorted_by_pos]
        token_imp = [t[1] for t in sorted_by_pos]
        
        # 标准化重要性分数用于颜色映射
        norm_imp = (token_imp - np.min(token_imp)) / (np.max(token_imp) - np.min(token_imp) + 1e-10)
        
        ax = plt.gca()
        ax.set_axis_off()
        
        for i, (token, imp) in enumerate(zip(token_text, norm_imp)):
            color_intensity = min(0.9, 0.1 + imp * 0.8)  # 避免太浅或太深
            plt.text(i, 0, token, fontsize=12, 
                     backgroundcolor=f'rgba(255, 0, 0, {color_intensity})',
                     color='black' if imp < 0.5 else 'white')
            
        plt.xlim(-1, len(token_text))
        plt.ylim(-0.5, 0.5)
        plt.title("Token Importance Visualization")
        plt.tight_layout()
        plt.savefig("bigbird_text_highlight.png")
        plt.show()
    
    # 比较LR和BigBird的性能
    print(f"--- BigBird Explainability Analysis ---")
    print(f"Sample {sample_idx} Text: {texts[sample_idx][:100]}...")
    print(f"True Label: {labels.asnumpy()[sample_idx]}")
    print(f"Predicted Probability: {prob:.4f}")
    print(f"Prediction: {label}")
    print(f"AUC Score: {auc:.4f}")
    print(f"Average Precision (AP): {ap:.4f}")
    print(f"Top 10 Important Tokens:")
    
    for i, (token, importance_value, position) in enumerate(top_tokens):
        print(f"  {i+1}. '{token}' (Position {position}): {importance_value:.6f}")
    
    # 计算总梯度和方差
    total_grad_sum = np.sum(importance)
    grad_variance = np.var(importance)
    print(f"Total Gradient Sum: {total_grad_sum:.6f}")
    print(f"Gradient Variance: {grad_variance:.6f}")

def compare_models(lr_model, bigbird_model, tfidf_data, train_ids, attention_mask, labels, texts):
    """比较LR和BigBird模型的性能和解释性"""
    # 获取模型指标
    lr_metrics = compute_auc(lr_model, tfidf_data, labels, batch_size=32, input_type=ms.float32, 
                          cache_file="models/cache/lr_metrics.pkl")
    bigbird_metrics = compute_auc(bigbird_model, train_ids, labels, batch_size=8, input_type=ms.int32, 
                              cache_file="models/cache/bigbird_metrics.pkl", attention_mask=attention_mask)
    
    # 计算各个阈值下的预测结果
    def get_predictions(model, data, mask=None, threshold=0.5):
        if mask is None:
            logits = model(data)
        else:
            logits = model(data, mask)
        probs = ops.sigmoid(logits).asnumpy().flatten()
        preds = (probs > threshold).astype(np.int32)
        return probs, preds
    
    # 获取样本预测
    n_samples = min(5, tfidf_data.shape[0])
    sample_indices = np.random.choice(tfidf_data.shape[0], n_samples, replace=False)
    
    lr_probs, lr_preds = get_predictions(lr_model, tfidf_data[sample_indices])
    bb_probs, bb_preds = get_predictions(bigbird_model, train_ids[sample_indices], attention_mask[sample_indices])
    true_labels = labels.asnumpy()[sample_indices]
    
    # 绘制性能对比图
    plt.figure(figsize=(14, 10))
    
    # AUC对比
    plt.subplot(2, 2, 1)
    models = ['Logistic Regression', 'BigBird']
    auc_values = [lr_metrics['auc'], bigbird_metrics['auc']]
    bars = plt.bar(models, auc_values, color=['#446B7C', '#E65D2E'])
    plt.title('AUC Comparison')
    plt.ylabel('AUC')
    plt.ylim(0.5, 1.0)
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.4f}', ha='center', va='bottom')
    
    # AP对比
    plt.subplot(2, 2, 2)
    ap_values = [lr_metrics['ap'], bigbird_metrics['ap']]
    bars = plt.bar(models, ap_values, color=['#446B7C', '#E65D2E'])
    plt.title('Average Precision Comparison')
    plt.ylabel('Average Precision')
    plt.ylim(0, 1.0)
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.4f}', ha='center', va='bottom')
    
    # PR曲线对比
    plt.subplot(2, 2, 3)
    plt.plot(lr_metrics['recall'], lr_metrics['precision'], 
             label=f'LR (AP={lr_metrics["ap"]:.4f})', color='#446B7C')
    plt.plot(bigbird_metrics['recall'], bigbird_metrics['precision'], 
             label=f'BigBird (AP={bigbird_metrics["ap"]:.4f})', color='#E65D2E')
    plt.axhline(y=np.mean(labels.asnumpy()), color='gray', linestyle='--', 
                label=f'Baseline ({np.mean(labels.asnumpy()):.2f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curves')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # 样本预测对比
    plt.subplot(2, 2, 4)
    width = 0.35
    x = np.arange(n_samples)
    plt.bar(x - width/2, lr_probs, width, label='LR Prob', color='#446B7C')
    plt.bar(x + width/2, bb_probs, width, label='BigBird Prob', color='#E65D2E')
    plt.scatter(x, true_labels, color='black', marker='*', s=100, label='True Label')
    plt.xlabel('Sample Index')
    plt.ylabel('Prediction Probability')
    plt.title('Model Predictions Comparison')
    plt.xticks(x)
    plt.ylim(0, 1.0)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig("model_comparison.png")
    plt.show()
    
    # 打印详细比较
    print("\n=== Model Performance Comparison ===")
    print(f"{'Metric':<20} {'Logistic Regression':<20} {'BigBird':<20} {'Difference':<20}")
    print(f"{'-'*70}")
    print(f"{'AUC':<20} {lr_metrics['auc']:<20.4f} {bigbird_metrics['auc']:<20.4f} {bigbird_metrics['auc']-lr_metrics['auc']:<20.4f}")
    print(f"{'Average Precision':<20} {lr_metrics['ap']:<20.4f} {bigbird_metrics['ap']:<20.4f} {bigbird_metrics['ap']-lr_metrics['ap']:<20.4f}")
    print(f"{'Best F1':<20} {lr_metrics['best_f1']:<20.4f} {bigbird_metrics['best_f1']:<20.4f} {bigbird_metrics['best_f1']-lr_metrics['best_f1']:<20.4f}")
    
    # 样本预测比较
    print("\n=== Sample Predictions ===")
    print(f"{'Sample':<10} {'True Label':<12} {'LR Prob':<12} {'LR Pred':<12} {'BigBird Prob':<15} {'BigBird Pred':<12} {'Agreement':<10}")
    print(f"{'-'*90}")
    for i, idx in enumerate(sample_indices):
        agreement = "Yes" if lr_preds[i] == bb_preds[i] else "No"
        print(f"{idx:<10} {true_labels[i]:<12} {lr_probs[i]:<12.4f} {lr_preds[i]:<12} {bb_probs[i]:<15.4f} {bb_preds[i]:<12} {agreement:<10}")
    
    # 返回性能差异
    return {
        'auc_diff': bigbird_metrics['auc'] - lr_metrics['auc'],
        'ap_diff': bigbird_metrics['ap'] - lr_metrics['ap'],
        'f1_diff': bigbird_metrics['best_f1'] - lr_metrics['best_f1']
    }

def main():
    """主函数：加载模型和数据，进行解释性分析"""
    context.set_context(mode=context.GRAPH_MODE, device_target="CPU")
    
    # 加载数据
    tfidf_data, text_data, labels = generate_medical_data(n_samples=5000)
    
    # 加载模型
    lr_model = get_model("lr")
    bigbird_model = get_model("bigbird")
    tokenizer = bigbird_model.tokenizer
    
    # 标记化文本
    train_ids, attention_mask = tokenize_with_limit(text_data, tokenizer, max_length=512, vocab_size=50304)
    
    # 加载检查点
    lr_ckpt = "models/checkpoints/lr_checkpoint.ckpt"
    if os.path.exists(lr_ckpt):
        ms.load_checkpoint(lr_ckpt, lr_model)
        print(f"Loaded LR model from {lr_ckpt}")
    
    bigbird_ckpt_dir = "models/checkpoints"
    best_bigbird_ckpt = os.path.join(bigbird_ckpt_dir, "bigbird_checkpoint_best.ckpt")
    if os.path.exists(best_bigbird_ckpt):
        ms.load_checkpoint(best_bigbird_ckpt, bigbird_model)
        print(f"Loaded best BigBird model from {best_bigbird_ckpt}")
    else:
        # 尝试加载最新的epoch检查点
        latest_bigbird_ckpt = max([f for f in os.listdir(bigbird_ckpt_dir) if f.startswith("bigbird_checkpoint_epoch_")], 
                                key=lambda x: int(x.split('_')[-1].split('.')[0]), default=None)
        if latest_bigbird_ckpt:
            full_path = os.path.join(bigbird_ckpt_dir, latest_bigbird_ckpt)
            ms.load_checkpoint(full_path, bigbird_model)
            print(f"Loaded BigBird model from {full_path}")
    
    # 获取TF-IDF特征名称
    feature_names = get_feature_names(tfidf_data, text_data)
    
    # 随机选择一个正样本和一个负样本进行解释
    pos_indices = np.where(labels.asnumpy() == 1)[0]
    neg_indices = np.where(labels.asnumpy() == 0)[0]
    
    if len(pos_indices) > 0 and len(neg_indices) > 0:
        pos_idx = np.random.choice(pos_indices)
        neg_idx = np.random.choice(neg_indices)
        
        print("\n=== Explaining Positive Sample ===")
        print(f"Sample ID: {pos_idx}, Label: Positive")
        
        # 解释逻辑回归模型（正样本）
        print("\nLogistic Regression Explanation (Positive Sample):")
        explain_lr(lr_model, tfidf_data, text_data, labels, feature_names, sample_idx=pos_idx)
        
        # 解释BigBird模型（正样本）
        print("\nBigBird Explanation (Positive Sample):")
        explain_bigbird(bigbird_model, train_ids, text_data, labels, tokenizer, attention_mask, sample_idx=pos_idx)
        
        print("\n=== Explaining Negative Sample ===")
        print(f"Sample ID: {neg_idx}, Label: Negative")
        
        # 解释逻辑回归模型（负样本）
        print("\nLogistic Regression Explanation (Negative Sample):")
        explain_lr(lr_model, tfidf_data, text_data, labels, feature_names, sample_idx=neg_idx)
        
        # 解释BigBird模型（负样本）
        print("\nBigBird Explanation (Negative Sample):")
        explain_bigbird(bigbird_model, train_ids, text_data, labels, tokenizer, attention_mask, sample_idx=neg_idx)
    
    # 模型比较
    print("\n=== Overall Model Comparison ===")
    diff_metrics = compare_models(lr_model, bigbird_model, tfidf_data, train_ids, attention_mask, labels, text_data)
    
    # 输出结论
    if diff_metrics['auc_diff'] > 0.03:  # AUC提升超过0.03
        print("\nConclusion: BigBird model significantly outperforms Logistic Regression in both performance and explainability.")
        print(f"AUC improvement: +{diff_metrics['auc_diff']:.4f}")
        print("BigBird can capture complex text patterns and long-range dependencies that LR model cannot detect.")
    elif diff_metrics['auc_diff'] > 0:  # AUC有提升但不显著
        print("\nConclusion: BigBird model slightly outperforms Logistic Regression.")
        print(f"AUC improvement: +{diff_metrics['auc_diff']:.4f}")
        print("The improvement suggests BigBird captures some text patterns that LR model misses.")
    else:  # AUC无提升或下降
        print("\nConclusion: BigBird model does not outperform Logistic Regression in this specific task.")
        print(f"AUC difference: {diff_metrics['auc_diff']:.4f}")
        print("Possible reasons:")
        print("1. The simulated data may not contain enough complex patterns that require deep learning models.")
        print("2. BigBird model may need further optimization in architecture or training process.")
        print("3. The task may be linearly separable, making LR sufficient.")
    
    print("\nSuggestions for further improvement:")
    print("1. Generate more realistic medical data with stronger sequential patterns")
    print("2. Increase training epochs for BigBird with learning rate adjustments")
    print("3. Experiment with different model architectures and hyperparameters")
    print("4. Apply more advanced explainability techniques like attention visualization")

if __name__ == "__main__":
    main()