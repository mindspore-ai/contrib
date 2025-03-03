import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
import numpy as np
from sklearn.linear_model import LogisticRegression as SklearnLR
from interpret.glassbox import ExplainableBoostingClassifier
from transformers import BigBirdTokenizer

INPUT_DIM = 1407
VOCAB_SIZE = 50304
HIDDEN_SIZE = 768
NUM_LAYERS = 12
NUM_HEADS = 12
MAX_LENGTH = 512
SEQ_LENGTH = 400

class LogisticRegression(nn.Cell):
    def __init__(self, input_dim=INPUT_DIM):
        super(LogisticRegression, self).__init__()
        self.fc = nn.Dense(input_dim, 1, weight_init='xavier_uniform', bias_init='zeros')
        # 添加L2正则化
        self.dropout = nn.Dropout(0.2)

    def construct(self, x):
        x = self.dropout(x)
        return self.fc(x)

class SklearnModelWrapper(nn.Cell):
    def __init__(self, sklearn_model):
        super(SklearnModelWrapper, self).__init__()
        self.model = sklearn_model
        self.is_trained = False

    def fit(self, x, y):
        x_np = x.asnumpy()
        y_np = y.asnumpy()
        self.model.fit(x_np, y_np)
        self.is_trained = True

    def construct(self, x):
        if not self.is_trained:
            raise ValueError("Model must be trained with fit() before inference")
        x_np = x.asnumpy()
        pred = self.model.predict_proba(x_np)[:, 1]
        return ms.Tensor(pred, dtype=ms.float32)

class BigBirdModel(nn.Cell):
    def __init__(self, vocab_size=VOCAB_SIZE, hidden_size=HIDDEN_SIZE, 
                 num_layers=NUM_LAYERS, num_heads=NUM_HEADS, max_length=MAX_LENGTH):
        super(BigBirdModel, self).__init__()
        # 优化嵌入层初始化
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        fan_in, fan_out = vocab_size, hidden_size
        limit = np.sqrt(6 / (fan_in + fan_out))
        embedding_table = ms.Tensor(np.random.uniform(-limit, limit, (vocab_size, hidden_size)), dtype=ms.float32)
        self.embedding.embedding_table.set_data(embedding_table)
        
        # 添加位置编码
        self.position_embedding = nn.Embedding(max_length, hidden_size)
        position_table = ms.Tensor(np.random.uniform(-limit, limit, (max_length, hidden_size)), dtype=ms.float32)
        self.position_embedding.embedding_table.set_data(position_table)
        
        self.tokenizer = BigBirdTokenizer.from_pretrained('google/bigbird-roberta-base')
        pad_token_id = self.tokenizer.pad_token_id
        self.embedding.embedding_table[pad_token_id] = ops.zeros(hidden_size, dtype=ms.float32)
        
        # 添加层归一化和Dropout
        self.layer_norm = nn.LayerNorm([hidden_size])
        self.dropout = nn.Dropout(0.2)
        
        # 使用更好的激活函数
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=hidden_size * 4,
            dropout=0.2,
            activation='gelu'  # 使用GELU激活函数
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=num_layers
        )
        
        # 添加多层分类头
        self.fc1 = nn.Dense(hidden_size, hidden_size // 2, weight_init='xavier_uniform')
        self.fc2 = nn.Dense(hidden_size // 2, 1, weight_init='xavier_uniform')
        self.act = nn.GELU()
        self.max_length = max_length

    def construct(self, input_ids, attention_mask=None):
        # 获取batch_size和序列长度
        batch_size, seq_length = input_ids.shape
        
        # 创建位置ID
        position_ids = ops.arange(0, seq_length, dtype=ms.int32)
        position_ids = ops.broadcast_to(position_ids, (batch_size, seq_length))
        
        # 词嵌入和位置嵌入
        embeddings = self.embedding(input_ids)
        position_embeddings = self.position_embedding(position_ids)
        embeddings = embeddings + position_embeddings
        
        # 应用层归一化和Dropout
        embeddings = self.layer_norm(embeddings)
        embeddings = self.dropout(embeddings)
        
        # 处理注意力掩码
        if attention_mask is None:
            # 默认掩码：把pad_token部分掩盖掉
            attention_mask = (input_ids != self.tokenizer.pad_token_id).astype(ms.float32)
        
        # 为Transformer准备输入
        embeddings = ops.clip_by_value(embeddings, -5, 5)
        embeddings = embeddings.transpose((1, 0, 2))  # [seq_len, batch_size, hidden_size]
        
        # 创建自注意力掩码
        padding_length = self.max_length - SEQ_LENGTH
        src_mask = ops.zeros((self.max_length, self.max_length), dtype=ms.bool_)
        src_mask[:, SEQ_LENGTH:] = ops.ones((self.max_length, padding_length), dtype=ms.bool_)
        
        # 应用Transformer
        transformer_out = self.transformer(embeddings, src_mask)
        
        # 全局平均池化
        pooled = ops.mean(transformer_out, axis=0)
        
        # 多层分类头
        pooled = self.fc1(pooled)
        pooled = self.act(pooled)
        pooled = self.dropout(pooled)
        logits = self.fc2(pooled)
        
        return logits

    def tokenize(self, texts):
        try:
            encodings = self.tokenizer(
                texts,
                max_length=self.max_length,
                padding='max_length',
                truncation=True,
                return_tensors='np'
            )
            input_ids = ms.Tensor.from_numpy(encodings['input_ids'])
            attention_mask = ms.Tensor.from_numpy(encodings['attention_mask'])
            return input_ids, attention_mask
        except Exception as e:
            raise ValueError(f"Tokenization failed: {str(e)}")

def get_model(model_name, input_dim=INPUT_DIM, vocab_size=VOCAB_SIZE):
    if model_name == "lr":
        return LogisticRegression(input_dim=input_dim)
    elif model_name == "ebm":
        ebm = ExplainableBoostingClassifier()
        return SklearnModelWrapper(ebm)
    elif model_name == "bigbird":
        return BigBirdModel(vocab_size=vocab_size)
    else:
        raise ValueError(f"Unknown model name: {model_name}")

if __name__ == "__main__":
    ms.set_context(mode=ms.GRAPH_MODE)
    lr_model = get_model("lr")
    bigbird_model = get_model("bigbird")
    sample_tfidf = ms.Tensor(np.random.rand(2, INPUT_DIM), dtype=ms.float32)
    sample_texts = ["Patient with fever and sepsis.", "Patient with jaundice."]
    sample_ids, sample_mask = bigbird_model.tokenize(sample_texts)
    print("LR output shape:", lr_model(sample_tfidf).shape)
    print("BigBird output shape:", bigbird_model(sample_ids, sample_mask).shape)