import mindspore
from mindspore import nn, ops, Tensor
from mindspore.dataset import GeneratorDataset
import numpy as np
import random
import string

mindspore.set_context(device_target = "Ascend" if mindspore.hal.is_available("Ascend") else "cpu")

class TextDataset():
    def __init__(self, text, seq_len):
        self.text = text
        self.seq_len = seq_len

    def __len__(self):
        return len(self.text) - self.seq_len

    def __getitem__(self, index):
        return np.asarray(self.text[index : index + self.seq_len]), np.asarray(self.text[index + self.seq_len], np.int32)

class Hyena(nn.Cell):
    def __init__(self, input_dim, output_dim, filter_size, depth, positional_dim):
        super(Hyena, self).__init__()
        self.depth = depth
        self.output_dim = output_dim
        self.input_dim = input_dim
        self.positional_dim = positional_dim
        self.filter_size = filter_size

        self.linear1 = nn.Dense(input_dim, (output_dim + 1) * depth)
        self.conv1d = nn.Conv1d(depth, depth, filter_size, pad_mode = "pad", padding = filter_size // 2, has_bias = True)
        self.linear2 = nn.Dense(depth * positional_dim, output_dim)

    def construct(self, x):
        x = x.astype(mindspore.float32)  # Convert input tensor to float
        x = self.linear1(x)
        x = x.view((x.shape[0], -1, self.depth)).swapaxes(1, 2)
        x = self.conv1d(x)
        x = x.swapaxes(1, 2).contiguous().view((-1, self.depth * self.positional_dim))
        x = self.linear2(x)
        return x

def train_hyena_model(text_file, input_dim, filter_size, depth, positional_dim, lr, num_epochs, batch_size = 128):
    text = open(text_file, 'r').read()
    text = text.lower()
    chars = list(set(text))
    char_to_idx = {ch: i for i, ch in enumerate(chars)}

    loader = TextDataset([char_to_idx[ch] for ch in text], input_dim)
    dataset = GeneratorDataset(source = loader, column_names = ["seqs", "targets"])
    dataset = dataset.batch(batch_size)
   # Move model to GPU if available
    model = Hyena(input_dim, len(chars), filter_size, depth, positional_dim)
    criterion = nn.CrossEntropyLoss()
    optimizer = nn.Adam(model.trainable_params(), learning_rate = lr)

    def forward_fn(seqs, targets):
        output = model(seqs)
        loss = criterion(output, targets)
        return loss, output
    
    grad_fn = mindspore.value_and_grad(forward_fn, None, optimizer.parameters, has_aux = True)
    for epoch in range(1, num_epochs + 1):
        for i, (seqs, targets) in enumerate(dataset.create_tuple_iterator()):
            (loss, _), grads = grad_fn(seqs, targets)
            optimizer(grads)

        if epoch % 100 == 0:
            print(f"Epoch [{epoch}/{num_epochs}], Loss: {loss.asnumpy():.4f}")
    print('Training completed.')
    return model, chars, char_to_idx

def generate_text(model, seed_text, length, char_to_idx, idx_to_char, vocab, input_dim):
    model.set_train(False)
    seed_indices = Tensor(
        [char_to_idx.get(c, random.randint(0, vocab - 1)) for c in seed_text.lower()],
        mindspore.int64
        )
    if len(seed_indices) < input_dim:
        seed_indices = ops.concat((seed_indices, ops.zeros(input_dim - len(seed_indices), dtype = mindspore.int64)))
    out = []
    for i in range(length):
        seed_input = seed_indices.astype(mindspore.float32).unsqueeze(0)
        outputs = model(seed_input)
        probs = ops.softmax(outputs[-1], axis = 0)
        probs = probs.asnumpy()
        next_idx = np.random.choice(len(probs), p = probs)
        out.append(idx_to_char[next_idx])
        seed_indices[:-1] = seed_indices[1:].copy()
        seed_indices[-1] = next_idx
    return out

def main():
    random_text = ''.join(
        random.choice(string.ascii_lowercase + string.digits + string.punctuation + ' ')
        for _ in range(1337)
    )
    with open('random_text.txt', 'w') as f:
        f.write(random_text)

    input_dim = 70
    output_dim = 64
    filter_size = 3
    depth = 3
    positional_dim = (input_dim - filter_size + 2 * (filter_size // 2)) // 1 + 1
    lr = 0.001
    num_epochs = 1000
    model, vocab, char_to_idx = train_hyena_model(
        'random_text.txt', input_dim, filter_size, depth, positional_dim, lr, num_epochs
    )
    print(model)
    idx_to_char = {idx: char for char, idx in char_to_idx.items()}
    seed_text = 'The quick brown fox'
    num_chars = 70
    generated_text = generate_text(model, seed_text, num_chars, char_to_idx, idx_to_char, len(vocab), input_dim)
    
    print('Generated text: ' + ''.join(generated_text))

if __name__ == '__main__':
    main()