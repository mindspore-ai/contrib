import mindspore
import mindspore.dataset as ds
import mindspore.dataset.transforms as C
import mindspore.dataset.transforms.c_transforms as C2
import mindspore.dataset.vision.c_transforms as CV
import mindspore.common.dtype as mstype
from mindspore import Tensor

def load_data(path, tokenizer, device, batch_size=32, shuffle=False):
    list_tokens = []
    list_masks = []

    with open(path, "r", encoding="utf-8") as file:
        for line in file:
            sentence = line.replace("\n", "")
            tokens = tokenizer.encode(sentence,
                                      add_special_tokens=True,
                                      do_lower_case=False)

            list_tokens.append(Tensor(tokens, dtype=mstype.int32))
            list_masks.append(Tensor([1] * len(tokens), dtype=mstype.int32))

    # pad the tokens and the masks
    pad_value = tokenizer.pad_token_id
    X = mindspore.ops.Pad(((0, 0), (0, max(len(t) for t in list_tokens) - len(list_tokens[0]))))(Tensor(list_tokens, dtype=mstype.int32))
    masks = mindspore.ops.Pad(((0, 0), (0, max(len(m) for m in list_masks) - len(list_masks[0]))))(Tensor(list_masks, dtype=mstype.int32))

    # create the loader
    dataset = ds.NumpySlicesDataset({"tokens": X, "masks": masks}, shuffle=shuffle)
    dataset = dataset.batch(batch_size, drop_remainder=True)

    return dataset