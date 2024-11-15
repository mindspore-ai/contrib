import os
import json
import ndjson
from mindspore.dataset import GeneratorDataset

class ShowerthoughtDataset:
    def __init__(self, showerthoughts_dataset_path='../../data'):
        self.showerthought_list = []
        self.begin_of_text_token = "<|showerthought|>"
        self.end_of_text_token = "<|endoftext|>"

        short_showerthoughts_path = os.path.join(showerthoughts_dataset_path, 'roberta_train_data_GPT2.ndjson')

        with open(short_showerthoughts_path) as f:
            reader = ndjson.reader(f)
            try:
                for post in reader:
                    if self.__isPostValid(post):
                        showerthought_str = f"{self.begin_of_text_token}{post['title']}{self.end_of_text_token}"
                        self.showerthought_list.append(showerthought_str)
            except json.JSONDecodeError:
                pass

    def __isPostValid(self, post):
        if 'removed_by_category' in post:
            return False
        if "post_hint" in post and post["post_hint"] == "image":
            return False
        return True

    def __iter__(self):
        for showerthought in self.showerthought_list:
            yield showerthought

    def __len__(self):
        return len(self.showerthought_list)

def create_showerthoughts_dataset(dataset_path, shuffle=True):
    showerthoughts_dataset = ShowerthoughtDataset(dataset_path)
    def generator():
        for showerthought in showerthoughts_dataset:
            yield (showerthought,)
    dataset = GeneratorDataset(generator, ["showerthought"])
    if shuffle:
        dataset = dataset.shuffle(buffer_size=len(showerthoughts_dataset))
    return dataset

if __name__ == '__main__':
    current_script_path = os.path.abspath(__file__)
    current_script_dir = os.path.dirname(current_script_path)
    dataset_path = os.path.join(current_script_dir, 'data') 
    print(dataset_path)

    showerthoughts_dataset = create_showerthoughts_dataset(dataset_path, shuffle=True)

    for batch in showerthoughts_dataset.create_dict_iterator(): 
        print(batch)