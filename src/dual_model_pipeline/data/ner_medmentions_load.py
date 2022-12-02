import os
import csv
import json
import torch
from transformers import AutoTokenizer
import numpy as np

path = os.path.join('..', 'data', 'ner', 'MedMentions', 'clean')

class NERMedMentionsDataset(torch.utils.data.Dataset):
    def __init__(self, tokenizer, dataset_name='train'):
        x, y = self.__get_raw_data(dataset_name)
        self.encodings = tokenizer(x, is_split_into_words=True, return_offsets_mapping=True, padding=True, truncation=True)
        self.labels = self.__encode_tags(y, self.encodings)
        self.encodings.pop('offset_mapping')

    def __encode_tags(self, tags, encodings):
        encoded_labels = []
        bad_offsets = []

        for i, doc_data in enumerate(zip(tags, encodings.offset_mapping)):
            doc_labels, doc_offset = doc_data
            # create an empty array of -100
            doc_enc_labels = np.ones(len(doc_offset),dtype=int) * -100
            arr_offset = np.array(doc_offset)

            # set labels whose first offset position is 0 and the second is not 0
            try:
                doc_enc_labels[(arr_offset[:,0] == 0) & (arr_offset[:,1] != 0)] = doc_labels
                encoded_labels.append(doc_enc_labels.tolist())
            except:
                bad_offsets.append(i)

        for i in reversed(bad_offsets):
            encodings['input_ids'].pop(i)
            encodings['token_type_ids'].pop(i)
            encodings['attention_mask'].pop(i)
            encodings['offset_mapping'].pop(i)
        assert len(encodings['input_ids']) == len(encoded_labels)

        return encoded_labels

    def __read_csv_data(self, f_name):
        data = []
        with open(f_name, 'r', encoding='utf-8') as data_file:
            reader = csv.reader(data_file)
            for row in reader:
                data.append(row)
        return data

    def __get_raw_data(self, dataset_name):
        # Load data.
        x = self.__read_csv_data(os.path.join(path, dataset_name+'_x.csv'))
        y = self.__read_csv_data(os.path.join(path, dataset_name+'_y.csv'))
        assert len(x) == len(y)
        for x_s, y_s in zip(x, y):
            assert len(x_s) == len(y_s)

        return x, y
    
    def encode_tags(tags, encodings):
        encoded_labels = []
        bad_offsets = []

        for i, doc_data in enumerate(zip(tags, encodings.offset_mapping)):
            doc_labels, doc_offset = doc_data
            # create an empty array of -100
            doc_enc_labels = np.ones(len(doc_offset),dtype=int) * -100
            arr_offset = np.array(doc_offset)

            # set labels whose first offset position is 0 and the second is not 0
            try:
                doc_enc_labels[(arr_offset[:,0] == 0) & (arr_offset[:,1] != 0)] = doc_labels
                encoded_labels.append(doc_enc_labels.tolist())
            except:
                bad_offsets.append(i)

        for i in reversed(bad_offsets):
            encodings['input_ids'].pop(i)
            encodings['token_type_ids'].pop(i)
            encodings['attention_mask'].pop(i)
            encodings['offset_mapping'].pop(i)
        assert len(encodings['input_ids']) == len(encoded_labels)

        return encoded_labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

def get_ner_tags(root_only=False):
    with open(os.path.join(path, 'tag2id.json'), 'r') as json_tags:
        tags2id = json.load(json_tags)
    if root_only:
        i = 0
        new_tags2id = {}
        for k,v in tags2id.items():
            if k.startswith('B-'):
                new_tags2id[k[2:]] = i
                i += 1
            elif k == 'O':
                new_tags2id[k] = i
                i += 1
        tags2id = new_tags2id
    id2tags = ['' for _ in tags2id.keys()]
    for k, v in tags2id.items():
        id2tags[v] = k
    return tags2id, id2tags

if __name__ == '__main__':
    tokenizer = AutoTokenizer.from_pretrained('microsoft/BiomedNLP-KRISSBERT-PubMed-UMLS-EL')
    data = NERMedMentionsDataset(tokenizer)
    pass