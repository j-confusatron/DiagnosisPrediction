#from nltk.lm import Vocabulary
import torch
import os
import csv
import json
import pickle
from collections import Counter
import numpy as np
from sys_config import SysConfig
from transformers import AutoTokenizer

from dual_model_pipeline.data.ner_medmentions_load import get_ner_tags

path = os.path.join('..', 'data', 'imr', 'clean')

class IMRDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_name) -> None:
        super().__init__()
        print(f"Constructing {dataset_name} dataset...")
        self.dataset_name = dataset_name
        tags2id, id2tags = get_ner_tags(root_only=True)

        # Read in the raw data from file.
        with open(os.path.join(path, dataset_name+'.json')) as json_data:
            raw_data = json.load(json_data)

        # Stack the word and tag matrices, then tokenize.
        words = [None for _ in range(len(raw_data))]
        tags = [None for _ in range(len(raw_data))]
        for i, raw in enumerate(raw_data):
            words[i] = raw['ner']['words']
            tags[i] = [tags2id[t] for t in ['O']+raw['ner']['tags']]
            #tags[i] = raw['ner']['tags']
        config = SysConfig()
        tokenizer = AutoTokenizer.from_pretrained(config.classification_model_name)
        word_toks = tokenizer(words, is_split_into_words=True, return_offsets_mapping=True, padding=True, truncation=True, max_length=512)

        # Match the tag ids to the new encoded word ids.
        tag_toks = np.zeros((len(word_toks['input_ids']), len(word_toks['input_ids'][0])), dtype=np.int32)
        for r_words_ids, r_tag_ids, r_tags in zip(word_toks['offset_mapping'], tag_toks, tags):
            i_tag = -1
            for i, w_id in enumerate(r_words_ids):
                if w_id[0] == 0:
                    i_tag += 1
                    if i_tag >= len(r_tags):
                        break
                r_tag_ids[i] = r_tags[i_tag]

        # Create the clean data for training.
        self.data = [None for _ in range(len(raw_data))]
        for i, raw in enumerate(raw_data):
            item = {
                'labels': raw['labels']['DiagnosisCategory'],
                'features': raw['features']['AgeRange']+raw['features']['PatientGender'],
                'word_input_ids': word_toks['input_ids'][i],
                'word_attention_mask': word_toks['attention_mask'][i],
                'tag_input_ids': tag_toks[i].tolist()
            }
            self.data[i] = item
    
    def __getitem__(self, idx):
        item = self.data[idx]
        t_item = {}
        for k,v in item.items():
            t_item[k] = torch.tensor(v)
        return t_item
    
    def __len__(self):
        return len(self.data)

class Vocabulary():

    def __init__(self, counter, unk_cutoff=1):
        self.PAD = '<PAD>'
        self.SEP = '<SEP>'
        self.UNK = '<UNK>'
        self.CLS = '<CLS>'
        self.idx2word = [self.PAD, self.SEP, self.UNK, self.CLS]
        self.word2index = {self.PAD: 0, self.SEP: 1, self.UNK: 2, self.CLS: 3}
        self.word2count = {}
        self.num_words = 4
        self.unk_cutoff = unk_cutoff

        for word, count in counter.items():
            self.idx2word.append(word)
            self.word2index[word] = self.num_words
            self.word2count[word] = count
            self.num_words += 1

    def add_word(self, word):
        if word in self.word2index:
            self.word2count[word] += 1
        else:
            self.idx2word.append(word)
            self.word2index[word] = self.num_words
            self.word2count[word] = 1
            self.num_words += 1

    def idx(self, word):
        if word in self.word2index and self.word2count[word] >= self.unk_cutoff:
            return self.word2index[word]
        else:
            return self.word2index[self.UNK]
    
    def word(self, idx):
        idx = idx if idx < self.num_words and idx >= 0 else self.word2index[self.UNK]
        word = self.idx2word[idx]
        if self.word2count[word] < self.unk_cutoff:
            word = self.UNK
        return word

class IMRDatasetCustomVocab(torch.utils.data.Dataset):
    def __init__(self, word_vocab, tag_vocab, fname, seq_len=384) -> None:
        super().__init__()
        with open(os.path.join(path, fname), 'r') as data_file:
            raw_data = json.load(data_file)
        
        self.size = len(raw_data)
        self.items = [None for _ in range(self.size)]
        for idx, sample in enumerate(raw_data):
            item = {
                'labels': sample['labels']['DiagnosisCategory'],
                'features': sample['features']['AgeRange']+sample['features']['PatientGender'],
                'word_entity_ids': np.zeros(seq_len, dtype=np.int64),
                'tag_entity_ids': np.zeros(seq_len, dtype=np.int64),
                'attention_mask': np.ones(seq_len, dtype=np.bool8)
            }
            for i in range(len(sample['ner']['words'])):
                item['word_entity_ids'][i] = word_vocab.idx(sample['ner']['words'][i])
                item['tag_entity_ids'][i] = tag_vocab.idx(sample['ner']['tags'][i])
                item['attention_mask'][i] = 0
            self.items[idx] = item
    
    def __getitem__(self, idx):
        t_item = {}
        item = self.items[idx]
        for k, v in item.items():
            t_item[k] = torch.tensor(v)
        return t_item

    def __len__(self):
        return self.size

def get_supplementary_data():
    word_vocab = get_vocab('vocab_counter.pickle')
    tag_vocab = get_vocab('tags_counter.pickle')
    mappings = get_mappings()
    return word_vocab, tag_vocab, mappings

def get_mappings():
    with open(os.path.join(path, 'mappings.json'), 'r') as json_mappings:
        mappings = json.load(json_mappings)
    return mappings

def get_vocab(fname, unk_cutoff=1):
    _counter = Counter()
    with open(os.path.join(path, fname), 'rb') as pkl:
        _counter = pickle.load(pkl)
    vocab = Vocabulary(_counter, unk_cutoff=unk_cutoff)
    return vocab

if __name__ == '__main__':
    word_vocab, tag_vocab, mappings = get_supplementary_data()
    train_ds = IMRDataset(word_vocab, tag_vocab, 'imr_train.json')
    pass