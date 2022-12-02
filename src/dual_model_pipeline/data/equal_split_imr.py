import os
import json
import numpy as np
from tqdm import tqdm

from dual_model_pipeline.data.classification_imr_load import get_mappings

path = os.path.join('..', 'data', 'imr', 'clean')
new_path = os.path.join(path, 'new')
files = ['imr_train.json', 'imr_val.json', 'imr_test.json']

def equal_split():
    print("Beginning dataset split")
    n_labels = get_mappings()['DiagnosisCategory']['DiagnosisCategory_len']
    merged_samples = merge_into_one()
    samples_by_label = sort_by_label(n_labels, merged_samples)
    train, val, test = split_labels_into_datasets(samples_by_label)
    write_datasets(train, val, test)
    print("Dataset split complete!")

def write_datasets(train, val, test):
    print("Writing datasets to file...")
    for ds, fname in zip([train, val, test], files):
        with open(os.path.join(new_path, fname), 'w') as json_file:
            json.dump(ds, json_file)

def split_labels_into_datasets(samples_by_label):
    print("Splitting labels into datasets...")
    train = []
    val = []
    test = []
    for label_samples in tqdm(samples_by_label):
        datasets = split_samples(label_samples)
        train += datasets[0]
        val += datasets[1]
        test += datasets[2]

    print("Shuffling datasets...")
    np.random.shuffle(train)
    np.random.shuffle(val)
    np.random.shuffle(test)

    return train, val, test

def split_samples(samples, splits=[.8, .1]):
    np.random.shuffle(samples)
    n_train = int(len(samples) * splits[0])
    n_val = int(len(samples) * splits[1]) + n_train
    datasets = samples[:n_train], samples[n_train:n_val], samples[n_val:]
    return datasets

def sort_by_label(n_labels, samples):
    print("Bucketing samples by label... ")
    samples_by_label = [[] for _ in range(n_labels)]
    for s in samples:
        samples_by_label[s['labels']['DiagnosisCategory']].append(s)
    for i, bucket in enumerate(samples_by_label):
        print(f"{i}: {len(bucket)}")
    return samples_by_label

def merge_into_one():
    print("Merging samples... ", end='')
    merged_samples = []
    for fname in files:
        with open(os.path.join(path, fname), 'r') as json_file:
            merged_samples += json.load(json_file)
    print(f"{len(merged_samples)} total samples")
    return merged_samples

if __name__ == '__main__':
    equal_split()