import argparse
import os
import csv
import json
import numpy as np
from collections import Counter
import pickle
from tqdm import tqdm

from dual_model_pipeline.models.ner_model import NER_Model
from util import one_hot

class IMR_Fields():
    ReferenceID = 'ReferenceID'
    ReportYear = 'ReportYear'
    DiagnosisCategory = 'DiagnosisCategory'
    DiagnosisSubCategory = 'DiagnosisSubCategory'
    TreatmentCategory = 'TreatmentCategory'
    TreatmentSubCategory = 'TreatmentSubCategory'
    Determination = 'Determination'
    Type = 'Type'
    AgeRange = 'AgeRange'
    PatientGender = 'PatientGender'
    IMRType = 'IMRType'
    DaysToReview = 'DaysToReview'
    DaysToAdopt = 'DaysToAdopt'
    Findings = 'Findings'

IMR_Field2Idx = {
    IMR_Fields.ReferenceID: 0,
    IMR_Fields.ReportYear: 1,
    IMR_Fields.DiagnosisCategory: 2,
    IMR_Fields.DiagnosisSubCategory: 3,
    IMR_Fields.TreatmentCategory: 4,
    IMR_Fields.TreatmentSubCategory: 5,
    IMR_Fields.Determination: 6,
    IMR_Fields.Type: 7,
    IMR_Fields.AgeRange: 8,
    IMR_Fields.PatientGender: 9,
    IMR_Fields.IMRType: 10,
    IMR_Fields.DaysToReview: 11,
    IMR_Fields.DaysToAdopt: 12,
    IMR_Fields.Findings: 13
}

# Orchestrate pulling in the raw IMR data, sorting it into proper samples, then writing it to disk.
def create_dataset(path,  fname_data):
    clean_path = os.path.join(path, 'clean')
    os.makedirs(clean_path, exist_ok=True)
    raw_data = read_csv(os.path.join(path, fname_data))
    #raw_data = raw_data[:10]
    mappings = create_category_mappings(raw_data, clean_path)
    samples = format_samples(raw_data, mappings)
    write_vocab(samples, clean_path)
    split_and_write_samples(samples, clean_path)

def split_and_write_samples(samples, path, splits=[.8, .1]):
    print("Writing train, validate, and test datasets...")
    np.random.shuffle(samples)
    n_train = int(len(samples) * splits[0])
    n_val = int(len(samples) * splits[1]) + n_train
    datasets = samples[:n_train], samples[n_train:n_val], samples[n_val:]

    for ds in tqdm(zip(['imr_train.json', 'imr_val.json', 'imr_test.json'], datasets)):
        with open(os.path.join(path, ds[0]), 'w') as json_ds:
            json.dump(ds[1], json_ds)

def write_vocab(samples, path):
    print("Writing vocab files...")
    words = Counter()
    tags = Counter()
    for s in samples:
        words.update(s['ner']['words'])
        tags.update(s['ner']['tags'])
    for pkl_file, data in zip(['vocab_counter.pickle', 'tags_counter.pickle'], [words, tags]):
        with open(os.path.join(path, pkl_file), 'wb') as pkl:
            pickle.dump(data, pkl)

def format_samples(raw_data, mappings, labels=[IMR_Fields.DiagnosisCategory], features=[IMR_Fields.AgeRange, IMR_Fields.PatientGender]):
    print("Formatting data samples...")
    ner_model = NER_Model()
    samples = [{'labels': {fld: None for fld in labels}, 'features': {fld: None for fld in features}, 'ner': {'words': [], 'tags': []}} for _ in range(len(raw_data))]
    
    for i, row in tqdm(enumerate(raw_data)):
        for fld in labels:
            val = row[IMR_Field2Idx[fld]]
            if fld in mappings:
                val = mappings[fld][f'{fld}2idx'][val]
            samples[i]['labels'][fld] = val

        for fld in features:
            val = row[IMR_Field2Idx[fld]]
            if fld in mappings:
                val = one_hot(mappings[fld][f'{fld}2idx'][val], mappings[fld][f'{fld}_len'])
            samples[i]['features'][fld] = val
        
        words, tags, conf = ner_model.inference(row[IMR_Field2Idx[IMR_Fields.Findings]])
        for word, tag in zip(words, tags):
            samples[i]['ner']['words'].append(word)
            samples[i]['ner']['tags'].append(tag)

    return samples


def create_category_mappings(raw_data, path, fields=[IMR_Fields.DiagnosisCategory, IMR_Fields.AgeRange, IMR_Fields.PatientGender]):
    print("Creating category mappings...")
    map_sets = {}
    mappings = {}
    for fld in fields:
        map_sets[fld] = set()
        mappings[fld] = {}
        mappings[fld][f'{fld}2idx'] = {}
        mappings[fld][f'idx2{fld}'] = []
        mappings[fld][f'{fld}_len'] = 0

    for row in raw_data:
        for fld in fields:
            map_sets[fld].add(row[IMR_Field2Idx[fld]])
    for fld, val_set in map_sets.items():
        mappings[fld][f'{fld}_len'] = len(val_set)
        for i, val in enumerate(val_set):
            mappings[fld][f'{fld}2idx'][val] = i
            mappings[fld][f'idx2{fld}'].append(val)

    with open(os.path.join(path, 'mappings.json'), 'w') as json_map_file:
        json.dump(mappings, json_map_file)

    return mappings

def read_csv(fname_data):
    print("Reading raw CSV data...")
    raw_data = []
    with open(fname_data, 'r') as f_raw_data:
        reader = csv.reader(f_raw_data)
        for row in reader:
            raw_data.append(row)
    return raw_data[1:]

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Build a dataset for IMR classification')
    parser.add_argument('-f',
                        dest='fname_data', 
                        default='independent-medical-review-imr-determinations-trend.csv', 
                        help='File name to load')
    args = parser.parse_args()

    path = os.path.join('data', 'imr')
    create_dataset(path, args.fname_data)