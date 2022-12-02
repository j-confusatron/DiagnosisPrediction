import argparse
import os
import csv
import json

# Orchestrate pulling in the raw MedMentions data, sorting it into proper samples, then writing it to disk.
def create_dataset(fname_data):
    raw_data = get_raw_samples(fname_data)
    m_labels = map_labels(raw_data)
    samples = create_samples(raw_data, m_labels)
    data_sets = read_ids()
    write_file('train_', samples, data_sets[0])
    write_file('val_', samples, data_sets[1])
    write_file('test_', samples, data_sets[2])

def write_file(fname, samples, ids):
    dir = os.path.join('data', 'ner', 'MedMentions', 'clean')
    os.makedirs(dir, exist_ok=True)
    
    x = []
    y = []
    for id in ids:
        rec_set = samples[id]
        for rec in rec_set:
            x.append([r['text'] for r in rec])
            y.append([r['t'] for r in rec])
            assert len(x[-1]) == len(y[-1])

    with open(os.path.join(dir, f'{fname}x.csv'), 'w', encoding='utf-8', newline='') as x_file:
        csv_writer = csv.writer(x_file)
        csv_writer.writerows(x)
    with open(os.path.join(dir, f'{fname}y.csv'), 'w', encoding='utf-8', newline='') as x_file:
        csv_writer = csv.writer(x_file)
        csv_writer.writerows(y)

# Read the training, validation, and test ids from their respective files.
def read_ids(id_files=['corpus_pubtator_pmids_trng.txt', 'corpus_pubtator_pmids_dev.txt', 'corpus_pubtator_pmids_test.txt']):
    data_sets = [None, None, None]
    for i, idf in enumerate(id_files):
        with open(os.path.join('data', 'ner', 'MedMentions', idf), 'r', encoding='utf-8') as f_id:
            data_sets[i] = f_id.read().split('\n')[:-1]
    return data_sets

# Convert the raw samples into organized data to use for inference.
def create_samples(raw_data, m_labels):
    samples = {}
    eos = ['.', '?', '!']

    # Iterate over all samples.
    for id, sample in raw_data.items():
        samples[id] = []
        text_to_label = []
        text = sample['title'] + '.' + sample['abstract']

        # Tokenize the current sample, according to the defined NER tags.
        i = 0
        for t in sample['tags']:
            # Handle any non-tagged text prior to the current tag.
            if t['s'] > i:
                for txt in text[i:t['s']].split():
                    if txt[-1] in eos and len(txt[:-1]) > 1:
                        text_to_label.append({'text': txt[:-1], 't': m_labels['O']})
                        text_to_label.append({'text': txt[-1], 't': m_labels['O']})
                    else:
                        text_to_label.append({'text': txt, 't': m_labels['O']})
                i = t['s']

            # Record the current tag.
            pfx = 'B-'
            #pfx = ''
            for txt in text[i:t['e']].split():
                text_to_label.append({'text': txt, 't': m_labels[f"{pfx}{t['t']}"]})
                pfx = 'I-'
                #pfx = ''
            i = t['e']
        
        # Record any text that falls after the final tag.
        if i < len(text):
            for txt in text[i:].split():
                if txt[-1] in eos and len(txt[:-1]) > 1:
                    text_to_label.append({'text': txt[:-1], 't': m_labels['O']})
                    text_to_label.append({'text': txt[-1], 't': m_labels['O']})
                else:
                    text_to_label.append({'text': txt, 't': m_labels['O']})

        # Group the data into sentences, according to ids.
        s = 0
        e = 1
        for t2l in text_to_label:
            if t2l['text'] in eos:
                samples[id].append(text_to_label[s:e])
                s = e
            e += 1

    return samples

# Map all unique labels to integers for classification.
def map_labels(raw_data):
    s_labels = set()
    for sample in raw_data.values():
        for tag in sample['tags']:
            s_labels.add(f"B-{tag['t']}")
            s_labels.add(f"I-{tag['t']}")
            #s_labels.add(f"{tag['t']}")

    m_labels = {'O': 0}
    for i, label in enumerate(sorted(s_labels)):
        m_labels[label] = i+1

    with open(os.path.join('data', 'ner', 'MedMentions', 'clean', 'tag2id.json'), 'w') as json_labels:
        json.dump(m_labels, json_labels)

    return m_labels

# Read the raw MedMentions data from file.
def get_raw_samples(fname_data):
    raw_data = ''
    with open(fname_data, 'r', encoding='utf-8') as data_file:
        raw_data = data_file.read()
    raw_data = raw_data.split('\n\n')

    data = {}
    for i, sample in enumerate(raw_data[:-1]):
        sample = sample.split('\n')
        id = sample[0][:sample[0].find('|')]
        title = sample[0][sample[0].find('|t|')+3:]
        abstract = sample[1][sample[1].find('|a|')+3:]

        raw_tags = sample[2:]
        tags = [None for _ in raw_tags]
        for i, t in enumerate(raw_tags):
            t = t.split('\t')
            tags[i] = {'s': int(t[1]), 'e': int(t[2]), 'w': t[3], 't': t[4]}

        data[id] = {'title': title, 'abstract': abstract, 'tags': tags}

    return data

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Build a dataset for MedMentions NER')
    parser.add_argument('-f',
                        dest='fname_data', 
                        default='corpus_pubtator_st21pv.txt', 
                        help='File name to load')
    args = parser.parse_args()

    fname_data = os.path.join('data', 'ner', 'MedMentions', args.fname_data)
    create_dataset(fname_data)