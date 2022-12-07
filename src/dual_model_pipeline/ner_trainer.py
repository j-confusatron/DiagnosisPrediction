from transformers import AutoModelForTokenClassification, AutoTokenizer, Trainer, TrainingArguments, TrainerCallback
import evaluate
import argparse
import numpy as np
import itertools
import os
import shutil
from copy import deepcopy

from dual_model_pipeline.data.ner_medmentions_load import NERMedMentionsDataset, get_ner_tags
from sys_config import SysConfig


def product_dict(**kwargs):
    keys = kwargs.keys()
    vals = kwargs.values()
    for instance in itertools.product(*vals):
        yield dict(zip(keys, instance))

def hyperparam_path(hyperparams):
    key = f"ep-{hyperparams['n_epochs']}_" \
        + f"lr-{hyperparams['lr']}_" \
        + f"wd-{hyperparams['wdecay']}_" \
        + f"bs-{hyperparams['batch_size']}_" \
        + f"freq-{hyperparams['metric_freq']}"
    return key

def get_hyperparameters():
    return {
        'n_epochs': [3],
        'lr': [5e-05],
        'wdecay': [0.01],
        'batch_size': [16],
        'metric_freq': [100]
    }

class CustomCallback(TrainerCallback):
    
    def __init__(self, trainer) -> None:
        super().__init__()
        self._trainer = trainer
    
    def on_step_end(self, args, state, control, **kwargs):
        if control.should_evaluate:
            control_copy = deepcopy(control)
            self._trainer.evaluate(eval_dataset=self._trainer.train_dataset, metric_key_prefix="train")
            return control_copy

metric_acc = evaluate.load("accuracy")
def compute_metrics(eval_pred):
    # Get the labels and predictions flattened out to a 1d array.
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1).flatten()
    labels = labels.flatten()

    # Ignore all -100s.
    valid_indices = labels[:] > -100
    predictions = predictions[valid_indices]
    labels = labels[valid_indices]

    # Return accuracy.
    metrics = metric_acc.compute(predictions=predictions, references=labels)
    metrics['conf_bound'] = 1.96 * np.sqrt((metrics['accuracy'] * (1 - metrics['accuracy'])) / len(predictions))
    return metrics

def train_model(name, checkpoint):
    # Load data & model.
    config = SysConfig()
    tags2id, id2tags = get_ner_tags()
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    model = AutoModelForTokenClassification.from_pretrained(checkpoint, num_labels=len(tags2id.keys()))
    ds_train = NERMedMentionsDataset(tokenizer, 'train')
    ds_validate = NERMedMentionsDataset(tokenizer, 'val')
    ds_test = NERMedMentionsDataset(tokenizer, 'test')
    hyperparameters = list(product_dict(**get_hyperparameters()))

    for i, h in enumerate(hyperparameters):
        m_dir = os.path.join(config.ner_model_path, hyperparam_path(h))
        shutil.rmtree(m_dir, ignore_errors=True)
        os.makedirs(m_dir, exist_ok=True)
        print("\nTraining:")
        for k, v in h.items():
            print(f"{k}: {v}")
        print("")

        training_args = TrainingArguments(
            output_dir=m_dir,
            num_train_epochs=h['n_epochs'],
            per_device_train_batch_size=h['batch_size'],
            per_device_eval_batch_size=h['batch_size'],
            warmup_steps=500,
            weight_decay=h['wdecay'],
            logging_dir='./logs',
            logging_steps=h['metric_freq'],
            evaluation_strategy="steps",
            eval_steps=h['metric_freq'],
            save_steps=600,
            save_total_limit=1
        )

        trainer = Trainer(
            model=model,                         # the instantiated 🤗 Transformers model to be trained
            args=training_args,                  # training arguments, defined above
            train_dataset=ds_train,              # training dataset
            eval_dataset=ds_validate,            # evaluation dataset
            compute_metrics=compute_metrics
        )
        trainer.add_callback(CustomCallback(trainer)) 
        trainer.train()

if __name__ == '__main__':
    name2checkpoint = {
        'kriss': 'microsoft/BiomedNLP-KRISSBERT-PubMed-UMLS-EL',
        'pubmed': 'microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract',
        'scibert-uncased': 'allenai/scibert_scivocab_uncased',
        'scibert-cased': 'allenai/scibert_scivocab_uncased'
    }
    parser = argparse.ArgumentParser(description='Train a language model for NER')
    parser.add_argument('-n', 
                        dest='name',
                        default='kriss',
                        action='store', 
                        help='Name of the model to load: kriss, pubmed')
    args = parser.parse_args()
    train_model(args.name, name2checkpoint[args.name])