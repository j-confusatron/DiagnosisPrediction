import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer
from scipy.special import softmax
import numpy as np
from tqdm import tqdm
import os

from dual_model_pipeline.data.ner_medmentions_load import NERMedMentionsDataset, get_ner_tags
from metrics.viz_metrics import viz_confusion_matrix
from metrics.standard_metrics import conf_matrix_metrics
from sys_config import SysConfig

def test_model():
    # Load the model.
    config = SysConfig()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = AutoModelForTokenClassification.from_pretrained(config.ner_checkpoint).to(device)
    model.eval()

    # Load the dataset.
    test_ds = NERMedMentionsDataset(AutoTokenizer.from_pretrained(config.ner_model_name), 'test')
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=16)

    # Metrics trackers.
    ner2id, id2ner = get_ner_tags()
    n_labels = len(id2ner)
    conf_matrix = np.zeros((n_labels,n_labels))
    acc_sum = 0.0
    n_acc = 0

    # Make predictions against the test dataset.
    print("Making predictions...")
    for i_batch, batch in (pbar := tqdm(enumerate(test_loader), total=len(test_loader))):
        # Get all of the batch data.
        y = batch['labels']
        y_gt = y.detach().view(-1).numpy()
        y = y.to(device)
        input_ids = batch['input_ids'].to(device)
        token_type_ids = batch['token_type_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)

        # Make model predictions.
        out = model(input_ids, attention_mask, token_type_ids)
        y_hat = torch.argmax(out.logits, dim=2).detach().cpu().view(-1).numpy()
        filter = y_gt != -100
        y_gt = y_gt[filter]
        y_hat = y_hat[filter]

        # Record metrics.
        acc_sum += np.count_nonzero(y_gt == y_hat)
        n_acc += len(y_hat)
        for gt, hat in zip(y_gt, y_hat):
            conf_matrix[gt, hat] += 1

    # Write results.
    print("Writing results...")
    test_dir = os.path.join(config.ner_checkpoint, 'test')
    os.makedirs(test_dir, exist_ok=True)
        
    # Confusion matrix.
    labels = id2ner
    sf_conf_matrix = np.around(softmax(conf_matrix, axis=1), 2)
    viz_confusion_matrix(sf_conf_matrix, labels, os.path.join(test_dir, 'confusion_matrix.png'))

    # General results.
    ignore_indices=[26]
    for i in reversed(sorted(ignore_indices)):
        labels.pop(i)
    ttl_p, ttl_r, ttl_f1, macro_p, macro_r, macro_f1, lbl_precision, lbl_recall, lbl_f1 = conf_matrix_metrics(conf_matrix, ignore_indices)
    with open(os.path.join(test_dir, 'results.txt'), 'w') as f_results:
        f_results.write(f"Accuracy: {acc_sum/n_acc}\n")
        f_results.write(f"Total Precision: {ttl_p}\n")
        f_results.write(f"Total Recall: {ttl_r}\n")
        f_results.write(f"Total F1: {ttl_f1}\n")
        f_results.write(f"Macro Precision: {macro_p}\n")
        f_results.write(f"Macro Recall: {macro_r}\n")
        f_results.write(f"Macro F1: {macro_f1}\n\n")
        f_results.write(f"Label,Precision,Recall,F1\n")
        for i, lbl in enumerate(labels):
            f_results.write(f"{lbl},{lbl_precision[i]},{lbl_recall[i]},{lbl_f1[i]}\n")

    print("Test complete!")

if __name__ == '__main__':
    test_model()