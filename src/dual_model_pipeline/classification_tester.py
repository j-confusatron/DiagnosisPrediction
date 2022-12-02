import torch
import os
import json
import csv
from tqdm import tqdm
import numpy as np
from scipy.special import softmax
import matplotlib.pyplot as plt

from dual_model_pipeline.models.classification_model import MedicalClassifier
from dual_model_pipeline.data.classification_imr_load import IMRDataset, get_mappings
from dual_model_pipeline.classification_trainer import get_batch_data
from metrics.viz_metrics import viz_confusion_matrix
from metrics.standard_metrics import conf_matrix_metrics
from sys_config import SysConfig

def test_model(mask_x=False):
    # Setup the config and test variables.
    config = SysConfig()
    model_path = os.path.join(config.classification_model_path, config.classification_model_checkpoint)
    with open(os.path.join(model_path, 'hyperparams.json'), 'r') as json_hyperparams:
        hyperparams = json.load(json_hyperparams)
    imr_mappings = get_mappings()
    n_labels = imr_mappings['DiagnosisCategory']['DiagnosisCategory_len']

    # Load the model.
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MedicalClassifier(config.classification_model_size).to(device)
    model.load_state_dict(torch.load(os.path.join(model_path, 'model.pt'), map_location=device))
    model.eval()

    # Get the dataset.
    test_ds = IMRDataset('imr_test')
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=hyperparams['batch_size'])

    # Metrics trackers.
    conf_matrix = np.zeros((n_labels,n_labels))
    acc_sum = 0.0
    n_acc = 0

    # Make predictions against the test dataset.
    print("Making predictions...")
    for i_batch, batch in (pbar := tqdm(enumerate(test_loader), total=len(test_loader))):
        # Make the prediction.
        y, y_gt, x, w_x_ids, w_x_attn, t_x_ids = get_batch_data(batch, device)
        y_est, attn = model(w_x_ids, w_x_attn, t_x_ids, x)
        y_hat = torch.argmax(y_est, dim=1).detach().cpu().numpy()

        # If masking the patient context, zero out the tensor.
        if mask_x:
            x = torch.zeros_like(x).to(device)

        # Record metrics.
        acc_sum += np.count_nonzero(y_gt == y_hat)
        n_acc += len(y_hat)
        for gt, hat in zip(y_gt, y_hat):
            conf_matrix[gt, hat] += 1

    # Write results.
    print("Writing results...")
    test_dir = os.path.join(model_path, 'test')
    os.makedirs(test_dir, exist_ok=True)
        
    # Confusion matrix.
    labels = imr_mappings['DiagnosisCategory']['idx2DiagnosisCategory']
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
    test_model(mask_x=True)