import torch
import itertools
import shutil
import os
from tqdm import tqdm
import numpy as np
import json

from dual_model_pipeline.data.classification_imr_load import get_mappings, IMRDataset
from dual_model_pipeline.models.classification_model import MedicalClassifier
from sys_config import SysConfig
from metrics.viz_metrics import viz_graph


def hyperparam_path(hyperparams):
    key = f"ep-{hyperparams['n_epochs']}_" \
        + f"lr-{hyperparams['lr']}_" \
        + f"bs-{hyperparams['batch_size']}_" \
        + f"chk-{hyperparams['chk_ratio']}_" \
        + f"cls-{hyperparams['cls_name']}_" \
        + f"sched-{hyperparams['scheduler']}"
    return key

def product_dict(**kwargs):
    keys = kwargs.keys()
    vals = kwargs.values()
    for instance in itertools.product(*vals):
        yield dict(zip(keys, instance))

def get_hyperparameters():
    return {
        'n_epochs': [5],
        'lr': [1e-05],
        'batch_size': [8],
        'chk_ratio': [0.5],
        'cls_name': ['small_head'],
        'scheduler': [True]
    }

def get_batch_data(batch, device):
    y = batch['labels']
    y_gt = y.detach().numpy()
    y = y.to(device)
    x = batch['features'].to(device)
    w_x_ids = batch['word_input_ids'].to(device)
    w_x_attn = batch['word_attention_mask'].to(device)
    t_x_ids = batch['tag_input_ids'].to(device)
    return y, y_gt, x, w_x_ids, w_x_attn, t_x_ids

def confidence_bounds(accuracy, n_predictions):
    return 1.96 * np.sqrt((accuracy * (1 - accuracy)) / n_predictions)

def eval(validate_loader, device, model, loss):
    # Metrics trackers.
    loss_sum = 0.0
    acc_sum = 0.0
    n_acc = 0

    # Set the model to eval mode and run through the validation batch.
    model.eval()
    with torch.no_grad():
        for i_batch, batch in (pbar := tqdm(enumerate(validate_loader), total=len(validate_loader), position=1, leave=False)):
            pbar.set_description('   VALIDATION')

            # Make predictions and update metrics.
            y, y_gt, x, w_x_ids, w_x_attn, t_x_ids = get_batch_data(batch, device)
            y_est, attn = model(w_x_ids, w_x_attn, t_x_ids, x)
            l = loss(y_est, y)
            y_hat = torch.argmax(y_est, dim=1).detach().cpu().numpy()
            loss_sum += l.cpu().detach().item()
            acc_sum += np.count_nonzero(y_gt == y_hat)
            n_acc += len(y_hat)

    # Set and return final metrics.
    acc = acc_sum/n_acc
    return (loss_sum/i_batch), acc, confidence_bounds(acc, n_acc)

def train_and_eval(train_ds, val_ds, hyperparams, device, m_dir):
    # Create data loaders.
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=hyperparams['batch_size'], shuffle=True)
    validate_loader = torch.utils.data.DataLoader(val_ds, batch_size=hyperparams['batch_size'])
    i_checkpoint = int(len(validate_loader) * hyperparams['chk_ratio'])
    print(f"Checkpoint every {i_checkpoint} iterations")

    # Create the model, optimizer, and loss.
    model = MedicalClassifier(hyperparams['cls_name']).to(device)
    best_model = model.state_dict()
    optim = torch.optim.AdamW(model.parameters(), lr=hyperparams['lr'])
    loss = torch.nn.CrossEntropyLoss()
    if hyperparams['scheduler']:
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optim, max_lr=hyperparams['lr'], steps_per_epoch=len(train_loader), epochs=hyperparams['n_epochs'])

    # Iterate over epochs.
    i = 0
    tr_loss_hist = np.array([999.0, 0.0])
    vl_loss_hist = np.array([999.0])
    tr_acc_hist = np.array([0.0, 0.0])
    vl_acc_hist = np.array([0.0])
    tr_conf_bounds = np.array([])
    vl_conf_bounds = np.array([])
    n_acc = 0.0
    vl_acc = 0.0
    best_vl_acc = 0.0
    print("Epoch, Checkpoint %, Train Loss, Train Acc, Val Loss, Val Acc")
    for i_epoch in range(hyperparams['n_epochs']):

        # Train the model on a batch.
        model.train()
        for i_batch, batch in (pbar := tqdm(enumerate(train_loader), total=len(train_loader), position=0)):
            str_epoch = f"{i_epoch+1}/{hyperparams['n_epochs']}, {((i%i_checkpoint)/i_checkpoint)*100:.1f}%, "
            str_tr_loss_acc = f"{tr_loss_hist[-2]:.3f}, {tr_acc_hist[-2]:.5f}, "
            str_vl_loss_acc = f"{vl_loss_hist[-1]:.3f}, {vl_acc_hist[-1]:.5f}"
            pbar.set_description(str_epoch+str_tr_loss_acc+str_vl_loss_acc)

            # Make predictions and calculate loss.
            optim.zero_grad()
            y, y_gt, x, w_x_ids, w_x_attn, t_x_ids = get_batch_data(batch, device)
            y_est, attn = model(w_x_ids, w_x_attn, t_x_ids, x)
            l = loss(y_est, y)
            l.backward()
            optim.step()
            if hyperparams['scheduler']:
                scheduler.step()

            # Record model metrics.
            y_hat = torch.argmax(y_est, dim=1).detach().cpu().numpy()
            tr_loss_hist[-1] += l.cpu().detach().item()
            tr_acc_hist[-1] += np.count_nonzero(y_gt == y_hat)
            n_acc += len(y_hat)

            # Update metrics trackers for the checkpoint.
            if i % i_checkpoint == 0:
                # Training metrics
                # If i is 0, we only have a single loss reading and loss/1 is redundant.
                if i > 0:
                    tr_loss_hist[-1] = tr_loss_hist[-1] / i_checkpoint
                tr_loss_hist = np.append(tr_loss_hist, [0.0])
                tr_acc_hist[-1] /= n_acc
                tr_conf_bounds = np.append(tr_conf_bounds, [confidence_bounds(tr_acc_hist[-1], n_acc)])
                tr_acc_hist = np.append(tr_acc_hist, [0.0])
                n_acc = 0.0

                # Validation metrics
                vl_loss, vl_acc, vl_conf = eval(validate_loader, device, model, loss)
                vl_loss_hist = np.append(vl_loss_hist, [vl_loss])
                vl_acc_hist = np.append(vl_acc_hist, [vl_acc])
                vl_conf_bounds = np.append(vl_conf_bounds, [vl_conf])

                # Check to see if we have a new best model.
                if vl_acc > best_vl_acc:
                    best_vl_acc = vl_acc
                    best_model = model.state_dict()

            # Increment the general counter.
            i += 1

        # End of epoch - save a copy of the model and metrics.
        torch.save(model.state_dict(), os.path.join(m_dir, 'model.pt'))
        torch.save(optim.state_dict(), os.path.join(m_dir, 'optim.pt'))
        state = {
            'tr_loss_hist': tr_loss_hist.tolist(), 'tr_acc_hist': tr_acc_hist.tolist(), 'tr_conf_bounds': tr_conf_bounds.tolist(),
            'vl_loss_hist': vl_loss_hist.tolist(), 'vl_acc_hist': vl_acc_hist.tolist(), 'vl_conf_bounds': vl_conf_bounds.tolist(),
            'i': i, 'n_acc': n_acc, 'i_epoch': i_epoch
        }
        with open(os.path.join(m_dir, 'state.json'), 'w') as json_state:
            json.dump(state, json_state)
        with open(os.path.join(m_dir, 'hyperparams.json'), 'w') as json_params:
            json.dump(hyperparams, json_params)

        # Save the current visuals.
        loss_data = [
            {'data_points': tr_loss_hist[1:-1].tolist(), 'label': 'Train', 'color': 'blue'},
            {'data_points': vl_loss_hist[1:].tolist(), 'label': 'Validation', 'color': 'red'}
        ]
        acc_data = [
            {'data_points': tr_acc_hist[1:-1].tolist(), 'bounds': tr_conf_bounds.tolist(), 'label': 'Train', 'color': 'blue'},
            {'data_points': vl_acc_hist[1:].tolist(), 'bounds': vl_conf_bounds.tolist(), 'label': 'Validation', 'color': 'red'}
        ]
        steps = [i * i_checkpoint for i in range(len(tr_acc_hist[1:-1]))]
        viz_graph("Loss Over Time", 'Loss', steps, loss_data, os.path.join(m_dir, 'loss.png'), False)
        viz_graph("Accuracy Over Time", 'Accuracy %', steps, acc_data, os.path.join(m_dir, 'acc.png'), False)

    # Save the best model, according to validation accuracy.
    print(f"Saving best model with validation accuracy: {best_vl_acc}")
    torch.save(best_model, os.path.join(m_dir, 'model.pt'))
    

def orchestrate_training():
    hyperparameters = list(product_dict(**get_hyperparameters()))
    mappings = get_mappings()
    train_ds = IMRDataset('imr_train')
    val_ds = IMRDataset('imr_val')
    #test_ds = IMRDataset('imr_test')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    sys_config = SysConfig()
    
    for i, h in enumerate(hyperparameters):
        m_dir = os.path.join(sys_config.classification_model_path, hyperparam_path(h))
        shutil.rmtree(m_dir, ignore_errors=True)
        os.makedirs(m_dir, exist_ok=True)
        print("\nTraining:")
        for k, v in h.items():
            print(f"{k}: {v}")
        print("---------------------------")
        train_and_eval(train_ds, val_ds, h, device, m_dir)

if __name__ == '__main__':
    orchestrate_training()