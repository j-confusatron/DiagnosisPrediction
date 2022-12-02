from genericpath import isdir
import json
import matplotlib.pyplot as plt
import seaborn as sn
import pandas as pd
import os
import numpy as np

def visualize_trainer_data(root, model_data):
    for arch, arch_models in model_data.items():
        for model, data in arch_models.items():
            loss_data = [
                {'data_points': data['train_loss'], 'label': 'Train', 'color': 'blue'},
                {'data_points': data['val_loss'], 'label': 'Validation', 'color': 'red'}
            ]
            acc_data = [
                {'data_points': data['train_acc'], 'bounds': data['train_conf'], 'label': 'Train', 'color': 'blue'},
                {'data_points': data['val_acc'], 'bounds': data['val_conf'], 'label': 'Validation', 'color': 'red'}
            ]
            viz_graph('Loss Over Time', 'Loss', data['steps'], loss_data, viz_filepath=os.path.join(root, arch, model, 'loss.png'), show_graph=True)
            viz_graph('Accuracy Over Time', 'Acc', data['steps'], acc_data, viz_filepath=os.path.join(root, arch, model, 'acc.png'), show_graph=True)
    

def viz_graph(title, y_axis_title, steps, data_points_collection, viz_filepath=None, show_graph=False):
    for data_points in data_points_collection:
        plt.plot(steps, data_points['data_points'], label=data_points['label'], color=data_points['color'])
        if 'bounds' in data_points:
            pos = np.add(data_points['data_points'], data_points['bounds'])
            neg = np.subtract(data_points['data_points'], data_points['bounds'])
            plt.fill_between(steps, pos, neg, color=data_points['color'], alpha=0.1)
    plt.title(title)
    plt.ylabel(y_axis_title)
    plt.xlabel('Steps')
    plt.legend()
    if viz_filepath:
        plt.savefig(viz_filepath)
    if show_graph:
        plt.show()
    plt.clf()

def viz_confusion_matrix(confusion_matrix, labels, viz_filepath=None, show_graph=False, figsize=(12,10), font_scale=.9, size=6):
    df_cm = pd.DataFrame(confusion_matrix, index=labels, columns=labels)
    plt.figure(figsize=figsize)
    sn.set(font_scale=font_scale)
    sn.heatmap(df_cm, annot=True, annot_kws={"size": size})
    plt.title("Confusion Matrix")
    plt.ylabel("Ground Truth")
    plt.xlabel("Predicted")
    plt.tight_layout()
    if viz_filepath:
        plt.savefig(viz_filepath)
    if show_graph:
        plt.show()
    plt.clf()

def get_model_data(dir):
    with open(os.path.join(dir, 'trainer_state.json'), 'r') as state_file:
        trainer_state = json.load(state_file)

    steps = list(sorted(set([log['step'] for log in trainer_state['log_history']])))
    train_loss = [log['train_loss'] for log in trainer_state['log_history'] if 'train_loss' in log]
    train_acc = [log['train_accuracy'] for log in trainer_state['log_history'] if 'train_accuracy' in log]
    train_conf = [log['train_conf_bound'] for log in trainer_state['log_history'] if 'train_conf_bound' in log]
    val_loss = [log['eval_loss'] for log in trainer_state['log_history'] if 'eval_loss' in log]
    val_acc = [log['eval_accuracy'] for log in trainer_state['log_history'] if 'eval_accuracy' in log]
    val_conf = [log['eval_conf_bound'] for log in trainer_state['log_history'] if 'eval_conf_bound' in log]

    results = {
        'steps': steps,
        'train_loss': train_loss,
        'train_acc': train_acc,
        'train_conf': train_conf,
        'val_loss': val_loss,
        'val_acc': val_acc,
        'val_conf': val_conf
    }
    return results

def get_all_model_data(root):
    data = {}

    # Get the model architecture and instance.
    for arch in os.listdir(root):
        data[arch] = {}
        p_arch = os.path.join(root, arch)
        if os.path.isdir(p_arch):
            for model in os.listdir(p_arch):
                p_model = os.path.join(p_arch, model)
                if os.path.isdir(p_model):

                    # Find the last checkpoint.
                    checkpoints = {}
                    for chk in os.listdir(p_model):
                        p_chk = os.path.join(p_model, chk)
                        if os.path.isdir(p_chk):
                            checkpoints[chk[chk.rindex('-')+1:]] = p_chk

                    # Get the checkpoint data.
                    if len(checkpoints) > 0:
                        keys = sorted([int(k) for k in checkpoints.keys()])
                        data[arch][model] = get_model_data(checkpoints[str(keys[-1])])

    return data


if __name__ == '__main__':
    path_model_base = os.path.join('d:', os.sep, 'medlangmodel', 'ner')
    model_data = get_all_model_data(path_model_base)
    visualize_trainer_data(path_model_base, model_data)