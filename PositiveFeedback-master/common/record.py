import os, json
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
import torch
import copy
from tqdm import tqdm

def get_record_output_path(output_dir, keys=['loss', 'grad-sim', 'grad-mag', 'auc'], clear=True):
    output_dir = os.path.join(output_dir, 'per-step')
    os.makedirs(output_dir, exist_ok=True)
    output = {}
    for key in keys:
        output[key] = os.path.join(output_dir, f'{key}.jsonl')
        if clear: 
            with open(output[key], 'w') as f: pass
    return output

def record(record_path, indexes, poisoneds, values, metric_name):
    with open(record_path, 'a') as f:
        output_jsons = []
        for i in range(len(indexes)):
            output_jsons.append({
                'poisoned': poisoneds[i],
                'qid': indexes[i],
                metric_name: values[i]
            })
            
            for key, val in output_jsons[-1].items():
                if isinstance(val, torch.Tensor): 
                    output_jsons[-1][key] = val.item()
        
        f.write(json.dumps(output_jsons) + '\n')
        
def plot(
    record_path, output_dir, 
    use_step_zero = False, use_marker = False, use_presum = False,
    merge_interval = 1, metric_name = 'loss', max_steps = None, title = None,
    label_key = 'poisoned', label_names = {0: 'clean', 1: 'poisoned'},
    mark_peaks = [], mark_troughs = [], y_bounds = [], linewidth = 2.5
):
    name_label_map = {name: label for label, name in label_names.items()}
    
    merge_interval = max(1, merge_interval)
    average_metrics = defaultdict(lambda: defaultdict(float))
    cur_dict = defaultdict(list)
    last_step = None
    with open(record_path, 'r') as f:
        for step, line in enumerate(f):
            batch_loss_jsons = json.loads(line)
            for label in label_names.keys():
                cur_dict[label].extend([js[metric_name] for js in batch_loss_jsons if js[label_key] == label])
            
            if (step + 1) % merge_interval == 0 or (use_step_zero and step == 0):
                for label in label_names.keys():
                    average_metrics[step + 1][label] = np.mean(cur_dict[label])
                if not use_presum:
                    cur_dict = defaultdict(list)
                last_step = step
            
            if max_steps and step >= max_steps:
                break
    
    if not last_step: return

    plt.figure(figsize=(8, 6))
    sorted_average_metrics = sorted(list(average_metrics.keys()))
    valid_keys = average_metrics[last_step + 1].keys()
    values = defaultdict(list)
    colors = {}

    for step in average_metrics:
        for label in label_names.keys():
            values[label].append(average_metrics[step][label])

    for label in valid_keys:
        color = 'C{}'.format(list(label_names.values()).index(label_names[label]) % 10)
        color = f'C{label}'
        if use_marker:
            line, = plt.plot(
                sorted_average_metrics, values[label],
                label=label_names[label], linestyle='-', linewidth=linewidth, marker='o', color=color
            )
        else:
            line, = plt.plot(
                sorted_average_metrics, values[label],
                label=label_names[label], linestyle='-', linewidth=linewidth, color=color
            )
        colors[label] = color

    for name in mark_peaks:
        label = name_label_map[name]
        peaks = max(values[label], key=lambda x: x) if len(values[label]) > 0 else None
        if peaks is not None:
            plt.axvline(x=sorted_average_metrics[list(values[label]).index(peaks)], color=colors[label], linestyle='--', label=f'{label_names[label]} Peak')
    
    for name in mark_troughs:
        label = name_label_map[name]
        troughs = min(values[label], key=lambda x: x) if len(values[label]) > 0 else None
        if troughs is not None:
            plt.axvline(x=sorted_average_metrics[list(values[label]).index(troughs)], color=colors[label], linestyle='--', label=f'{label_names[label]} Trough')

    plt.legend()

    if title:
        plt.title(title)
    plt.xlabel('Steps')
    plt.ylabel(metric_name.capitalize())
    if len(y_bounds) > 0:
        plt.ylim(y_bounds[0], y_bounds[1])

    grid_color = 'lightgray'
    plt.grid(True, axis='y', color=grid_color, linestyle='-', linewidth=0.5)
    for spine in plt.gca().spines.values():
        spine.set_color(grid_color)

    plt.savefig(os.path.join(output_dir, f'{metric_name}.png'), dpi=300, bbox_inches='tight')
    plt.close()


def selectedStepPlot(
    record_path, output_dir, 
    use_step_zero = False, use_marker = False, use_presum = False,
    merge_interval = 1, metric_name = 'loss', max_steps = None, title = None,
    label_key = 'poisoned', label_names = {0: 'clean', 1: 'poisoned'},
    mark_peaks = [], mark_troughs = [], y_bounds = [], linewidth = 2.5,
    select_step_size = 1
):
    name_label_map = {name: label for label, name in label_names.items()}
    
    merge_interval = max(1, merge_interval)
    average_metrics = defaultdict(lambda: defaultdict(float))
    cur_dict = defaultdict(list)
    last_step = None
    with open(record_path, 'r') as f:
        step = 0
        for _, line in enumerate(f):
            step += merge_interval
            
            batch_loss_jsons = json.loads(line)
            for label in label_names.keys():
                cur_dict[label].extend([js[metric_name] for js in batch_loss_jsons if js[label_key] == label])
            
            if step % (merge_interval * select_step_size) == 0 or \
                (use_step_zero and step == 0):
                for label in label_names.keys():
                    average_metrics[step + 1][label] = np.mean(cur_dict[label])
                if not use_presum:
                    cur_dict = defaultdict(list)
                last_step = step
                
            if max_steps and step >= max_steps:
                break
    
    valid_keys = average_metrics[last_step].keys()    
    if not last_step: return

    plt.figure(figsize=(8, 6))
    sorted_average_metrics = sorted(list(average_metrics.keys()))
    
    values = defaultdict(list)
    colors = {}

    for step in average_metrics:
        for label in label_names.keys():
            values[label].append(average_metrics[step][label])

    for label in valid_keys:
        color = 'C{}'.format(list(label_names.values()).index(label_names[label]) % 10)
        color = f'C{label}'
        if use_marker:
            line, = plt.plot(
                sorted_average_metrics, values[label],
                label=label_names[label], linestyle='-', linewidth=linewidth, marker='o', color=color
            )
        else:
            line, = plt.plot(
                sorted_average_metrics, values[label],
                label=label_names[label], linestyle='-', linewidth=linewidth, color=color
            )
        colors[label] = color

    for name in mark_peaks:
        label = name_label_map[name]
        peaks = max(values[label], key=lambda x: x) if len(values[label]) > 0 else None
        if peaks is not None:
            plt.axvline(x=sorted_average_metrics[list(values[label]).index(peaks)], color=colors[label], linestyle='--', label=f'{label_names[label]} Peak')
    
    for name in mark_troughs:
        label = name_label_map[name]
        troughs = min(values[label], key=lambda x: x) if len(values[label]) > 0 else None
        if troughs is not None:
            plt.axvline(x=sorted_average_metrics[list(values[label]).index(troughs)], color=colors[label], linestyle='--', label=f'{label_names[label]} Trough')

    plt.legend()

    if title:
        plt.title(title)
    plt.xlabel('Steps')
    plt.ylabel(metric_name.capitalize())
    if len(y_bounds) > 0:
        plt.ylim(y_bounds[0], y_bounds[1])

    grid_color = 'lightgray'
    plt.grid(True, axis='y', color=grid_color, linestyle='-', linewidth=0.5)
    for spine in plt.gca().spines.values():
        spine.set_color(grid_color)

    plt.savefig(os.path.join(output_dir, f'{metric_name}.png'), dpi=300, bbox_inches='tight')
    plt.close()