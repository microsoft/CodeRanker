import sys 
import json
from xmlrpc.client import TRANSPORT_ERROR
from tqdm import tqdm
import scipy
from scipy import special
import re
import numpy as np
import pdb 
from sklearn.metrics import precision_recall_fscore_support
import os


def get_data(logits_prediction_file, data_file, labels_file, prompt_key='prompt'):
    # outfile only contains logits
    # datafile contains all data and they should be in the same order
    all_logits = []
    with open(logits_prediction_file, 'r') as f:
        for line in tqdm(f):
            # skip header
            if line.startswith('index'):
                continue
            line = line.strip()
            logits = line.split('\t')[1]
            logits = eval(logits)
            all_logits.append(logits)
    print(len(all_logits))

    with open(data_file, 'r') as f:
        data_dict = {}
        for i, line in tqdm(enumerate(f)):
            data = json.loads(line)
            data["logits"] = all_logits[i]
            
            key = data[prompt_key]
            if key not in data_dict:
                data_dict[key] = []
            data_dict[key].append(data)
        assert(i == len(all_logits) -1 )
        print(len(data_dict))

    with open(labels_file, 'r') as f:
        labels = []
        for line in tqdm(f):
            line = line.strip()
            labels.append(line)
    
    return data_dict, labels

def process(data_dict, labels, label_key, pass_label="Correct"):
    # convert data_dict to list of data
    data = list(data_dict.values())
    pass_idx = labels.index(pass_label)
    
    probs = [[scipy.special.softmax(d['logits'], axis=0)[pass_idx] for d in data[i]] for i in range(len(data))]

    grouped_labels = [[d['ternary_label'] for d in data[i]] for i in range(len(data))]

    # make all rows the same length
    max_len = max([len(d) for d in data])
    probs = np.array([d + [-1] * (max_len - len(d)) for d in probs] ) # num_prompts x max_num_suggestions
    grouped_labels = np.array([d + ["error"] * (max_len - len(d)) for d in grouped_labels]) # num_prompts x max_num_suggestions

    results = {}
    # compute vanillar metrics
    res = compute_vanilla_metrics(grouped_labels)

    # compute ranked accuracies
    print("Renked metrics")
    res = compute_metrics(probs, grouped_labels)

def pass_at_k(n, c, k):
    """
    :param n: total number of samples
    :param c: number of correct samples
    :param k: k in pass@$k$
    """
    if n - c < k: return 1.0
    return 1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1))

def compute_vanilla_metrics(grouped_labels):    
    pass_1 = []
    exec_1 = []
    pass_5 = []
    exec_5 = []
    pass_100 = []
    exec_100 = []
    for i in range(len(grouped_labels)):
        labels = grouped_labels[i]
        num_correct = 0
        num_error_free = 0
        for j in range(len(labels)):
            if labels[j] == "Correct":
                num_correct += 1
            if labels[j] != "Execution error":
                num_error_free += 1
        pass_1.append(pass_at_k(len(labels), num_correct, 1))
        exec_1.append(pass_at_k(len(labels), num_error_free, 1))
        pass_5.append(pass_at_k(len(labels), num_correct, 5))
        exec_5.append(pass_at_k(len(labels), num_error_free, 5))

        pass_100.append(pass_at_k(len(labels), num_correct, len(labels)))
        exec_100.append(pass_at_k(len(labels), num_error_free, len(labels)))

    print("Vanilla model")
    res = {
        "pass_1": np.mean(pass_1),
        "pass_5": np.mean(pass_5),
        "exec_1": np.mean(exec_1),
        "exec_5": np.mean(exec_5),
        "pass_100": np.mean(pass_100),
        "exec_100": np.mean(exec_100),
    }
    print(res)

    return res
    
        
def compute_metrics(probs, grouped_labels):
    # top-1 accuracy
    best = np.argmax(probs, axis=1)
    predictions = grouped_labels[np.arange(len(grouped_labels)), best]
    correct = predictions == "Correct"
    error_free = predictions != "Execution error"

    ranker_accuracy = np.mean(correct)
    ranker_accuracy_error_free = np.mean(error_free)

    # top-5 accuracy
    best = np.argsort(probs, axis=1)[:, -5:]
    predictions_top_5 = [] 
    for i in range(len(best)):
        predictions_top_5.append([grouped_labels[i, j] for j in best[i]])
    predictions_top_5 = np.array(predictions_top_5)
    correct_top_5 = predictions_top_5 == "Correct"
    error_free_top_5 = predictions_top_5 != "Execution error"

    ranker_accuracy_top_5 = np.mean(correct_top_5)
    ranker_accuracy_error_free_top_5 = np.mean(error_free_top_5)

    # top-5 best accuracy
    best_accuracy_top_5 = np.max(correct_top_5, axis=1).mean()
    best_accuracy_error_free_top_5 = np.max(error_free_top_5, axis=1).mean()
    
    res = {
        "pass_1": round(ranker_accuracy, 3),
        "pass_5": round(best_accuracy_top_5, 3),
        "exec_1": round(ranker_accuracy_error_free, 3),
        "exec_5": round(best_accuracy_error_free_top_5, 3),
    }

    print(res)
    return res


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Compute metrics with ranker")
    parser.add_argument("--data_file", type=str, default="../ranker_datasets/gpt_neo_125m/val.json")
    parser.add_argument("--logits_prediction_file", type=str, default="~/ranker_model_for_gpt_neo_125m/predict_results_test.txt")
    parser.add_argument("--labels_file", type = str, default = "../ranker_datasets/gpt_neo_125m/labels_binary.txt")
    parser.add_argument("--task", type = str, default = "binary")
 
    args = parser.parse_args()

    label_key = args.task + "_label"

    data_dict, labels = get_data(args.logits_prediction_file, args.data_file, args.labels_file)
    process(data_dict, labels, label_key)
    