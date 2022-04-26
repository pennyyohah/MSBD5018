import pandas as pd
import numpy as np
import gc
import json

# F1 score
def system_precision_recall_f1(toxic_char_preds, gold_char_offsets):
    def per_post_precision_recall_f1(predictions, gold):
        if len(gold) == 0:
            return [1.0, 1.0, 1.0] if len(predictions) == 0 else [0.0, 0.0, 0.0]

        if len(predictions) == 0:
            return [0.0, 0.0, 0.0]

        predictions_set = set(predictions)
        gold_set = set(gold)
        nom = len(predictions_set.intersection(gold_set))
        precision = nom / len(predictions_set)
        recall = nom / len(gold_set)
        f1_score = (2 * nom) / (len(predictions_set) + len(gold_set))

        return [float(precision), float(recall), float(f1_score)]

    # get the respective metrics per post
    precision_recall_f1_scores = [per_post_precision_recall_f1(toxic_offsets, gold_offsets) for
                                  toxic_offsets, gold_offsets in zip(toxic_char_preds, gold_char_offsets)]

    # compute average precision, recall and f1 score of all posts
    return np.array(precision_recall_f1_scores).mean(axis=0)

def f1(preds,trues):
    if len(trues) == 0:
        return 1. if len(preds) == 0 else 0.
    if len(preds) == 0:
        return 0.
    predictions_set = set(preds)
    gold_set = set(trues)
    nom = 2 * len(predictions_set.intersection(gold_set))
    denom = len(predictions_set) + len(gold_set)
    return float(nom)/float(denom)

def avg_f1(preds,trues):
    avg_f1_total = 0.0
    for pred,true in zip(preds,trues):
        avg_f1_total += f1(pred,true)
    return avg_f1_total/len(preds)

test_set = pd.read_csv("../data/tsd_test.csv")
test_set['spans'] = test_set['spans'].apply(lambda x : json.loads(x))
# train_set = pd.read_csv("../data/tsd_train.csv")
# train_set['spans'] = train_set['spans'].apply(lambda x : json.loads(x))

result =  pd.read_csv("ensemble_vote.txt",sep="\t", header=None)
result[1] = result[1].apply(lambda x : json.loads(x))

# f1_toxic = avg_f1(test_set['spans'].to_numpy(),result[1].to_numpy())
# print("test F1 = %f"%(f1_toxic))

test_scores=system_precision_recall_f1(test_set['spans'].to_numpy(), result[1].to_numpy())
print(f'\nPrecision: {test_scores[0]}, Recall: {test_scores[1]}, F1: {test_scores[2]}')
