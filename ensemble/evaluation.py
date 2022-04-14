import pandas as pd
import numpy as np
import gc
import json
from tensorflow.keras import *
import tensorflow as tf
from tensorflow.keras import *
import tensorflow.keras.backend as K
from sklearn.model_selection import KFold
from sklearn.metrics import classification_report
from transformers import TFRobertaModel,RobertaTokenizer

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
train_set = pd.read_csv("../data/tsd_train.csv")
train_set['spans'] = train_set['spans'].apply(lambda x : json.loads(x))

# toxic_span_dataset = test_set.append(train_set,ignore_index=True)
# toxic_span_dataset['text'] = toxic_span_dataset['text'].apply(lambda x : x.lower())
#
# texts = toxic_span_dataset['text'].to_numpy()
# all_spans = toxic_span_dataset['spans'].to_numpy()
# kf = KFold(n_splits=5)
# train_test_indices = []
# for train_index,test_index in kf.split(texts):
#     train_test_indices.append((train_index,test_index))
#
# train_index,test_index = train_test_indices.pop()
# spans_test = all_spans[test_index]
#
# print(spans_test)

result =  pd.read_csv("ensemble_intersect.txt",sep="\t", header=None)
result[1] = result[1].apply(lambda x : json.loads(x))

f1_toxic = avg_f1(test_set['spans'].to_numpy(),result[1].to_numpy())
print("test F1 = %f"%(f1_toxic))
