import pandas as pd
from ast import literal_eval
from sklearn.model_selection import train_test_split
import random
import numpy as np
import ast
import csv
import itertools
import string
import sys

# from pre_process.load_dataset import load_dataset
# from utils.compute_metrics import system_precision_recall_f1

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


# fix span
SPECIAL_CHARACTERS = string.whitespace

def _contiguous_ranges(span_list):
    """Extracts continguous runs [1, 2, 3, 5, 6, 7] -> [(1,3), (5,7)]."""
    output = []
    for _, span in itertools.groupby(
        enumerate(span_list), lambda p: p[1] - p[0]):
        span = list(span)
        output.append((span[0][1], span[-1][1]))
    return output


def fix_spans(spans, text, special_characters=SPECIAL_CHARACTERS):
    """Applies minor edits to trim spans and remove singletons."""
    cleaned = []
    for begin, end in _contiguous_ranges(spans):
        while text[begin] in special_characters and begin < end:
            begin += 1
        while text[end] in special_characters and begin < end:
            end -= 1
        if end - begin > 1:
            cleaned.extend(range(begin, end + 1))
    return cleaned

# load data
def load_dataset(dataset_path):
    dataset = pd.read_csv(dataset_path)
    print(f"\n> Loading {dataset.shape[0]} examples located at '{dataset_path}'\n")

    dataset["spans"] = dataset.spans.apply(literal_eval)
    texts, spans = dataset["text"], dataset["spans"]
    texts = [text for text in texts]
    spans = [fix_spans(span, texts[i]) for i, span in enumerate(spans)]

    return texts, spans


def load_testset(dataset_path):
    dataset = pd.read_csv(dataset_path)
    print(f"\n> Loading {dataset.shape[0]} test examples located at '{dataset_path}'\n")
    texts = dataset["text"]
    texts = [text for text in texts]
    return texts


def training_validation_split(texts, spans, test_size):
    # Use sklearn function to split the dataset
    training_texts, val_texts, training_spans, val_spans = train_test_split(texts, spans, test_size=test_size)
    # Create list of lists
    training_texts = [train_text for train_text in training_texts]
    training_spans = [fix_spans(training_span, training_texts[i]) for i, training_span in enumerate(training_spans)]
    val_texts = [val_text for val_text in val_texts]
    val_spans = [fix_spans(val_span, val_texts[i]) for i, val_span in enumerate(val_spans)]

    return training_texts, val_texts, training_spans, val_spans


seed_value = 42
random.seed(seed_value)
np.random.seed(seed_value)

# Build a random baseline (yields offsets at random)
# Credits: https://github.com/ipavlopoulos/toxic_spans/blob/master/ToxicSpans_SemEval21.ipynb
random_baseline = lambda text: [i for i, char in enumerate(text) if random.random() > 0.5]

# Load the train, dev and test sets
train_texts, train_spans = load_dataset('data/tsd_train.csv')
val_texts, val_spans = load_dataset('data/tsd_trial.csv')
test_texts, test_spans = load_dataset('data/tsd_test.csv')

# Make random predictions
train_preds = [random_baseline(text) for text in train_texts]
val_preds = [random_baseline(text) for text in val_texts]
test_preds = [random_baseline(text) for text in test_texts]

# Compute performance metrics
train_scores = system_precision_recall_f1(train_preds, train_spans)
dev_scores = system_precision_recall_f1(val_preds, val_spans)
test_scores = system_precision_recall_f1(test_preds, test_spans)

# Print the results
print(f'\n> Train Scores: Precision: {train_scores[0]}, Recall: {train_scores[1]}, F1: {train_scores[2]}')
print(f'\n> Dev Scores: Precision: {dev_scores[0]}, Recall: {dev_scores[1]}, F1: {dev_scores[2]}')
print(f'\n> Test Scores: Precision: {test_scores[0]}, Recall: {test_scores[1]}, F1: {test_scores[2]}')
