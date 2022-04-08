import os
from csv import DictReader
from ast import literal_eval
import itertools
import string
import pandas as pd

SPECIAL_CHARACTERS = string.whitespace

def _contiguous_ranges(span_list):
    """
    Extracts continguous runs [1, 2, 3, 5, 6, 7] -> [(1,3), (5,7)].
    Returns begin and end inclusive
    """
    output = []
    for _, span in itertools.groupby(enumerate(span_list), lambda p: p[1] - p[0]):
        span = list(span)
        output.append((span[0][1], span[-1][1]))
    return output

def fix_spans(spans, text, special_characters=SPECIAL_CHARACTERS, collapse=False):
    """
    Applies minor edits to trim spans and remove singletons.
    If spans begin/end in the middle of a word, correct according to collapse strategy:
        If false, expand spans until word limits; if true collapse until word limits
    """
    cleaned_spans = []
    for begin, end in _contiguous_ranges(spans):
        # Trim spans
        while text[begin] in special_characters and begin < end:
            begin += 1
        while text[end] in special_characters and begin < end:
            end -= 1
        # Assert word limits
        while 0 < begin < end and text[begin - 1].isalnum():
            offset_move = 1 if collapse else -1
            begin += offset_move
        while len(text) - 1 > end > begin and text[end + 1].isalnum():
            offset_move = -1 if collapse else 1
            end += offset_move
        # Remove singletons
        if end - begin > 1:
            cleaned_spans.extend(range(begin, end + 1))
    return cleaned_spans



def get_sentences_from_data_split(data_path, split):
    sentences = []
    original_spans = []
    fixed_spans = []
    with open(os.path.join(data_path, split + '.csv'), encoding='utf-8') as csv_file:
        reader = DictReader(csv_file)
        for row in reader:
            if split == 'tsd_test':
                span = fixed_span = []
            else:
                span = literal_eval(row['spans'])
                fixed_span = fix_spans(span, row['text'])
            sentences.append(row['text'])
            original_spans.append(span)
            fixed_spans.append(fixed_span)

    return sentences, original_spans, fixed_spans


data_path = 'data'
train_sentence, train_original_spans, train_fixed_spans = get_sentences_from_data_split(data_path, split='tsd_train')
test_sentence, test_original_spans, test_fixed_spans = get_sentences_from_data_split(data_path, split='tsd_test')
trial_sentence, trial_original_spans, trial_fixed_spans = get_sentences_from_data_split(data_path, split='tsd_trial')
dftrain = pd.DataFrame()
dftrain['spans'] = train_fixed_spans
dftrain['text'] = train_sentence
dftrain.to_csv('clean_train.csv', index=False)
dftest = pd.DataFrame()
dftest['spans'] = test_fixed_spans
dftest['text'] = test_sentence
dftest.to_csv('clean_test.csv', index=False)
dftrial = pd.DataFrame()
dftrial['spans'] = trial_fixed_spans
dftrial['text'] = trial_sentence
dftrial.to_csv('clean_trial.csv', index=False)
