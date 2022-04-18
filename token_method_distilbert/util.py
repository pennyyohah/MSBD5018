import itertools
import string
import torch

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


def preds2spans(preds, masks, offsets):
    """
    :param preds: tensor (batch, seq_len)
    :param masks: tensor (batch, seq_len)
    :param offsets: tensor (batch, seq_len, 2)
    :returns spans: list[set] (batch, spans)
    """
    out_spans = []
    for sent_id in range(preds.shape[0]):
        n_tokens = masks[sent_id].sum()
        token_preds = preds[sent_id, 1:n_tokens + 1]
        token_offsets = offsets[sent_id, 1:n_tokens + 1]

        pred_spans = set()
        for token_id in range(n_tokens):
            # Add token's offsets
            if token_preds[token_id] == 1:
                pred_spans.update(range(token_offsets[token_id, 0], token_offsets[token_id, 1]))
                # Add offsets between two toxic tokens (whitespaces, hyphens, ...)
                if token_id != 0 and token_preds[token_id - 1] == 1:
                    pred_spans.update(range(token_offsets[token_id - 1, 1], token_offsets[token_id, 0]))
        pred_spans = pred_spans - {-1}

        out_spans.append(pred_spans)
    return out_spans

def preds2spans(preds, masks, offsets):
    """
    :param preds: tensor (batch, seq_len)
    :param masks: tensor (batch, seq_len)
    :param offsets: tensor (batch, seq_len, 2)
    :returns spans: list[set] (batch, spans)
    """
    out_spans = []
    for sent_id in range(preds.shape[0]):
        n_tokens = masks[sent_id].sum()
        token_preds = preds[sent_id, 1:n_tokens + 1]
        token_offsets = offsets[sent_id, 1:n_tokens + 1]

        pred_spans = set()
        for token_id in range(n_tokens):
            # Add token's offsets
            if token_preds[token_id] == 1:
                pred_spans.update(range(token_offsets[token_id, 0], token_offsets[token_id, 1]))
                # Add offsets between two toxic tokens (whitespaces, hyphens, ...)
                if token_id != 0 and token_preds[token_id - 1] == 1:
                    pred_spans.update(range(token_offsets[token_id - 1, 1], token_offsets[token_id, 0]))
        pred_spans = pred_spans - {-1}

        out_spans.append(pred_spans)
    return out_spans

def f1(predictions, gold):
    """
    F1 (a.k.a. DICE) operating on two lists of offsets (e.g., character).
    >>> assert f1([0, 1, 4, 5], [0, 1, 6]) == 0.5714285714285714
    :param predictions: a list of predicted offsets
    :param gold: a list of offsets serving as the ground truth
    :return: a score between 0 and 1
    """
    if len(gold) == 0:
        return 1. if len(predictions) == 0 else 0.
    nom = 2 * len(set(predictions).intersection(set(gold)))
    denom = len(set(predictions)) + len(set(gold))
    return nom / denom

def f1_score(preds, targets, reduce=True):
    """
    Mean batch f1-score
    :param preds: list (batch, spans)
    :param targets: tensor (batch, seq_len)
    :param reduce: If True it returns all individual metrics
    """
    f1s = []
    for sent_id in range(len(preds)):
        predicted_spans = preds[sent_id]
        target_spans = set(targets[sent_id].tolist()) - {-1}
        f1s.append(f1(predicted_spans, target_spans))
    if not reduce:
        return f1s
    return sum(f1s) / len(f1s)


def CEL_label_smoothing(pred, target, smoothing=0., mask=None):
    batch, n_classes = pred.shape

    pred = pred.log_softmax(dim=-1)
    oh_mask = torch.arange(n_classes).unsqueeze(0).expand(batch, n_classes).to(pred.device)
    exp_target = target.unsqueeze(-1).expand(batch, n_classes)
    one_hot = exp_target.eq(oh_mask)

    base = torch.ones(batch, n_classes).to(pred.device) * smoothing / (n_classes - 1)
    smoothed = base + one_hot * (1 - smoothing - smoothing / (n_classes - 1))

    loss = -pred * smoothed
    loss = loss.sum(-1)

    if mask is not None:
        loss = loss * mask
        return loss.sum() / mask.sum()

    return loss.mean()
