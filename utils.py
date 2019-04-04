import re
import string
from collections import Counter


def humanize_time(secs):
    mins, secs = divmod(secs, 60)
    hours, mins = divmod(mins, 60)
    return '%02d:%02d:%02d' % (hours, mins, secs)


def convert_tokens(eval_file, qa_id, pp1, pp2):
    answer_dict = {}
    remapped_dict = {}
    for qid, p1, p2 in zip(qa_id, pp1, pp2):
        context = eval_file[str(qid)]["context"]
        spans = eval_file[str(qid)]["spans"]
        uuid = eval_file[str(qid)]["uuid"]
        start_idx = spans[p1][0]
        end_idx = spans[p2][1]
        answer_dict[str(qid)] = context[start_idx: end_idx]
        remapped_dict[uuid] = context[start_idx: end_idx]
    return answer_dict, remapped_dict


def evaluate(eval_file, answer_dict):
    f1 = exact_match = em_start = em_end = total = 0
    for key, value in answer_dict.items():
        total += 1
        ground_truths = eval_file[key]["answers"]
        prediction = value
        em = metric_max_over_ground_truths(
            exact_match_score, prediction, ground_truths)
        exact_match += em[0]
        em_start += em[1]
        em_end += em[2]
        f1 += metric_max_over_ground_truths(f1_score,
                                            prediction, ground_truths)
    exact_match = 100.0 * exact_match / total
    em_start = 100.0 * em_start / total
    em_end = 100.0 * em_end / total
    f1 = 100.0 * f1 / total
    return {'exact_match': exact_match, 'em_start': em_start, 'em_end': em_end, 'f1': f1}


def normalize_answer(s):

    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def f1_score(prediction, ground_truth):
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def exact_match_score(prediction, ground_truth):
    norm_preds, norm_truth = normalize_answer(prediction), normalize_answer(ground_truth)
    if len(norm_preds) > 0:
        preds_split = norm_preds.split()
        truth_split = norm_truth.split()
        em_score = norm_preds == norm_truth
        if len(norm_truth) > 0:
            em_start = preds_split[0] == truth_split[0]
            em_end = preds_split[-1] == truth_split[-1]
        else:
            em_start = em_end = False if len(norm_preds) > 0 else True
    else:
        em_score = em_start = em_end = False
    return [em_score, em_start, em_end]


def metric_max_over_ground_truths(metric_fn, prediction, ground_truths):
    scores_for_ground_truths = []
    for ground_truth in ground_truths:
        score = metric_fn(prediction, ground_truth)
        scores_for_ground_truths.append(score)
    return max(scores_for_ground_truths)
