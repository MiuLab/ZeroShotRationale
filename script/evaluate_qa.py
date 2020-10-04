import transformers.data.metrics.squad_metrics as squad_metrics
import collections

def static_vars(**kwargs):
    def decorate(func):
        for k in kwargs:
            setattr(func, k, kwargs[k])

        return func

    return decorate

@static_vars(answers = None, raw = None)
def get_answer(raw_data):
    if get_answer.answers is None or raw_data != get_answer.raw:
        get_answer.answers = {}
        get_answer.raw = raw_data

        if isinstance(raw_data, dict): #raw data with version
            data = raw_data["data"]
        elif isinstance(raw_data, list): #raw_data["data"]
            data = raw_data

        for entry in data:
            for paragraph in entry["paragraphs"]:
                for qa in paragraph["qas"]:
                    get_answer.answers[qa["id"]] = []

                    if qa["is_impossible"]:
                        get_answer.answers[qa["id"]].append("")
                    else:
                        for answer in qa["answers"]:
                            get_answer.answers[qa["id"]].append(answer["text"])

    return get_answer.answers

def compute_f1(a_gold, a_pred):
    gold_toks = squad_metrics.get_tokens(a_gold)
    pred_toks = squad_metrics.get_tokens(a_pred)
    common = collections.Counter(gold_toks) & collections.Counter(pred_toks)
    num_same = sum(common.values())

    if len(gold_toks) == 0 or len(pred_toks) == 0:
        precision = 1 if len(pred_toks) == 0 else 0
        recall = 1 if len(gold_toks) == 0 else 0
    elif num_same == 0:
        return 0, 0, 0
    else:
        precision = 1.0 * num_same / len(pred_toks)
        recall = 1.0 * num_same / len(gold_toks)

    f1 = (2 * precision * recall) / (precision + recall)
    return precision, recall, f1

def get_raw_scores(answers, pred):
    exact_scores = {}
    precision_scores = {}
    recall_scores = {}
    f1_scores = {}

    for qas_id in answers.keys():
        gold_answers = [answer for answer in answers[qas_id] if squad_metrics.normalize_answer(answer)]

        if not gold_answers:
            gold_answers = [""]

        if qas_id not in pred:
            print("Missing prediction for %s" % qas_id)
            continue

        prediction = pred[qas_id]
        exact_scores[qas_id] = max(squad_metrics.compute_exact(a, prediction) for a in gold_answers)

        max_f1_score = None
        final_precision_score = None
        final_recall_score = None

        for a in gold_answers:
            precision_score, recall_score, f1_score = compute_f1(a, prediction)

            if max_f1_score is None or f1_score > max_f1_score:
                max_f1_score = f1_score
                final_precision_score = precision_score
                final_recall_score = recall_score

        precision_scores[qas_id] = final_precision_score
        recall_scores[qas_id] = final_recall_score
        f1_scores[qas_id] = max_f1_score

    return exact_scores, precision_scores, recall_scores, f1_scores

def make_eval_dict(token_precision, token_recall, token_f1, iou_precision, iou_recall, iou_f1):
    total = len(token_precision)
    return {
        "token scores": collections.OrderedDict(
            [
                ("precision", 100.0 * sum(token_precision.values()) / total),
                ("recall", 100.0 * sum(token_recall.values()) / total),
                ("f1", 100.0 * sum(token_f1.values()) / total)
            ]
        ),
        "IOU scores": collections.OrderedDict(
            [
                ("precision", 100.0 * iou_precision),
                ("recall", 100.0 * iou_recall),
                ("f1", 100.0 * iou_f1)
            ]
        )
    }


def compute_iou_f1_beer(pred, answer):
    num_classifications = {key: 1 for key, value in pred.items()}
    num_truth = {key: len(value) for key, value in answer.items()}
    ious = collections.defaultdict(dict)

    for id in answer.keys():
        if id not in pred:
            print("Missing prediction for %s" % id)
            continue

        prediction = pred[id]
        pred_toks = squad_metrics.get_tokens(prediction)
        pred_counter = collections.Counter(pred_toks)
        best_iou = 0.0

        for ans in answer.get(id, []):
            gold_toks = squad_metrics.get_tokens(ans)
            gold_counter = collections.Counter(gold_toks)
            num_intersection = sum((pred_counter & gold_counter).values())
            num_union = sum((pred_counter | gold_counter).values())
            iou = 1 if num_union == 0 else num_intersection / num_union

            if iou > best_iou:
                best_iou = iou

        ious[id][prediction] = best_iou

    threshold_tps = {}

    for id, iou_results in ious.items():
        threshold_tps[id] = sum(int(x >= 0.5) for x in iou_results.values())

    recalls = list(threshold_tps.get(k, 0.0) / n if n > 0 else 1 for k, n in num_truth.items())
    precisions = list(threshold_tps.get(k, 0.0) / n if n > 0 else 1 for k, n in num_classifications.items())
    recall = sum(recalls) / len(recalls) if len(recalls) > 0 else 1
    precision = sum(precisions) / len(precisions) if len(precisions) > 0 else 1
    f1 = 0 if recall == 0 and precision == 0 else (2 * recall * precision) / (recall + precision)

    return precision, recall, f1

def evaluate(pred, data):
    if not isinstance(pred, dict):
        raise ValueError("Expect pred to be a dictionary.")
    elif not isinstance(data, dict):
        raise ValueError("Expect raw data to be a dictionary or a list.")

    answers = get_answer(data)
    _, precision, recall, f1 = get_raw_scores(answers, pred)
    iou_precision, iou_recall, iou_f1 = compute_iou_f1_beer(pred, answers)
    result_dict = make_eval_dict(precision, recall, f1, iou_precision, iou_recall, iou_f1)
    print(result_dict)
    return result_dict

def get_raw_scores_movie(answers, pred):
    precision_scores = {}
    recall_scores = {}
    f1_scores = {}

    micro_f1_sum = 0.0
    micro_precision_sum = 0.0
    micro_recall_sum = 0.0
    micro_sum = 0

    for qas_id in answers.keys():
        gold_answers = answers[qas_id]

        if not gold_answers:
            gold_answers = [""]

        if qas_id not in pred:
            print("Missing prediction for %s" % qas_id)
            continue

        prediction = pred[qas_id]
        (long, short, longer) = (gold_answers, prediction, "answer") if len(gold_answers) > len(prediction) else (prediction, gold_answers, "prediction")

        f1_score_sum = 0.0
        precision_score_sum = 0.0
        recall_score_sum = 0.0

        for long_answer in long:
            max_f1_score = None
            final_precision_score = None
            final_recall_score = None

            for short_answer in short:
                if longer == "answer":
                    precision_score, recall_score, f1_score = compute_f1(long_answer, short_answer)
                else:
                    precision_score, recall_score, f1_score = compute_f1(short_answer, long_answer)

                if max_f1_score is None or f1_score > max_f1_score:
                    max_f1_score = f1_score
                    final_precision_score = precision_score
                    final_recall_score = recall_score

            f1_score_sum += max_f1_score
            precision_score_sum += final_precision_score
            recall_score_sum += final_recall_score

            micro_f1_sum += max_f1_score
            micro_precision_sum += final_precision_score
            micro_recall_sum += final_recall_score
            micro_sum += 1

        precision_scores[qas_id] = precision_score_sum / len(long)
        recall_scores[qas_id] = recall_score_sum / len(long)
        f1_scores[qas_id] = f1_score_sum / len(long)

    return precision_scores, recall_scores, f1_scores, micro_precision_sum / micro_sum, micro_recall_sum / micro_sum, micro_f1_sum / micro_sum

def make_eval_dict_movie(token_macro_precision, token_macro_recall, token_macro_f1, token_micro_precision, token_micro_recall, token_micro_f1,
                         iou_micro_precision, iou_micro_recall, iou_micro_f1, iou_macro_precision, iou_macro_recall, iou_macro_f1):
    total = len(token_macro_precision)
    return {
        "token scores": collections.OrderedDict(
            [
                ("macro precision", 100.0 * sum(token_macro_precision.values()) / total),
                ("macro recall", 100.0 * sum(token_macro_recall.values()) / total),
                ("macro f1", 100.0 * sum(token_macro_f1.values()) / total),
                ("micro precision", 100.0 * token_micro_precision),
                ("micro recall", 100.0 * token_micro_recall),
                ("micro f1", 100.0 * token_micro_f1)
            ]
        ),
        "IOU scores": collections.OrderedDict(
            [
                ("macro precision", 100.0 * iou_macro_precision),
                ("macro recall", 100.0 * iou_macro_recall),
                ("macro f1", 100.0 * iou_macro_f1),
                ("micro precision", 100.0 * iou_micro_precision),
                ("micro recall", 100.0 * iou_micro_recall),
                ("micro f1", 100.0 * iou_micro_f1)
            ]
        )
    }


def compute_iou_f1(pred, answer):
    num_classifications = {key: len(value) for key, value in pred.items()}
    num_truth = {key: len(value) for key, value in answer.items()}
    ious = collections.defaultdict(dict)

    for id in answer.keys():
        for prediction in pred.get(id, []):
            pred_toks = squad_metrics.get_tokens(prediction)
            pred_counter = collections.Counter(pred_toks)
            best_iou = 0.0

            for ans in answer.get(id, []):
                gold_toks = squad_metrics.get_tokens(ans)
                gold_counter = collections.Counter(gold_toks)
                num_intersection = sum((pred_counter & gold_counter).values())
                num_union = sum((pred_counter | gold_counter).values())
                iou = 1 if num_union == 0 else num_intersection / num_union

                if iou > best_iou:
                    best_iou = iou

            ious[id][prediction] = best_iou

    threshold_tps = {}

    for id, iou_results in ious.items():
        threshold_tps[id] = sum(int(x >= 0.5) for x in iou_results.values())

    micro_recall = sum(threshold_tps.values()) / sum(num_truth.values()) if sum(num_truth.values()) > 0 else 1
    micro_precision = sum(threshold_tps.values()) / sum(num_classifications.values()) if sum(num_classifications.values()) > 0 else 1
    micro_f1 = 0 if micro_recall == 0 and micro_precision == 0 else (2 * micro_recall * micro_precision) / (micro_recall + micro_precision)

    macro_recalls = list(threshold_tps.get(k, 0.0) / n if n > 0 else 1 for k, n in num_truth.items())
    macro_precisions = list(threshold_tps.get(k, 0.0) / n if n > 0 else 1 for k, n in num_classifications.items())
    macro_recall = sum(macro_recalls) / len(macro_recalls) if len(macro_recalls) > 0 else 1
    macro_precision = sum(macro_precisions) / len(macro_precisions) if len(macro_precisions) > 0 else 1
    macro_f1 = 0 if macro_recall == 0 and macro_precision == 0 else (2 * macro_recall * macro_precision) / (macro_recall + macro_precision)

    return micro_precision, micro_recall, micro_f1, macro_precision, macro_recall, macro_f1

def evaluate_movie(pred, answer):
    if not isinstance(pred, dict):
        raise ValueError("Expect pred to be a dictionary.")
    elif not isinstance(answer, dict):
        raise ValueError("Expect answer to be a dictionary.")

    token_macro_precision, token_macro_recall, token_macro_f1, token_micro_precision, token_micro_recall, token_micro_f1 = get_raw_scores_movie(answer, pred)
    iou_micro_precision, iou_micro_recall, iou_micro_f1, iou_macro_precision, iou_macro_recall, iou_macro_f1 = compute_iou_f1(pred, answer)

    result_dict = make_eval_dict_movie(token_macro_precision, token_macro_recall, token_macro_f1, token_micro_precision, token_micro_recall, token_micro_f1,
                                       iou_micro_precision, iou_micro_recall, iou_micro_f1, iou_macro_precision, iou_macro_recall, iou_macro_f1)
    print(result_dict)
    return result_dict
