from typing import Callable
import re
import ast

def parse_ICD_result(text):

    pattern = r"<answer>(.*?)</answer>"
    answers = re.findall(pattern, text, re.DOTALL)
    if len(answers) != 0:
        answers = ast.literal_eval(answers[0])
        answers = [ans.split(":")[0].replace(".", "") for ans in answers]
        return answers
    else:
        raise Exception ("No answer found")

def micro_f1(y_true_list, y_pred_list):
    global_tp = global_fp = global_fn = 0
    for true, pred in zip(y_true_list, y_pred_list):
        true_set, pred_set = set(true), set(pred)
        global_tp += len(true_set & pred_set)  # 交集
        global_fp += len(pred_set - true_set)  # 预测多出的词
        global_fn += len(true_set - pred_set)  # 未预测到的词
    precision = global_tp / (global_tp + global_fp) if (global_tp + global_fp) > 0 else 0
    recall = global_tp / (global_tp + global_fn) if (global_tp + global_fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall + 1e-10)
    return f1

def f1_reward(completions, **kwargs):
    completion_contents = [completion[0]["content"] for completion in completions]
    ground_truth = [reward_model["ground_truth"] for reward_model in kwargs['reward_model']]
    res = []
    for c, gt in zip(completion_contents, ground_truth):
        print("content", c)
        print("ground_truth", gt)
        try:
            parse_c = parse_ICD_result(c)
        except Exception as e:
            res.append(0)
            continue
        parse_gt = parse_ICD_result(gt)
        res.append(micro_f1([parse_gt], [parse_c]))
    return res

def hierarchical_reward(completions, **kwargs):
    completion_contents = [completion[0]["content"] for completion in completions]
    ground_truth = [reward_model["ground_truth"] for reward_model in kwargs['reward_model']]
    res = []
    for c, gt in zip(completion_contents, ground_truth):
        print("content", c)
        print("ground_truth", gt)
        try:
            pred_codes = parse_ICD_result(c)
        except Exception as e:
            res.append(0)
            continue
        real_codes = parse_ICD_result(gt)
        real_codes_sub = [code[:3] for code in real_codes]
        pred_codes_new = list(set(pred_codes))
        total_score = 0
        for pred_code in pred_codes_new:
            if pred_code in real_codes:
                total_score += 1.0
            elif len(pred_code) >= 3 and pred_code[:3] in real_codes_sub:
                total_score += 0.5
        mean_score = total_score / len(pred_codes)
        res.append(mean_score)
    return res


def get_reward_funcs() -> list[Callable]:
    REWARD_FUNCS_REGISTRY = {
        "F1": f1_reward,
        "Hierarchical": hierarchical_reward,
    }
    reward_funcs = [REWARD_FUNCS_REGISTRY[func] for func in
                    ['F1', 'Hierarchical']]

    return reward_funcs



