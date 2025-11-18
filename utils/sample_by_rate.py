import random

def sample_by_ratio(data):
    positive = [i for i in data if i["score"] == 1]
    negative = [i for i in data if i["score"] == -1]
    
    # 最大采样批次数
    max_batches = min(len(positive) // 8, len(negative) // 2)
    if max_batches == 0:
        return []  # 没法采样，任一类别数据不够

    pos_num = max_batches * 8
    neg_num = max_batches * 2

    sampled_pos = random.sample(positive, pos_num)
    sampled_neg = random.sample(negative, neg_num)
    print(f"positive samples: {len(positive)}; sample {pos_num}")
    print(f"negative samples: {len(negative)}; sample {neg_num}")
    result = sampled_pos + sampled_neg
    random.shuffle(result)
    return result