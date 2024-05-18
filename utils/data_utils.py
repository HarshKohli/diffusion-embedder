# Author: Harsh Kohli
# Date Created: 23-04-2024

import torch
from datasets import load_dataset


def load_all_datasets(data_dir):
    dataset = load_dataset("json", data_dir=data_dir, split='train')
    return dataset.train_test_split(test_size=0.01)


def preprocess_dataset(all_data, tokenizer, max_tokens):
    inputs = tokenizer(all_data['query'], max_length=max_tokens, padding="max_length", truncation=True,
                       return_tensors="pt")
    all_data["question_ids"] = inputs.input_ids

    para_tokens = tokenizer(all_data['positive'], max_length=max_tokens, padding="max_length", truncation=True,
                            return_tensors="pt")
    all_data["para_ids"] = para_tokens.input_ids

    return all_data


def preprocess_test_dataset(all_data, tokenizer, max_tokens):
    inputs = tokenizer(all_data['query'], max_length=max_tokens, padding="max_length", truncation=True,
                       return_tensors="pt")
    all_data["question_ids"] = inputs.input_ids

    para_tokens = tokenizer(all_data['positive'], max_length=max_tokens, padding="max_length", truncation=True,
                            return_tensors="pt")
    all_data["para_ids"] = para_tokens.input_ids

    neg_tokens = tokenizer(all_data['negative'], max_length=max_tokens, padding="max_length", truncation=True,
                           return_tensors="pt")
    all_data["neg_ids"] = neg_tokens.input_ids

    return all_data


def collate_fn(examples):
    question_ids = torch.stack([example["question_ids"] for example in examples])
    para_ids = torch.stack([example["para_ids"] for example in examples])
    if "neg_ids" not in examples[0]:
        return {"question_ids": question_ids, "para_ids": para_ids}
    neg_ids = torch.stack([example["neg_ids"] for example in examples])
    return {"question_ids": question_ids, "para_ids": para_ids, "neg_ids": neg_ids}


def write_logs(total_correct_diff, total_correct_base, num_samples, dist_query_to_pred, dist_query_to_pos,
               dist_query_to_neg, dist_pred_to_pos, dist_pred_to_neg, sim_pos, sim_neg, sim_pos_base, sim_neg_base,
               log_file, logger):
    f = open(log_file, 'a', encoding='utf8')
    info = [str(sum(dist_query_to_pred) / len(dist_query_to_pred)),
            str(sum(dist_query_to_pos) / len(dist_query_to_pos)),
            str(sum(dist_query_to_neg) / len(dist_query_to_neg)),
            str(sum(dist_pred_to_pos) / len(dist_pred_to_pos)),
            str(sum(dist_pred_to_neg) / len(dist_pred_to_neg)),
            str(sum(sim_pos) / len(sim_pos)),
            str(sum(sim_neg) / len(sim_neg)),
            str(sum(sim_pos_base) / len(sim_pos_base)),
            str(sum(sim_neg_base) / len(sim_neg_base)),
            str(total_correct_diff / num_samples),
            str(total_correct_base / num_samples)]

    logger.info(f"Total correct (diff): {total_correct_diff}")
    logger.info(f"Total correct (base): {total_correct_base}")
    logger.info(f"Number of samples processed: {num_samples}")
    f.write('\t'.join(info) + '\n')
