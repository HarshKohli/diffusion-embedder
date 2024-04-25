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
