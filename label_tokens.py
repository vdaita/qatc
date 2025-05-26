# From LLMLingua (Microsoft)
from fire import Fire
import json
import logging
import os
from collections import defaultdict
from datasets import load_dataset
import spacy
from tqdm import tqdm
import torch
from typing import List, Dict, Set
from tokenizer import AutoTokenizer

nlp = spacy.load("en_core_web_sm")
def split_string(input_string: str, ignore_tokens: Set = set([","])):
    doc = nlp(input_string)
    word_list = []
    for word in doc:
        if word.lemma_ not in ignore_tokens:
            word_list.append(word.lemma_)
    return word_list

def is_equal(a: str, b: str):
    return a.lower() == b.lower()

def label_chunk(origin: str, comp: str, window_size: int, verbose: bool = False):
    num_sample += 1
    origin_tokens = split_string(origin)
    comp_tokens = split_string(comp)
    origin_tokens_set = set(origin_tokens)
    for token in origin_tokens:
        origin_tokens_set.add(token.lower())

    num_find = 0
    prev_idx = 0
    back_cnt = 0
    num_origin_tokens = len(origin_tokens)
    labels = [False] * num_origin_tokens
    for token in comp_tokens:
        flag = False
        if token in origin_tokens_set or token.lower() in origin_tokens_set:
            num_find += 1
        for i in range(window_size):
            # look forward
            token_idx = min(prev_idx + i, num_origin_tokens - 1)
            if is_equal(origin_tokens[token_idx], token) and not labels[token_idx]:
                labels[token_idx] = True
                # window do not go too fast
                if token_idx - prev_idx > window_size // 2:
                    prev_idx += window_size // 2
                else:
                    prev_idx = token_idx
                if verbose:
                    print(
                        token,
                        token_idx,
                        prev_idx,
                        origin_tokens[token_idx - 1 : token_idx + 2],
                    )
                flag = True
                break
            # look backward
            token_idx = max(prev_idx - i, 0)
            if is_equal(origin_tokens[token_idx], token) and not labels[token_idx]:
                labels[token_idx] = True
                prev_idx = token_idx
                if verbose:
                    print(
                        token,
                        token_idx,
                        prev_idx,
                        origin_tokens[token_idx - 1 : token_idx + 2],
                    )
                flag = True
                break

    retrieval_tokens = []
    for idx, token in enumerate(origin_tokens):
        if labels[idx]:
            retrieval_tokens.append(token)
    retrieval = " ".join(retrieval_tokens)

    comp_rate = len(comp_tokens) / len(origin_tokens)
    if len(comp_tokens) > 0:
        find_rate = num_find / len(comp_tokens)
    else:
        find_rate = 0.0
    variation_rate = 1 - find_rate
    hitting_rate = num_find / len(origin_tokens)
    matching_rate = sum(labels) / len(labels)
    alignment_gap = hitting_rate - matching_rate

    if alignment_gap > 0.1:
        if verbose:
            print(origin)
            print("-" * 50)
            print(comp)
            print("-" * 50)
            print(retrieval)
            print("-" * 50)
            print(origin_tokens)
            print("-" * 50)
            print(comp_tokens)
            print("-" * 50)
            print(retrieval_tokens)
            print("=" * 50)

            print(
                f"comp rate: {comp_rate}, variation_rate: {variation_rate}, alignment_gap: {alignment_gap}"
            )

    return {
        "labels": labels,
        "origin": origin,
        "comp": comp,
        "retrieval": retrieval,
        "origin_tokens": origin_tokens,
        "comp_rate": comp_rate,
        "variation_rate": variation_rate,
        "hitting_rate": hitting_rate,
        "matching_rate": matching_rate,
        "alignment_gap": alignment_gap,
    }