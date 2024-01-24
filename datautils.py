import random
import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, LlamaTokenizer


"""
设置随机数种子，确保代码的随机性具有可重复性
"""
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.random.manual_seed(seed)

"""
根据模型名加载其相应的分词器tokenizer
"""
def get_tokenizer(model):
    if "llama" in model.lower():
        tokenizer = LlamaTokenizer.from_pretrained(model, use_fast=False)
        if tokenizer.bos_token_id != 1 or tokenizer.eos_token_id != 2:
            try:
                tokenizer.bos_token_id = 1
                tokenizer.eos_token_id = 2
            except AttributeError:
                pass
    else:
        tokenizer = AutoTokenizer.from_pretrained(model, use_fast=False)
    return tokenizer


"""
加载Wikitext-2数据集
"""
def get_wikitext2(nsamples, seed, seqlen, model, tokenizer):
    train_data = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train')
    test_data = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')

    train_encoding = tokenizer(" ".join(train_data['text']), return_tensors='pt')
    test_encoding = tokenizer("\n\n".join(test_data['text']), return_tensors='pt')

    set_seed(seed)
    train_dataloader = []

    for _ in range(nsamples):
        i = random.randint(0, train_encoding.input_ids.shape[1] - seqlen - 1)
        inp = train_encoding.input_ids[:, i : i + seqlen]
        tar = inp.clone()
        tar[:, :-1] = -100
        train_dataloader.append((inp, tar))

    return train_dataloader, test_encoding


"""
加载ptb数据集
"""
def get_ptb(nsamples, seed, seqlen, model, tokenizer):
    train_data = load_dataset('ptb_text_only', 'penn_treebank', split='train')
    test_data = load_dataset('ptb_text_only', 'penn_treebank', split='test')

    train_encoding = tokenizer(" ".join(train_data['sentence']), return_tensors='pt')
    test_encoding = tokenizer(" ".join(test_data['sentence']), return_tensors='pt')

    set_seed(seed)
    train_dataloader = []

    for _ in range(nsamples):
        i = random.randint(0, train_encoding.input_ids.shape[1] - seqlen - 1)
        inp = train_encoding.input_ids[:, i : i + seqlen]
        tar = inp.clone()
        tar[:, :-1] = -100
        train_dataloader.append((inp, tar))

    return train_dataloader, test_encoding


"""
加载c4数据集
"""
def get_c4(nsamples, seed, seqlen, model, tokenizer):
    train_data = load_dataset('allenai/c4', 'allenai--c4', data_files={'train': 'en/c4-train.00000-of-01024.json.gz'}, split='train')
    val_data = load_dataset('allenai/c4', 'allenai--c4', data_files={'validation': 'en/c4-validation.00000-of-00008.json.gz'}, split='validation')

    set_seed(seed)
    train_dataloader = []

    for _ in range(nsamples):
        while True:
            i = random.randint(0, len(train_data) - 1)
            train_encoding = tokenizer(train_data[i]['text'], return_tensors='pt')
            if train_encoding.input_ids.shape[1] > seqlen:
                break
        i = random.randint(0, train_encoding.input_ids.shape[1] - seqlen - 1)
        inp = train_encoding.input_ids[:, i : i + seqlen]
        tar = inp.clone()
        tar[:, :-1] = -100
        train_dataloader.append((inp, tar))

    val_encoding = tokenizer(' '.join(val_data[ : 1100]['text']), return_tensors='pt')
    val_encoding = val_encoding.input_ids[:, :(256 * seqlen)]

    class TokenizerWrapper:
        def __init__(self, input_ids):
            self.input_ids = input_ids

    val_encoding = TokenizerWrapper(val_encoding)

    return train_dataloader, val_encoding


"""
根据提供的数据集名称name、样本数量nsamples、随机种子seed、序列长度seqlen、模型名称model
来获取相应的数据加载器
"""
def get_loaders(name, nsamples=128, seed=0, seqlen=2048, model=''):
    tokenizer = get_tokenizer(model)
    if 'wikitext2' in name:
        return get_wikitext2(nsamples, seed, seqlen, model, tokenizer)
    if 'ptb' in name:
        return get_ptb(nsamples, seed, seqlen, model, tokenizer)
    if 'c4' in name:
        return get_c4(nsamples, seed, seqlen, model, tokenizer)
