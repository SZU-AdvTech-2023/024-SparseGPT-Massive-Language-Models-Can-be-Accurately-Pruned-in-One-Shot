import torch
import torch.nn as nn

DEV = torch.device('cuda:0')

"""
递归搜索模型中所有指定类型的层，并返回一个字典，其中包含了所有指定类型的层及其在模型中的名称。
"""
def find_layers(module, layers=[nn.Conv2d, nn.Linear], name=''):
    if type(module) in layers:
        return {name: module}

    ans = {}
    for name_c, model_c in module.named_children():
        ans.update(find_layers(model_c, name=name + '.' + name_c if name != '' else name_c))

    return ans