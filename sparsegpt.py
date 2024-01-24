import math
import time
import torch
import torch.nn as nn
import transformers
from quant import *

DEBUG = False
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

"""
定义SparseGPT类，实现SparseGPT算法
"""
class SparseGPT:
    def __init__(self, layer):
        self.layer = layer
        self.dev = self.layer.weight.device
        W = layer.weight.data.clone()
        if isinstance(self.layer, nn.Conv2d):
            W = W.flatten(1)
        if isinstance(self.layer, transformers.Conv1D):
            W = W.t()
        self.rows = W.shape[0]
        self.columns = W.shape[1]
        self.H = torch.zeros((self.columns, self.columns), device=self.dev)
        self.nsamples = 0

    def add_batch(self, inp, out, blocksize=1024):
        if DEBUG:
            self.inp1 = inp
            self.out1 = out
        if len(inp.shape) == 2:
            inp = inp.unsqueeze(0)
        tmp = inp.shape[0]
        if isinstance(self.layer, nn.Linear) or isinstance(self.layer, transformers.Conv1D):
            if len(inp.shape) == 3:
                inp = inp.reshape((-1, inp.shape[-1]))
            inp = inp.t()
        self.H *= self.nsamples / (self.nsamples + tmp)
        self.nsamples += tmp
        inp = math.sqrt(2 / self.nsamples) * inp.float()
        self.H += inp.matmul(inp.t())

    """
    实现快速近似重构、自适应掩码选择等sparsegpt中最核心的算法
    sparsity：稀疏度
    prunen : prunem   n:m
    blocksize：块大小
    """
    def fasterprune(self, sparsity, prunen=0, prunem=0, blocksize=128, percdamp=.01):
        W = self.layer.weight.data.clone()
        if isinstance(self.layer, nn.Conv2d):
            W = W.flatten(1)
        if isinstance(self.layer, transformers.Conv1D):
            W = W.t()
        W = W.float()

        if hasattr(self, 'quantizer'):      # 对W进行量化
            if not self.quantizer.ready():
                self.quantizer.find_params(W, weight=True)

        tick = time.time()

        H = self.H
        del self.H
        # 这行代码首先调用 torch.diag(H) 来提取张量 H 的对角线元素，然后通过 == 0 比较操作，检查这些对角线元素是否等于0。
        # 如果对角线元素等于0，则对应的结果为 True；否则为 False。这样，dead 成为一个布尔型张量，其每个元素表示 H 的对角线上相应元素是否为0。
        dead = torch.diag(H) == 0
        # 这行代码使用 dead 张量作为索引，来修改 H 张量。具体来说，它将 H 的对角线上那些原本为0的元素设置为1。
        # 这是一种常见的技术，用于避免数值计算中的除以零错误或提高数值稳定性。
        H[dead, dead] = 1
        # 这行代码同样使用 dead 张量作为索引，但这次是对 W 张量进行操作。它将 W 中所有与 dead 中为 True 的列对应的元素设置为0。
        # 这意味着如果 H 的某个对角线元素原本是0（即 dead 中相应元素为 True），那么 W 中对应的整列都会被设置为0。
        W[:, dead] = 0

        Losses = torch.zeros(self.rows, device=self.dev)

        # 对于较小的模型，采用dampening ，即在H的对角线元素上添加一个小常数λ（我们总是选择平均对角线值的 1%），似乎足以避免数值问题。
        damp = percdamp * torch.mean(torch.diag(H))
        # 这个张量用于海森矩阵H的对角线元素
        diag = torch.arange(self.columns, device=self.dev)
        # 在H的对角线元素上添加dampening
        H[diag, diag] += damp
        # 这行代码执行Cholesky分解。Cholesky分解是一种将正定矩阵分解为一个下三角矩阵和其转置的上三角矩阵的乘积的方法。
        H = torch.linalg.cholesky(H)
        # 这行代码计算Cholesky分解后的矩阵 H 的逆。torch.cholesky_inverse 是一种高效计算逆矩阵的方法，特别是当矩阵已经通过Cholesky分解时。
        H = torch.cholesky_inverse(H)
        # 这行代码再次执行Cholesky分解，但这次是生成上三角矩阵。参数 upper=True 指定了生成的是上三角矩阵。
        H = torch.linalg.cholesky(H, upper=True)
        # 这行代码将 H 赋值给 Hinv。这里 Hinv 可能代表了 H 的逆矩阵。
        Hinv = H

        mask = None

        for i1 in range(0, self.columns, blocksize):
            i2 = min(i1 + blocksize, self.columns)
            count = i2 - i1

            W1 = W[:, i1 : i2].clone()
            Q1 = torch.zeros_like(W1)
            Err1 = torch.zeros_like(W1)
            Losses1 = torch.zeros_like(W1)
            Hinv1 = Hinv[i1 : i2, i1 : i2]

            if prunen == 0:     # 如果 prunen 等于0，意味着需要进行剪枝操作。
                if mask is not None:
                    mask1 = mask[:, i1 : i2]
                else:
                    # tmp是伪代码中的W
                    tmp = W1 ** 2 / (torch.diag(Hinv1).reshape((1, -1))) ** 2
                    # 根据稀疏度sparsity求出阈值
                    threshold = torch.sort(tmp.flatten())[0][int(tmp.numel() * sparsity)]
                    # 根据阈值，求出W所对应的掩码矩阵mask
                    mask1 = tmp <= threshold
            else:   # 如果 prunen 不等于0，这意味着不需要进行剪枝操作。
                # 这行代码创建了一个与 W1 形状相同的零矩阵，然后检查每个元素是否等于1。
                # 由于零矩阵中的所有元素都是0，这将产生一个全为 False 的掩码矩阵。这意味着没有权重会被剪枝。
                mask1 = torch.zeros_like(W1) == 1

            for i in range(count):
                w = W1[:, i]
                d = Hinv1[i, i]

                if prunen != 0 and i % prunem == 0:
                    tmp = W1[:, i : (i + prunem)] ** 2 / (torch.diag(Hinv1)[i : (i + prunem)].reshape((1, -1))) ** 2
                    mask1.scatter_(1, i + torch.topk(tmp, prunen, dim=1, largest=False)[1], True)

                q = w.clone()
                q[mask1[:, i]] = 0

                if hasattr(self, 'quantizer'):
                    q = quantize(q.unsqueeze(1), self.quantizer.scale, self.quantizer.zero, self.quantizer.maxq).flatten()

                Q1[:, i] = q
                Losses1[:, i] = (w - q) ** 2 / d ** 2

                err1 = (w - q) / d
                W1[:, i:] -= err1.unsqueeze(1).matmul(Hinv1[i, i:].unsqueeze(0))
                Err1[:, i] = err1

            W[:, i1 : i2] = Q1
            Losses += torch.sum(Losses1, 1) / 2

            W[:, i2:] -= Err1.matmul(Hinv[i1:i2, i2:])

            if DEBUG:
                self.layer.weight.data[:, :i2] = W[:, :i2]
                self.layer.weight.data[:, i2:] = W[:, i2:]
                print(torch.sum((self.layer(self.inp1) - self.out1) ** 2))
                print(torch.sum(Losses))

        torch.cuda.synchronize()
        print('time %.2f' % (time.time() - tick))
        print('error', torch.sum(Losses).item())

        if isinstance(self.layer, transformers.Conv1D):
            W = W.t()
        self.layer.weight.data = W.reshape(self.layer.weight.shape).to(self.layer.weight.data.dtype)

        if DEBUG:
            print(torch.sum((self.layer(self.inp1) - self.out1) ** 2))

    def free(self):
        if DEBUG:
            self.inp1 = None
            self.out1 = None
        self.H = None
        torch.cuda.empty_cache()
