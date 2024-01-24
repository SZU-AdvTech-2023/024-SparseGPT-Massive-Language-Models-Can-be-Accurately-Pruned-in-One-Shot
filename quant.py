import numpy as np
import torch
import torch.nn as nn

"""
首先将浮点数x量化为整数表示，然后再将这些量化的整数值转换回接近原始浮点数值的形式
r = s * (q - z)
"""
def quantize(x, scale, zero, maxq):
    # 将x进行量化到整数q
    q = torch.clamp(torch.round(x / scale) + zero, 0, maxq)
    # 然后根据整数q，反量化还原出它所对应的浮点数张量r' = scale * (q - zero)
    return scale * (q - zero)

"""
定义Quantizer类，用于实现量化操作
"""
class Quantizer(nn.Module):
    def __init__(self, shape=1):
        super(Quantizer, self).__init__()
        self.register_buffer('maxq', torch.tensor(0))
        self.register_buffer('scale', torch.zeros(shape))
        self.register_buffer('zero', torch.zeros(shape))

    # 用于配置量化器的不同参数，这些参数控制量化过程的各个方面
    def configure(self, bits, perchannel=False, sym=True,  mse=False, norm=2.4, grid=100, maxshrink=.8, grouprows=1):
        self.maxq = torch.tensor(2 ** bits - 1)
        self.perchannel = perchannel
        self.sym = sym
        self.mse = mse
        self.norm = norm
        self.grid = grid
        self.maxshrink = maxshrink 
        self.grouprows = grouprows

    # 为量化过程找到最佳的缩放因子scale和零点偏移量zero
    def find_params(self, x, weight=False):
        dev = x.device
        self.maxq = self.maxq.to(dev)
        shape = x.shape

        # per-channel量化
        if self.perchannel:
            if weight:  # 权重量化
                # x本来是[out_channels, in_channels, height, width], 然后转化为
                # [out_channels, in_channels * height *width]
                x = x.flatten(1)
                # 表示权重分组，需要进一步重塑张量，将输出通道分为多个组
                if self.grouprows > 1:      # 对权重分组
                    x = x.reshape((x.shape[0] // self.grouprows, -1))
            else:   # 非权重量化（激活量化？）
                if len(shape) == 4:
                    # 如果是四维数据（通常是图像数据），使用permute和flatten调整数据的形状，使得通道维度成为最外层维度（方便per-channel）
                    x = x.permute([1, 0, 2, 3])
                    x = x.flatten(1)
                if len(shape) == 3:
                    # 如果是三维数据，x可能是[batch_size, seq_len, hidden_dim], reshape后变成[batch_size * seq_len, hidden_dim]
                    # 再经过转置.t()后，变成[hidden_dim, batch_size * seq_len]
                    # 这样的操作通常用于将特征维度放在前面，以便于进行特征级别的分析和操作
                    x = x.reshape((-1, shape[-1])).t()
                if len(shape) == 2:
                    # 这样的操作通常用于将特征维度或通道维度放在前面，以便于进行特征级别的分析和操作
                    x = x.t()
        else:
            # 如果不进行per-channel，即对整个张量统一量化，那么简单地将张量x扁平化为一维，并增加一个维度，使其成为二维张量
            x = x.flatten().unsqueeze(0)

        tmp = torch.zeros(x.shape[0], device=dev)
        xmin = torch.minimum(x.min(1)[0], tmp)
        xmax = torch.maximum(x.max(1)[0], tmp)

        # 对称量化
        if self.sym:
            xmax = torch.maximum(torch.abs(xmin), xmax)
            tmp = xmin < 0
            if torch.any(tmp):
                xmin[tmp] = -xmax[tmp]

        # 用于检查检查xmin和xmax是否同时为0
        tmp = (xmin == 0) & (xmax == 0)
        # 如果xmin和xmax同时为0，那么将它们设置为-1和+1。这是为了避免出现零范围的量化区间，因为这会导致量化后的值都是0，从而失去信息。
        xmin[tmp] = -1
        xmax[tmp] = +1

        # 计算量化的比例因子S
        self.scale = (xmax - xmin) / self.maxq

        # 如果是对称量化，意味着量化范围在正负方向上是对称的，zero point通常位于量化范围的中心
        if self.sym:
            # 若是对称量化，这行代码会设置零点为量化范围的中点。比如8位量化，中点是(255+1)/2=128
            self.zero = torch.full_like(self.scale, (self.maxq + 1) / 2)
        else:       # 非对称量化
            self.zero = torch.round(-xmin / self.scale)

        # 启用均方误差mean square error
        if self.mse:
            # 这个张量用于存储每个元素的最佳量化误差
            best = torch.full([x.shape[0]], float('inf'), device=dev)
            for i in range(int(self.maxshrink * self.grid)):
                # 根据网格grid计算比例因子
                p = 1 - i / self.grid
                # 分别计算量化的最小值和最大值
                xmin1 = p * xmin
                xmax1 = p * xmax
                # 计算量化的缩放因子S
                scale1 = (xmax1 - xmin1) / self.maxq
                # 计算量化的零点zero point
                zero1 = torch.round(-xmin1 / scale1) if not self.sym else self.zero
                # 进行线性量化，然后得到还原后的量化张量q（它是浮点数哦）
                q = quantize(x, scale1.unsqueeze(1), zero1.unsqueeze(1), self.maxq)
                # x是原始浮点张量，q是进行线性量化后的浮点张量，这里计算它俩的差值
                q -= x
                # 计算差的绝对值
                q.abs_()
                # 将差的绝对值提升到self.norm指定的幂次
                q.pow_(self.norm)
                # 计算每个样本的量化误差总和
                err = torch.sum(q, 1)
                # 检查当前误差是否小于之前的最佳误差
                tmp = err < best
                # 更新最佳量化参数
                if torch.any(tmp):
                    best[tmp] = err[tmp]
                    self.scale[tmp] = scale1[tmp]
                    self.zero[tmp] = zero1[tmp]

        # 如果不对每个通道单独量化，而是对整个权重或输入进行统一量化
        if not self.perchannel:
            if weight:  # 处理神经网络的权重
                tmp = shape[0]  # 这通常是输出通道的数量
            else:       # 处理输入或其他张量
                tmp = shape[1] if len(shape) != 3 else shape[2]
            # 重复tmp次，这是为了确保量化参数的大小与处理的张量的相关维度匹配。
            # 在不对每个通道单独量化的情况下，这意味着所有通道将使用相同的量化参数。
            self.scale = self.scale.repeat(tmp)
            self.zero = self.zero.repeat(tmp)

        # 这行代码的目的是根据权重的分组情况和形状，调整量化参数scale和zero的形状
        if weight:
            if self.grouprows > 1:  # 这种操作通常用于将量化参数扩展到多个分组
                self.scale = self.scale.unsqueeze(1).repeat(1, self.grouprows)
                self.zero = self.zero.unsqueeze(1).repeat(1, self.grouprows)
            # 创建一个新的形状列表shape, 这个列表的第一个元素是-1(表示自动推断该维度的大小), 其余元素都是1
            # 数量等于原始形状的维度数减1. 这样做的目的是为了重塑scale和zero, 使其与权重张量的形状兼容
            shape = [-1] + [1] * (len(shape) - 1)
            # 重塑scale和zero的形状
            self.scale = self.scale.reshape(shape)
            self.zero = self.zero.reshape(shape)
            return

        if len(shape) == 4:
            self.scale = self.scale.reshape((1, -1, 1, 1))
            self.zero = self.zero.reshape((1, -1, 1, 1))
        if len(shape) == 3:
            self.scale = self.scale.reshape((1, 1, -1))
            self.zero = self.zero.reshape((1, 1, -1)) 
        if len(shape) == 2:
            self.scale = self.scale.unsqueeze(0)
            self.zero = self.zero.unsqueeze(0)

    # 执行线性量化
    def quantize(self, x):
        if self.ready():
            return quantize(x, self.scale, self.zero, self.maxq)
        return x

    # 检查self.maxq是否大于0
    def enabled(self):
        return self.maxq > 0

    # 检查self.scale中的所有元素是否都不为0
    def ready(self):
        return torch.all(self.scale != 0)
