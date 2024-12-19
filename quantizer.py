import torch
import torch.nn as nn
import math


class BlockRounding(torch.autograd.Function):
    @staticmethod
    def forward(self, x, forward_bits, backward_bits, mode, small_block="None", block_dim="B"):
        self.backward_bits = backward_bits
        self.mode = mode
        if forward_bits == -1: return x
        self.small_block = small_block
        self.block_dim = block_dim
        return block_quantize(x, forward_bits, self.mode, small_block=self.small_block, block_dim=self.block_dim)

    @staticmethod
    def backward(self, grad_output):
        if self.needs_input_grad[0]:
            if self.backward_bits != -1:
                grad_input = block_quantize(grad_output, self.backward_bits, self.mode,
                                            small_block=self.small_block, block_dim=self.block_dim)
            else:
                grad_input = grad_output
        return grad_input, None, None, None, None, None, None


class BlockQuantizer(nn.Module):
    def __init__(self, wl_activate, wl_error, mode,
            small_block="None", block_dim="B"):
        super(BlockQuantizer, self).__init__()
        self.wl_activate = wl_activate
        self.wl_error = wl_error
        self.mode = mode
        self.small_block="None"
        self.block_dim="B"

    def forward(self, x):
        return quantize_block(x, self.wl_activate,
                              self.wl_error, self.mode,
                              self.small_block, self.block_dim)


def block_quantize(data, bits, mode, ebit=8, small_block="FC", block_dim="B"):
    with torch.no_grad():
        data_h = data.clone().clamp(min=1e-10)
        data_l = data.clone().clamp(max=-1e-10)
        data = torch.where(data>=0, data_h, data_l)
    assert data.dim() <= 4
    if small_block == "Conv":
        dim_threshold = 2
    elif small_block == "FC":
        dim_threshold = 1
    elif small_block == "None":
        dim_threshold = 4
    else:
        raise ValueError("Invalid small block option {}".format(small_block))

    if data.dim() <= dim_threshold:
        max_entry = torch.max(torch.abs(data)).item()
        if max_entry == 0: return data
        try:
            max_exponent = math.floor(math.log2(max_entry))
            max_exponent = float(min(max(max_exponent, -2 ** (ebit - 1)), 2 ** (ebit - 1) - 1))
        except (OverflowError, ValueError):
            print(f"max_entru: {max_entry:e}")
            print(f'there is inf in data: {torch.isinf(data).any()}')
            print(f'there is nan in data: {torch.isnan(data).any()}')
            raise
    else:
        if block_dim == "B":
            max_entry = torch.max(torch.abs(data.view(data.size(0), -1)), 1)[0]
            max_exponent = torch.floor(torch.log2(max_entry))
            max_exponent = torch.clamp(max_exponent, -2 ** (ebit - 1), 2 ** (ebit - 1) - 1)
            max_exponent = max_exponent.view([data.size(0)] + [1 for _ in range(data.dim() - 1)])
        elif block_dim == "BC":
            max_entry = torch.max(torch.abs(data.view(data.size(0) * data.size(1), -1)), 1)[0]
            max_exponent = torch.floor(torch.log2(max_entry))
            max_exponent = torch.clamp(max_exponent, -2 ** (ebit - 1), 2 ** (ebit - 1) - 1)
            max_exponent = max_exponent.view([data.size(0), data.size(1)] + [1 for _ in range(data.dim() - 2)])
        else:
            raise ValueError("invalid block dim option {}".format(block_dim))
    i_temp = 2 ** (-max_exponent + (bits - 2))
    i = data * i_temp

    if mode == "stochastic":
        add_r_(i)
        i.floor_()

    elif mode == "nearest":
        i.round_()
    i.clamp_(-2 ** (bits - 1), 2 ** (bits - 1) - 1)
    try:
        temp = i * 2 ** (max_exponent - (bits - 2))
    except RuntimeError:
        print(type(i))
        print(block_dim)
        print(type(max_exponent))
        print(data.size())
        print(small_block)
        raise

    return temp


def add_r_(data):
    r = torch.rand_like(data)
    data.add_(r)


quantize_block = BlockRounding.apply