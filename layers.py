import sys
import torch


def _precalculate_reversion_sign(dim=256):
    r = []
    for i in range(dim):
        a = bin(i).count('1')
        a = a % 4
        if a == 2 or a == 3:
            r.append(-1.)
        else:
            r.append(1.)
    return torch.tensor(r)


def _blade_combine(a, b):
    if a == 0:
        return b, 1
    if b == 0:
        return a, 1
    c = a ^ b
    s = 1
    p = max(a, b)
    # d = MultiVector._rank(a)
    d = bin(a).count('1')
    e = 1
    while e <= p:
        if e & a:
            d -= 1
        if (d & 1) and (e & b):
            s = -s
        e *= 2
    return c, s


def _precalculate_combine(dim=256):
    rs = torch.zeros((dim, dim), dtype=int)
    rk = torch.zeros((dim, dim), dtype=int)
    for i in range(dim):
        for j in range(dim):
            k, s = _blade_combine(i, j)
            rs[i, j] = s
            rk[i, j] = k
    return rs, rk


class CFLiner(Module):
    def __init__(self, in_channels=8, out_channels=4, bias=True):
        '''
        input shape = [B, SHAPE]
        if out_channels = N, the output shape will be [B, N, SHAPE]
        '''
        super(CFLiner, self).__init__()
        self.shape = in_channels
        self.out_channels = out_channels
        self.use_bias = bias
        self.weight = torch.nn.Parameter(torch.randn(out_channels, self.shape))

        if self.use_bias:
            self.bias = torch.nn.Parameter(torch.randn(self.shape))

        self._init_params()

    def _init_params(self):
        torch.nn.init.normal_(self.weight, 0, 0.01)
        if self.use_bias:
            torch.nn.init.zeros_(self.bias)

        # pre-calculate some used tensor
        rs, rk = _precalculate_combine(self.shape)
        signs = _precalculate_reversion_sign(self.shape)
        self.register_buffer('rs', rs)
        self.register_buffer('rk', rk)
        self.register_buffer('signs', signs)

    def A_geoProduct_B(self, A, B):
        # b is weight
        bs, shape = A.shape
        R = torch.zeros((bs, self.out_channels, shape), device=A.device)
        for k in range(self.out_channels):
            for i in range(self.shape):
                for j in range(self.shape):
                    R[:, k, self.rk[i, j]] += A[:, i] * B[k, j] * self.rs[i, j]
        return R

    def forward(self, input):
        b, s = input.shape
        if s != self.shape:
            sys.exit(f'Wrong Input Features. Please use tensor with {self.shape} Input Features')
        output = self.A_geoProduct_B(input, self.weight)
        if self.use_bias:
            output += self.bias
        return output


if __name__ == '__main__':
    bs = 2
    shape = 8
    module = CFLiner(shape, out_channels=4, bias=True)
    input = torch.ones((bs, shape))
    print('\n input \n', input, '\n input shape \n', input.shape)
    output = module(input)
    print('\n output \n', output, '\n output.shape \n',  output.shape)

    # from torch.autograd.gradcheck import gradcheck
    # moduleConv = CFLiner(shape, 3)
    # input = torch.randn((bs, shape), dtype=torch.double, requires_grad=True)
    # test = gradcheck(moduleConv, input, eps=1e-2, atol=1e-2)
    # print("Are the gradients correct: ", test)
