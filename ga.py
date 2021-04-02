from numbers import Number
from mv import MultiVector
import math


class GA:
    def __init__(self, n):
        self._dim = n

    @property
    def n(self):
        return self._dim

    def scalar(self, s=0.):
        x = MultiVector(self.n)
        x[0] = s
        return x

    def blade(self, coef, *indices):
        x = self.scalar(coef)
        for i in indices:
            y = MultiVector(self.n)
            y[2 ** (i - 1)] = 1.
            x *= y
        return x

    @property
    def I(self):
        x = MultiVector(self.n)
        x[-1] = 1.
        return x

    def __getitem__(self, index):
        if isinstance(index, tuple):
            return self.blade(1., *index)
        return self.blade(1., index)

    @classmethod
    def tensor_to_mv(cls, t, dim=None):
        if dim:
            shape = dim
        else:
            shape = math.ceil(math.sqrt(len(t)))
        x = MultiVector(dim=shape, data=t)
        return x

    @classmethod
    def mv_to_tensor(cls, mv):
        return mv._data


if __name__ == '__main__':
    import torch

    ga = GA(3)
    x = 3 + ga[1] + 2 * ga[2] - 2 * ga[3] + ga[1, 2] - ga[2, 3] + 2.5 * ga[1, 3] - ga[1, 2, 3]
    print(x)
    r = x * (~x) * ga.I
    print(r)

    q = ga.tensor_to_mv(torch.tensor([1, 1, 1, 1]))
    print(q)
