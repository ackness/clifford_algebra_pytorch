import functools
from numbers import Number

import numpy as np
import torch

HANDLED_FUNCTIONS = {}


# def implements(torch_function):
#     """Register a torch function override for ScalarTensor"""
#
#     @functools.wraps(torch_function)
#     def decorator(func):
#         HANDLED_FUNCTIONS[torch_function] = func
#         return func
#
#     return decorator
#
#
# def ensure_tensor(data):
#     if isinstance(data, MultiVector):
#         return data.tensor()
#     return torch.as_tensor(data)


def comb(n, k):
    """\
    Returns /n\\
            \\k/

    comb(n, k) --> PyInt
    """

    def fact(n):
        if n == 0:
            return 1
        return np.multiply.reduce(range(1, n + 1))

    return int(fact(n) / (fact(k) * fact(n - k)))


def elements(dims, firstIdx=0):
    """Return a list of tuples representing all 2**dims of blades
    in a dims-dimensional GA.

    elements(dims, firstIdx=0) --> bladeTupList
    """

    indcs = list(range(firstIdx, firstIdx + dims))

    blades = [()]

    for k in range(1, dims + 1):
        # k = grade

        if k == 1:
            for i in indcs:
                blades.append((i,))
            continue

        curBladeX = indcs[:k]

        for i in range(comb(dims, k)):
            if curBladeX[-1] < firstIdx + dims - 1:
                # increment last index
                blades.append(tuple(curBladeX))
                curBladeX[-1] = curBladeX[-1] + 1

            else:
                marker = -2
                tmp = curBladeX[:]  # copy
                tmp.reverse()

                # locate where the steady increase begins
                for j in range(k - 1):
                    if tmp[j] - tmp[j + 1] == 1:
                        marker = marker - 1
                    else:
                        break

                if marker < -k:
                    blades.append(tuple(curBladeX))
                    continue

                # replace
                blades.append(tuple(curBladeX))
                curBladeX[marker:] = list(range(
                    curBladeX[marker] + 1, curBladeX[marker] + 1 - marker))

    return blades


class MultiVector:
    '''
    for now, this just supports Cl(p, q=0)
    '''

    def __init__(self, dim=4, data=None, names=None, device='cpu'):
        if data is None:
            self.device = torch.device(device)
            self._data = torch.zeros(2 ** dim, dtype=torch.float32, device=self.device)
        else:
            self.device = data.device
            self._data = data
        self._dim = dim

        # sig = [+1] * dim
        bladeTupleList = elements(dim, firstIdx=1)
        self.bladeTupleList = list(map(tuple, bladeTupleList))
        self.gradeList = list(map(len, self.bladeTupleList))
        # create names
        if names is None or isinstance(names, str):
            if isinstance(names, str):
                e = names
            else:
                e = 'e'
            self.names = []
            for i in range(self.gaDims):
                if self.gradeList[i] >= 1:
                    self.names.append(e + ''.join(map(str, self.bladeTupleList[i])))
                else:
                    self.names.append('')
        elif len(names) == self.gaDims:
            self.names = names
        else:
            raise ValueError(
                "names list of length %i needs to be of length %i" %
                (len(names), self.gaDims))

    @property
    def dim(self):
        return self._dim

    @property
    def len(self):
        return len(self._data)

    @property
    def gaDims(self):
        return 2 ** self.dim

    @property
    def value(self):
        return self._data

    def __getitem__(self, i):
        return self._data[i]

    def __setitem__(self, i, value):
        self._data[i] = value

    def __delitem__(self, i):
        self._data[i] = 0.

    def __iter__(self):
        for i in range(self.gaDims):
            if self._data[i]:
                yield i

    def __pos__(self):
        x = MultiVector(self.dim, device=self.device)
        x._data = self._data.clone()
        return x

    def __neg__(self):
        x = MultiVector(self.dim, device=self.device)
        x._data = - self._data
        return x

    def __eq__(self, other):
        return torch.eq(self._data, other._data)

    def __len__(self):
        """Return the number of nonzero terms in this multivector."""
        # return len(self._data) - self._data.count(0.0)
        try:
            return len(self._data.nonzero(as_tuple=True)[0])
        except IndexError:
            return 0

    def __add__(self, other):
        if isinstance(other, (Number, torch.Tensor)):
            x = +self
            x[0] += other
            return x
        elif isinstance(other, MultiVector):
            # "mv have the same gaDims"
            if self.gaDims == other.gaDims:
                x = MultiVector(self.dim, device=self.device)
                x._data = self._data + other._data
                return x
            else:
                m = max(self.dim, other.dim)
                x = MultiVector(m)
                for i in range(self.gaDims):
                    x._data[i] = self._data[i]
                for i in range(other.gaDims):
                    x._data[i] += other._data[i]
                return x
        else:
            raise NotImplemented

    def __radd__(self, other):
        return self + other

    def __sub__(self, other):
        return self + (-other)

    def __rsub__(self, other):
        return - self + other

    def __mul__(self, other):
        if isinstance(other, (Number, torch.Tensor)):
            x = MultiVector(self.dim, device=self.device)
            x._data = self._data * other
            return x
        elif isinstance(other, MultiVector):
            if self.dim == other.dim:
                x = MultiVector(self.dim, device=self.device)
                for i in range(self.gaDims):
                    for j in range(other.gaDims):
                        k, s = MultiVector._blade_combine(i, j)
                        x._data[k] += self._data[i] * other._data[j] * s
                return x
            else:
                m = max(self.dim, other.dim)
                x = MultiVector(m, device=self.device)
                for i in range(self.len):
                    if self._data[i]:
                        for j in range(other.len):
                            if other._data[j]:
                                k, s = MultiVector._blade_combine(i, j)
                                x._data[k] += self._data[i] * other._data[j] * s
                return x
        else:
            raise NotImplemented

    def __rmul__(self, other):
        if isinstance(other, (Number, torch.Tensor)):
            return self * other
        elif isinstance(other, MultiVector):
            return other * self
        else:
            raise NotImplemented

    def __and__(self, other):
        '''
        Find the outer (wedge) product of the two multivectors, or of a multivector
        and a number.
        '''
        x = self * other - other * self
        x._data = x._data / 2
        return x

    def __matmul__(self, other):
        '''
        Find the inner (dot) product of the two multivectors, or of a multivector
        and a number.
        '''
        x = self * other + other * self
        x._data = x._data / 2
        return x

    def __or__(self, other):
        '''
        Find the meet (vee) of the two multivectors, or of a multivector
        and a number.
        '''
        return self.dual() * other

    def __abs__(self):
        '''
        Find the norm of the multivector
        '''
        x = 0.0
        for i in self:
            a = self._data[i]
            x += a * a
        return torch.sqrt(x)

    def __invert__(self):
        x = MultiVector(self.dim, device=self.device)
        for i in range(self.len):
            v = self._data[i]
            if v:
                r = MultiVector._rank(i) % 4
                if r == 2 or r == 3:
                    x[i] = -v
                else:
                    x[i] = v
        return x

    def rank(self):
        r = - torch.Tensor(float('Inf'))
        for i in range(len(self._data)):
            r = max(r, MultiVector._rank(i))
        return r

    def cross_product(self, other):
        return (self & other).dual()

    # def left_inv(self):
    #     try:
    #         x = ~self
    #         s = 1.0 / float(self * x)
    #     except TypeError:
    #         return NotImplemented
    #     x._data = x._data * s
    #     return x
    #
    # def right_inv(self):
    #     try:
    #         x = ~self
    #         s = 1.0 / float(x * self)
    #     except TypeError:
    #         return NotImplemented
    #     x._data = x._data * s
    #     return x

    def __truediv__(self, other):
        '''
        Divide the multivectors, if possible.  Or divide the multivector by the scalar.
        '''
        if isinstance(other, (Number, torch.Tensor)):
            if isinstance(other, torch.Tensor):
                assert len(other.shape) == 0, "the tensor must be scalar"
            x = MultiVector(self.dim, device=self.device)
            x._data = self._data / other
            return x
        if isinstance(other, MultiVector):
            # TODO: still have some bugs! Not use this!
            raise NotImplementedError("Not Implement the mv / mv")

    def dual(self):
        """
        Return the dual of the multivector.
        """
        x = self.I
        return self * x * x * x

    @property
    def I(self):
        """
        Return the standard pseudoscalar of the algebra this multivector is in.
        """
        x = MultiVector(self.dim, device=self.device)
        x._data[-1] = 1.0
        return x

    @staticmethod
    def _rank(a):
        return bin(a).count('1')

    @staticmethod
    def _blade_combine(a, b):
        if a == 0:
            return b, 1
        if b == 0:
            return a, 1
        c = a ^ b
        s = 1
        p = max(a, b)
        d = MultiVector._rank(a)
        e = 1
        while e <= p:
            if e & a:
                d -= 1
            if (d & 1) and (e & b):
                s = -s
            e *= 2
        return c, s

    def __str__(self):
        result = ""
        for i in self:
            coef = self[i].item()
            if result:
                result += ' + '
            result += str(coef)
            bit = 1
            k = 1
            first = True
            while bit <= i:
                if i & bit:
                    if first:
                        result += '*['
                        first = False
                    else:
                        result += ','
                    result += 'e%d' % k
                k += 1
                bit *= 2
            if not first:
                result += ']'
        if not result:
            return '0'
        return result

    def __repr__(self):
        return self.__str__()

    def set_values(self, values, index):
        self._data = torch.zeros(self.gaDims, dtype=torch.float32, device=self.device)
        self._data.index_add_(0, index, values)


if __name__ == '__main__':
    a = MultiVector(4)
    a.set_values(torch.tensor([1., 1, 2, 3, 4]), torch.tensor([0, 1, 2, 3, 4]))
    b = MultiVector(4)
    b.set_values(torch.tensor([1., -1]), torch.tensor([1, 2]))
    c = MultiVector(4)
    c.set_values(torch.ones(16, dtype=torch.float32), torch.arange(16))
    print(a + b)
    print(a - b)
    print(a * b)
    print(a / 2)

