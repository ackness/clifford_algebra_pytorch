# clifford_algebra_pytorch
Simple pytorch implement to compute basic Clifford algebra (Geometric algebra) operation.

## Usage

### Use ga.GA

```python
import torch
from ga import GA

ga = GA(3)
x = 3 + ga[1] + 2 * ga[2] - 2 * ga[3] + ga[1, 2] - ga[2, 3] + 2.5 * ga[1, 3] - ga[1, 2, 3]
print(x)
r = x * (~x) * ga.I
print(r)
q = ga.tensor_to_mv(torch.tensor([1, 1, 1, 1]))
print(q)

# output
# 3.0 + 1.0*[e1] + 2.0*[e2] + 1.0*[e1,e2] + -2.0*[e3] + 2.5*[e1,e3] + -1.0*[e2,e3] + -1.0*[e1,e2,e3]
# -15.0*[e1,e2] + -19.0*[e1,e3] + 2.0*[e2,e3] + 27.25*[e1,e2,e3]
# 1 + 1*[e1] + 1*[e2] + 1*[e1,e2]

```

### Use mv.MultiVector

```python
from mv import MultiVector
a = MultiVector(4, device='cpu')
a.set_values(torch.tensor([1., 1, 2, 3, 4]), torch.tensor([0, 1, 2, 3, 4]))
b = MultiVector(4, device='cpu')
b.set_values(torch.tensor([1., -1]), torch.tensor([1, 2]))
c = MultiVector(4, device='cpu')
c.set_values(torch.ones(16, dtype=torch.float32), torch.arange(16))
print(a + b)
print(a - b)
print(a * b)
print(a / 2)

# output
# 1.0 + 2.0*[e1] + 1.0*[e2] + 3.0*[e1,e2] + 4.0*[e3]
# 1.0 + 3.0*[e2] + 3.0*[e1,e2] + 4.0*[e3]
# -1.0 + -2.0*[e1] + -4.0*[e2] + -3.0*[e1,e2] + -4.0*[e1,e3] + 4.0*[e2,e3]
# 0.5 + 0.5*[e1] + 1.0*[e2] + 1.5*[e1,e2] + 2.0*[e3]

```

### A simple Linear layer using geometry product

```python
from layers import CFLiner
bs = 2
shape = 8
module = CFLiner(shape, out_channels=4, bias=True)
input = torch.ones((bs, shape))
print('\n input \n', input, '\n input shape \n', input.shape)
output = module(input)
print('\n output \n', output, '\n output.shape \n',  output.shape)

# output
#  input 
#  tensor([[1., 1., 1., 1., 1., 1., 1., 1.],
#         [1., 1., 1., 1., 1., 1., 1., 1.]]) 
#  input shape 
#  torch.Size([2, 8])

#  output 
#  tensor([[[ 0.0007,  0.0007,  0.0165,  0.0165, -0.0502, -0.0502, -0.0285,
#           -0.0285],
#          [-0.0107, -0.0107, -0.0236, -0.0236, -0.0301, -0.0301,  0.0165,
#            0.0165],
#          [-0.0160, -0.0160,  0.0046,  0.0046, -0.0592, -0.0592,  0.0104,
#            0.0104],
#          [ 0.0799,  0.0799, -0.0691, -0.0691,  0.0050,  0.0050,  0.0024,
#            0.0024]],

#         [[ 0.0007,  0.0007,  0.0165,  0.0165, -0.0502, -0.0502, -0.0285,
#           -0.0285],
#          [-0.0107, -0.0107, -0.0236, -0.0236, -0.0301, -0.0301,  0.0165,
#            0.0165],
#          [-0.0160, -0.0160,  0.0046,  0.0046, -0.0592, -0.0592,  0.0104,
#            0.0104],
#          [ 0.0799,  0.0799, -0.0691, -0.0691,  0.0050,  0.0050,  0.0024,
#            0.0024]]], grad_fn=<AddBackward0>) 
#  output.shape 
#  torch.Size([2, 4, 8])

```

## Other


## Reference

Original vanilla python implement: https://github.com/tdvance/GeometricAlgebra

