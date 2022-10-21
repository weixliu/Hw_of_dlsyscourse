"""The module.
"""
from typing import List, Callable, Any
from needle.autograd import Tensor
from needle import ops
import needle.init as init
import numpy as np

import logging

LOGGER = logging.getLogger(__name__)

class Parameter(Tensor):
    """A special kind of tensor that represents parameters."""


def _unpack_params(value: object) -> List[Tensor]:
    if isinstance(value, Parameter):
        return [value]
    elif isinstance(value, Module):
        return value.parameters()
    elif isinstance(value, dict):
        params = []
        for k, v in value.items():
            params += _unpack_params(v)
        return params
    elif isinstance(value, (list, tuple)):
        params = []
        for v in value:
            params += _unpack_params(v)
        return params
    else:
        return []


def _child_modules(value: object) -> List["Module"]:
    if isinstance(value, Module):
        modules = [value]
        modules.extend(_child_modules(value.__dict__))
        return modules
    if isinstance(value, dict):
        modules = []
        for k, v in value.items():
            modules += _child_modules(v)
        return modules
    elif isinstance(value, (list, tuple)):
        modules = []
        for v in value:
            modules += _child_modules(v)
        return modules
    else:
        return []




class Module:
    def __init__(self):
        self.training = True

    def parameters(self) -> List[Tensor]:
        """Return the list of parameters in the module."""
        parameters = _unpack_params(self.__dict__)
        return parameters

    def _children(self) -> List["Module"]:
        return _child_modules(self.__dict__)

    def eval(self):
        self.training = False
        for m in self._children():
            m.training = False

    def train(self):
        self.training = True
        for m in self._children():
            m.training = True

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class Identity(Module):
    def forward(self, x):
        return x


class Linear(Module):
    def __init__(self, in_features, out_features, bias=True, device=None, dtype="float32"):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        ### BEGIN YOUR SOLUTION
        self.weight = Parameter(init.kaiming_uniform(in_features, out_features, dtype=dtype))
        if (bias):
          self.bias = Parameter(init.kaiming_uniform(out_features, 1, dtype=dtype).reshape((1, out_features)))
        LOGGER.info("self.weight shape {}".format(self.weight.shape))
        LOGGER.info("self.bias shape {}".format(self.bias.shape))
        ### END YOUR SOLUTION

    def forward(self, X: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        mulA = ops.matmul(X, self.weight)
        if self.bias:
          return mulA + ops.broadcast_to(self.bias, mulA.shape)
        else:
          return mulA
        ### END YOUR SOLUTION



class Flatten(Module):
    def forward(self, X):
        ### BEGIN YOUR SOLUTION
        shape = X.shape
        dimension2 = 1
        for d in shape[1:]:
          dimension2 *= d
        return X.reshape((X.shape[0], dimension2))
        ### END YOUR SOLUTION


class ReLU(Module):
    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        return ops.relu(x)
        ### END YOUR SOLUTION


class Sequential(Module):
    def __init__(self, *modules):
        super().__init__()
        self.modules = modules

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        start = x
        for module in self.modules:
          start = module.forward(start)
        return start
        ### END YOUR SOLUTION


class SoftmaxLoss(Module):
    def forward(self, logits: Tensor, y: Tensor):
        ### BEGIN YOUR SOLUTION
        one_hot_y = init.one_hot(logits.shape[1], y)
        Zy = one_hot_y * logits
        temp_log_sum_exp = ops.logsumexp(logits, axes=(1,))
        numerator = (temp_log_sum_exp - Zy.sum(axes=(1,))).sum()
        res = numerator / np.float32(logits.shape[0])
        return res
        ### END YOUR SOLUTION


class BatchNorm1d(Module):
    def __init__(self, dim, eps=1e-5, momentum=0.1, device=None, dtype="float32"):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.momentum = momentum
        ### BEGIN YOUR SOLUTION
        self.running_mean = init.zeros(dim, dtype=dtype)
        self.running_var = init.ones(dim, dtype=dtype)
        self.weight = Parameter(init.ones(dim, dtype=dtype))
        self.bias = Parameter(init.zeros(dim, dtype=dtype))
        ### END YOUR SOLUTION

    #TODO ensure the dytpe should be float32
    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        if not self.training:
          y = (x - self.running_mean) / ((self.running_var + self.eps)**0.5)
          return self.weight * y + self.bias;

        E_x = (x.sum(axes = (0,)) / x.shape[0])
        t_E_x = E_x.reshape((1, self.dim)).broadcast_to(x.shape)

        V_x = ops.power_scalar(x - t_E_x, 2).sum(axes=(0,)) / x.shape[0]
        t_V_x = V_x.reshape((1, self.dim)).broadcast_to(x.shape)
        
        self.running_mean = ((1 - self.momentum) * self.running_mean + self.momentum * E_x).data
        self.running_var = ((1 - self.momentum) * self.running_var + self.momentum * V_x).data

        ele = (x - t_E_x) / ((t_V_x + self.eps)**0.5)
        bc_weight = self.weight.reshape((1, self.dim)).broadcast_to(ele.shape)
        bc_bias = self.bias.reshape((1, self.dim)).broadcast_to(ele.shape)
        return bc_weight * ele + bc_bias
        ### END YOUR SOLUTION


class LayerNorm1d(Module):
    def __init__(self, dim, eps=1e-5, device=None, dtype="float32"):
        super().__init__()
        self.dim = dim
        self.eps = eps
        ### BEGIN YOUR SOLUTION
        self.weight = Parameter(init.ones(dim, dtype=dtype))
        self.bias = Parameter(init.zeros(dim, dtype=dtype))
        LOGGER.info("The weight:{}".format(self.weight))
        LOGGER.info("The bias:{}".format(self.bias))
        ### END YOUR SOLUTION

    #TODO ensure the dytpe should be float32
    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        e_x = x.sum(axes=(1,)) / self.dim
        b_e_x = e_x.reshape((x.shape[0], 1))
        b_e_x = b_e_x.broadcast_to(x.shape)
        v_x = ((x - b_e_x)**2).sum(axes=(1,)) / self.dim
        b_v_x = v_x.reshape((x.shape[0], 1))
        b_v_x = b_v_x.broadcast_to(x.shape)
        denominator = (b_v_x + self.eps)**0.5
        numerator = x - b_e_x
        ele = numerator/denominator
        res = self.weight.reshape((1, self.dim)).broadcast_to(ele.shape) * ele + self.bias.reshape((1, self.dim)).broadcast_to(ele.shape)
        return res
        ### END YOUR SOLUTION


class Dropout(Module):
    def __init__(self, p = 0.5):
        super().__init__()
        self.p = p

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        if not self.training:
          return x
        random_tensor = init.randb(*x.shape, p = (1 - self.p)) / (1-self.p)
        return x * (random_tensor.data)
        ### END YOUR SOLUTION


class Residual(Module):
    def __init__(self, fn: Module):
        super().__init__()
        self.fn = fn

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        return self.fn.forward(x) + x
        ### END YOUR SOLUTION



