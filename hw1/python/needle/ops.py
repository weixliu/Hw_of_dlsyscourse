"""Operatpr table."""
# Global operator table.
from numbers import Number
from typing import Optional, List
from .autograd import NDArray
from .autograd import Op, Tensor, Value, TensorOp
from .autograd import TensorTuple, TensorTupleOp
import numpy

# NOTE: we will numpy as the array_api
# to backup our computations, this line will change in later homeworks
import numpy as array_api

import time
import logging
import functools

LOGGER = logging.getLogger(__name__)

def logMetric(_func=None, *, name = 'ops'):
  def innerLogMetric(func):
    def wrapper(*args, **kwargs):
      stime = time.perf_counter()
      result = func(*args, **kwargs)
      LOGGER.info('{:.6f}s for {}'.format(time.perf_counter() - stime, name))
      return result
    return wrapper
  if _func is None:
    return innerLogMetric
  else:
    return innerLogMetric(_func)


class MakeTensorTuple(TensorTupleOp):
    def compute(self, *args) -> tuple:
        return tuple(args)

    def gradient(self, out_grad, node):
        assert isinstance(out_grad, TensorTuple)
        return tuple(*[out_grad[i] for i in range(len(out_grad))])


def make_tuple(*args):
    return MakeTensorTuple()(*args)


class TupleGetItem(TensorOp):
    def __init__(self, index):
        self.index = index

    def __call__(self, a: TensorTuple, fold_const=True) -> Value:
        assert isinstance(a, TensorTuple)
        # constant folding
        if fold_const and isinstance(a.op, MakeTensorTuple):
            return a.inputs[self.index]
        return Tensor.make_from_op(self, [a])

    def compute(self, a):
        return a[self.index]

    def gradient(self, out_grad, node):
        index = self.index
        in_grad = []
        for i, value in enumerate(node.inputs[0]):
            if i != index:
                in_grad.append(zeros_like(value))
            else:
                in_grad.append(out_grad)
        return MakeTensorTuple()(*in_grad)


def tuple_get_item(value, index):
    return TupleGetItem(index)(value)


class FusedAddScalars(TensorTupleOp):
    def __init__(self, c0: float, c1: float):
        self.c0 = c0
        self.c1 = c1

    def compute(self, a):
        return a + self.c0, a + self.c1

    def gradient(self, out_grad, node):
        return out_grad[0] + out_grad[1]


def fused_add_scalars(x, c0, c1):
    return FusedAddScalars(c0, c1)(x)


class EWiseAdd(TensorOp):
    def compute(self, a: NDArray, b: NDArray):
        return a + b

    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad, out_grad


def add(a, b):
    return EWiseAdd()(a, b)


class AddScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: NDArray):
        return a + self.scalar

    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad


def add_scalar(a, scalar):
    return AddScalar(scalar)(a)


class EWiseMul(TensorOp):
    def compute(self, a: NDArray, b: NDArray):
        return a * b

    def gradient(self, out_grad: Tensor, node: Tensor):
        lhs, rhs = node.inputs
        return out_grad * rhs, out_grad * lhs


def multiply(a, b):
    return EWiseMul()(a, b)


class MulScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: NDArray):
        return a * self.scalar

    def gradient(self, out_grad: Tensor, node: Tensor):
        return (out_grad * self.scalar,)


def mul_scalar(a, scalar):
    return MulScalar(scalar)(a)


class PowerScalar(TensorOp):
    """Op raise a tensor to an (integer) power."""

    def __init__(self, scalar: int):
        self.scalar = scalar

    def compute(self, a: NDArray) -> NDArray:
        ### BEGIN YOUR SOLUTION
        return array_api.power(a, self.scalar)
        ### END YOUR SOLUTION
    
    @logMetric(name='PowerScalar')
    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        first = node.inputs[0]
        tensor = Tensor.make_const(array_api.power(first, self.scalar - 1))
        return out_grad * self.scalar * tensor
        ### END YOUR SOLUTION


def power_scalar(a, scalar):
    return PowerScalar(scalar)(a)


class EWiseDiv(TensorOp):
    """Op to element-wise divide two nodes."""

    def compute(self, a, b):
        ### BEGIN YOUR SOLUTION
        return array_api.divide(a, b)
        ### END YOUR SOLUTION

    @logMetric(name='EWiseDiv')
    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        lhs, rhs = node.inputs
        lgrad = divide(Tensor.make_const(array_api.ones(rhs.shape)), rhs)
        rgrad = -1 * divide(lhs, power_scalar(rhs, 2))
        return (out_grad * lgrad, out_grad * rgrad)
        ### END YOUR SOLUTION


def divide(a, b):
    return EWiseDiv()(a, b)


class DivScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return a / self.scalar
        ### END YOUR SOLUTION
    
    @logMetric(name='DivScalar')
    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return out_grad / self.scalar
        ### END YOUR SOLUTION


def divide_scalar(a, scalar):
    return DivScalar(scalar)(a)


class Transpose(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def inner_transpose(self, a):
        local_axes = [x for x in range(len(a.shape))]
        last_idx = len(local_axes) - 1
        if (self.axes == None):
          temp = local_axes[last_idx]
          local_axes[last_idx] = local_axes[last_idx - 1]
          local_axes[last_idx - 1] = temp
        else:
          local_axes[self.axes[0]] = self.axes[1]
          local_axes[self.axes[1]] = self.axes[0]
        return array_api.transpose(a, axes=tuple(local_axes))

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return self.inner_transpose(a)
        ### END YOUR SOLUTION
    
    @logMetric(name='Transpose')
    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return out_grad.transpose(self.axes)
        ### END YOUR SOLUTION


def transpose(a, axes=None):
    return Transpose(axes)(a)


class Reshape(TensorOp):
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.reshape(a, self.shape)
        ### END YOUR SOLUTION
    
    #@logMetric(name='Reshape')
    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return out_grad.reshape(node.inputs[0].shape)
        ### END YOUR SOLUTION


def reshape(a, shape):
    return Reshape(shape)(a)


class BroadcastTo(TensorOp):
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a):
        return array_api.broadcast_to(a, self.shape)
    
    @logMetric(name='BroadcastTo')
    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        ### NOT SURE ###
        dest_shape = node.inputs[0].shape
        src_shape = out_grad.shape
        gap = (len(src_shape) - len(dest_shape))
        axes = []
        for i in range(len(src_shape)):
          if(i < gap):
            axes.append(i)
          else:
            if(dest_shape[i - gap] == 1):
              axes.append(i)
        return out_grad.sum(tuple(axes)).reshape(dest_shape)
        ### END YOUR SOLUTION


def broadcast_to(a, shape):
    return BroadcastTo(shape)(a)


class Summation(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.sum(a, axis=self.axes)
        ### END YOUR SOLUTION
    
    @logMetric(name='Summation')
    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        ### NOT SURE ###
        dest_shape = node.inputs[0].shape
        src_shape = out_grad.shape
        axes = []
        if(self.axes != None):
          axes_set = set(self.axes)
          for i in range(len(dest_shape)):
            if(i not in axes_set):
              axes.append(dest_shape[i])
            else:
              axes.append(1)
        else:
          axes = []
        return out_grad.reshape(tuple(axes)).broadcast_to(dest_shape)
        ### END YOUR SOLUTION


def summation(a, axes=None):
    return Summation(axes)(a)


class MatMul(TensorOp):
    def compute(self, a, b):
        ### BEGIN YOUR SOLUTION
        return array_api.matmul(a, b)
        ### END YOUR SOLUTION
    
    @logMetric(name='MatMul')    
    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        lhs, rhs = node.inputs
        
        lout = out_grad.matmul(rhs.transpose())
        rout = lhs.transpose().matmul(out_grad)

        if(len(lout.shape) != len(lhs.shape)):
          lout = lout.sum(tuple([x for x in range(len(lout.shape) - len(lhs.shape))]))
        if(len(rout.shape) != len(rhs.shape)):
          rout = rout.sum(tuple([x for x in range(len(rout.shape) - len(rhs.shape))]))

        return (lout, rout)
        ### END YOUR SOLUTION


def matmul(a, b):
    return MatMul()(a, b)


class Negate(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.negative(a)
        ### END YOUR SOLUTION
    
    @logMetric(name='Negate')
    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return array_api.negative(out_grad)
        ### END YOUR SOLUTION


def negate(a):
    return Negate()(a)


class Log(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.log(a)
        ### END YOUR SOLUTION

    @logMetric(name='Log')
    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        input_node = node.inputs[0]
        return out_grad * divide(Tensor.make_const(array_api.ones(input_node.shape)), input_node)
        ### END YOUR SOLUTION


def log(a):
    return Log()(a)


class Exp(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.exp(a)
        ### END YOUR SOLUTION

    @logMetric(name='Exp')
    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return out_grad * node
        ### END YOUR SOLUTION


def exp(a):
    return Exp()(a)


# TODO
class ReLU(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.maximum(0, a)
        ### END YOUR SOLUTION

    @logMetric(name='ReLU')
    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        input = node.inputs[0]
        mask = (input.realize_cached_data() > 0) * 1
        return out_grad * Tensor.make_const(mask)
        ### END YOUR SOLUTION


def relu(a):
    return ReLU()(a)



class LogSumExp(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, Z):
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION


def logsumexp(a, axes=None):
    return LogSumExp(axes=axes)(a)
