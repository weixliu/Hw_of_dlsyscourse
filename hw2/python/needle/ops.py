"""Operatpr table."""
# Global operator table.
from numbers import Number
from typing import Optional, List
from .autograd import NDArray
from .autograd import Op, Tensor, Value, TensorOp
from .autograd import TensorTuple, TensorTupleOp
import needle as ndl
import numpy

# NOTE: we will numpy as the array_api
# to backup our computations, this line will change in later homeworks
import numpy as array_api

import time
import logging
import sys
import functools

OPS_COUNT = 0

LOGGER = logging.getLogger(__name__)

def logMetric(_func=None, *, name = 'ops'):
  def innerLogMetric(func):
    def wrapper(*args, **kwargs):
      stime = time.perf_counter()
      before = ndl.autograd.TENSOR_COUNTER
      global OPS_COUNT
      OPS_COUNT += 1
      result = func(*args, **kwargs)
      # LOGGER.warn('{:.6f}s for {}'.format(time.perf_counter() - stime, name))
      if name.find("backward") > 0:
        if type(result) == Tensor:
          result = result.detach()
        elif type(result) == tuple:
          result = tuple([x.detach() for x in result])
        else:
          LOGGER.warn("the result type:{}".format(type(result)))
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
    @logMetric(name='EWiseAdd_forward')
    def compute(self, a: NDArray, b: NDArray):
        return a + b

    @logMetric(name='EWiseAdd_backward')
    def gradient(self, out_grad: Tensor, node: Tensor):
        out_grad = out_grad.data
        return out_grad, out_grad


def add(a, b):
    return EWiseAdd()(a, b)


class AddScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar
    @logMetric(name='AddScalar_forward')
    def compute(self, a: NDArray):
        return a + self.scalar

    @logMetric(name='AddScalar_backward')
    def gradient(self, out_grad: Tensor, node: Tensor):
        out_grad = out_grad.data
        return out_grad


def add_scalar(a, scalar):
    return AddScalar(scalar)(a)


class EWiseMul(TensorOp):
    @logMetric(name='EWiseMul_forward')
    def compute(self, a: NDArray, b: NDArray):
        return a * b

    @logMetric(name='EWiseMul_backward')
    def gradient(self, out_grad: Tensor, node: Tensor):
        out_grad = out_grad.data
        lhs, rhs = node.inputs
        lhs = lhs.data
        rhs = rhs.data
        return (out_grad * rhs), (out_grad * lhs)


def multiply(a, b):
    return EWiseMul()(a, b)


class MulScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar
    @logMetric(name='MulScalar_forward')
    def compute(self, a: NDArray):
        return a * self.scalar

    @logMetric(name='MulScalar_backward')
    def gradient(self, out_grad: Tensor, node: Tensor):
        out_grad = out_grad.data
        return out_grad * self.scalar


def mul_scalar(a, scalar):
    return MulScalar(scalar)(a)


class PowerScalar(TensorOp):
    """Op raise a tensor to an (integer) power."""

    def __init__(self, scalar):
        self.scalar = scalar
    
    @logMetric(name='PowerScalar_forward')
    def compute(self, a: NDArray) -> NDArray:
        ### BEGIN YOUR SOLUTION
        # replace power with float_power to pass test case in BatchNorm1D
        return array_api.float_power(a, self.scalar)
        ### END YOUR SOLUTION
    
    @logMetric(name='PowerScalar_backward')
    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        out_grad = out_grad.data
        first = node.inputs[0].data
        tensor = power_scalar(first, self.scalar - 1.0).data
        return self.scalar * out_grad * tensor
        ### END YOUR SOLUTION


def power_scalar(a, scalar):
    return PowerScalar(scalar)(a)


class EWiseDiv(TensorOp):
    """Op to element-wise divide two nodes."""

    @logMetric(name='EWiseDiv_forward')
    def compute(self, a, b):
        ### BEGIN YOUR SOLUTION
        return array_api.divide(a, b)
        ### END YOUR SOLUTION

    @logMetric(name='EWiseDiv_backward')
    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        out_grad = out_grad.data
        lhs, rhs = node.inputs
        lhs = lhs.data
        rhs = rhs.data
        lgrad = divide(Tensor.make_const(array_api.ones(rhs.shape)), rhs).data
        rgrad = negate(divide(lhs, power_scalar(rhs, 2))).data
        return (out_grad * lgrad, out_grad * rgrad)
        ### END YOUR SOLUTION


def divide(a, b):
    return EWiseDiv()(a, b)


class DivScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar
    
    @logMetric(name='DivScalar_forward')
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return a / self.scalar
        ### END YOUR SOLUTION
    
    @logMetric(name='DivScalar_backward')
    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        out_grad = out_grad.data
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

    @logMetric(name='Transpose_forward')
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return self.inner_transpose(a)
        ### END YOUR SOLUTION
    
    @logMetric(name='Transpose_backward')
    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        out_grad = out_grad.data
        return out_grad.transpose(self.axes)
        ### END YOUR SOLUTION


def transpose(a, axes=None):
    return Transpose(axes)(a)


class Reshape(TensorOp):
    def __init__(self, shape):
        self.shape = shape
    
    @logMetric(name='Reshape_forward')
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.reshape(a, self.shape)
        ### END YOUR SOLUTION
    
    @logMetric(name='Reshape_backward')
    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        out_grad = out_grad.data
        return out_grad.reshape(node.inputs[0].shape)
        ### END YOUR SOLUTION


def reshape(a, shape):
    return Reshape(shape)(a)


class BroadcastTo(TensorOp):
    def __init__(self, shape):
        self.shape = shape

    @logMetric(name='BroadcastTo_forward')
    def compute(self, a):
        return array_api.broadcast_to(a, self.shape)
    
    @logMetric(name='BroadcastTo_backward')
    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        ### NOT SURE ###
        out_grad = out_grad.data
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

    @logMetric(name='Summation_forward')
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.sum(a, axis=self.axes)
        ### END YOUR SOLUTION
    
    @logMetric(name='Summation_backward')
    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        ### NOT SURE ###
        out_grad = out_grad.data
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
    @logMetric(name='MatMul_forward')
    def compute(self, a, b):
        ### BEGIN YOUR SOLUTION
        return array_api.matmul(a, b)
        ### END YOUR SOLUTION
    
    @logMetric(name='MatMul_backward')    
    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        out_grad = out_grad.data
        lhs, rhs = node.inputs
        lhs = lhs.data
        rhs = rhs.data
        
        lout = out_grad.matmul(rhs.transpose()).data
        rout = lhs.transpose().matmul(out_grad).data

        if(len(lout.shape) != len(lhs.shape)):
          lout = lout.sum(tuple([x for x in range(len(lout.shape) - len(lhs.shape))]))
        if(len(rout.shape) != len(rhs.shape)):
          rout = rout.sum(tuple([x for x in range(len(rout.shape) - len(rhs.shape))]))

        return (lout, rout)
        ### END YOUR SOLUTION


def matmul(a, b):
    return MatMul()(a, b)


class Negate(TensorOp):
    @logMetric(name='Negate_forward')
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.negative(a, dtype="float32")
        ### END YOUR SOLUTION
    
    @logMetric(name='Negate_backward')
    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        out_grad = out_grad.data
        return negate(out_grad)
        ### END YOUR SOLUTION


def negate(a):
    return Negate()(a)


class Log(TensorOp):
    @logMetric(name='Log_forward')
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.log(a, dtype="float32")
        ### END YOUR SOLUTION

    @logMetric(name='Log_backward')
    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        out_grad = out_grad.data
        input_node = node.inputs[0].data
        return divide(out_grad, input_node)
        ### END YOUR SOLUTION


def log(a):
    return Log()(a)


class Exp(TensorOp):
    @logMetric(name='Exp_forward')
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.exp(a, dtype="float32")
        ### END YOUR SOLUTION

    @logMetric(name='Exp_backward')
    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        out_grad = out_grad.data
        return out_grad * node
        ### END YOUR SOLUTION


def exp(a):
    return Exp()(a)


# TODO
class ReLU(TensorOp):
    @logMetric(name='ReLU_forward')
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.maximum(a, 0, dtype="float32")
        ### END YOUR SOLUTION

    @logMetric(name='ReLU_backward')
    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        out_grad = out_grad.data
        input = node.inputs[0].data
        mask = array_api.where(input.realize_cached_data() > 0, 1.0, 0.0).astype("float32")
        return out_grad * Tensor.make_const(mask)
        ### END YOUR SOLUTION


def relu(a):
    return ReLU()(a)

class LogSumExp(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    @logMetric(name='LogSumExp_forward')
    def compute(self, Z):
        ### BEGIN YOUR SOLUTION
        r_shape = [x for x in Z.shape]
        axes = self.axes
        if (axes == None):
          axes = []
        for i in axes:
          r_shape[i] = 1
        maxz = array_api.amax(Z, axis=self.axes)
        if(self.axes == None):
          r_shape=(1,)
        bmaxz = array_api.broadcast_to(maxz.reshape(r_shape), Z.shape)
        minus = Z - bmaxz
        expMinus = array_api.exp(minus)
        expSum = array_api.sum(expMinus, axis=self.axes)
        logExp = array_api.log(expSum)
        return logExp + maxz
        ### END YOUR SOLUTION

    @logMetric(name='LogSumExp_backward')
    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        ### TODO I can not understand how it works. And just debug it to pass UT
        out_grad = out_grad.data
        LOGGER.info("The out_grad shape {}".format(out_grad.shape))
        input = node.inputs[0].data
        LOGGER.info("The input shape {}".format(input.shape))

        Z = input.realize_cached_data()
        r_shape = [x for x in Z.shape]
        axes = self.axes
        if (axes == None):
          axes = []
        for i in axes:
          r_shape[i] = 1
        LOGGER.info("The r_shape:{}".format(r_shape))
        maxz = array_api.amax(Z, axis=self.axes)
        LOGGER.info("The maxz {}".format(maxz))
        if(self.axes == None):
          r_shape=(1,)
        bmaxz = array_api.broadcast_to(maxz.reshape(r_shape), Z.shape)
        LOGGER.info("The bmaxz {}".format(bmaxz))
        maxzTensor = Tensor.make_const(bmaxz)

        numerator = exp(input - maxzTensor).detach()
        denominator = summation(exp(input - maxzTensor), self.axes).detach()
        reshape_denominator = denominator.reshape(r_shape)
        local_grad = (numerator / reshape_denominator)
        reshape_out_grad = out_grad.reshape(r_shape).detach()
        return local_grad * reshape_out_grad
        ### END YOUR SOLUTION


def logsumexp(a, axes=None):
    return LogSumExp(axes=axes)(a)
