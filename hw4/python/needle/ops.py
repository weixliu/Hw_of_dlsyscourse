"""Operatpr table."""
# Global operator table.
from numbers import Number
from typing import Optional, List
from .autograd import NDArray
from .autograd import Op, Tensor, Value, TensorOp
from .autograd import TensorTuple, TensorTupleOp
from . import init
import numpy

from .backend_selection import array_api, NDArray

import time
import logging
import sys
import functools
import needle as ndl

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
        return tuple([out_grad[i] for i in range(len(out_grad))])


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
                in_grad.append(init.zeros_like(value))
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
        return a + numpy.float32(self.scalar)

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
        return a * numpy.float32(self.scalar)

    def gradient(self, out_grad: Tensor, node: Tensor):
        return (out_grad * self.scalar,)


def mul_scalar(a, scalar):
    return MulScalar(scalar)(a)


class PowerScalar(TensorOp):
    """Op raise a tensor to an (integer) power."""

    def __init__(self, scalar):
        self.scalar = scalar
    
    @logMetric(name='PowerScalar_forward')
    def compute(self, a: NDArray) -> NDArray:
        return array_api.power(a, self.scalar)
    
    @logMetric(name='PowerScalar_backward')
    def gradient(self, out_grad, node):
        out_grad = out_grad.data
        first = node.inputs[0].data
        tensor = power_scalar(first, self.scalar - 1.0).data
        return self.scalar * out_grad * tensor


def power_scalar(a, scalar):
    return PowerScalar(scalar)(a)


class EWiseDiv(TensorOp):
    """Op to element-wise divide two nodes."""

    @logMetric(name='EWiseDiv_forward')
    def compute(self, a, b):
        return array_api.divide(a, b)

    @logMetric(name='EWiseDiv_backward')
    def gradient(self, out_grad, node):
        out_grad = out_grad.data
        lhs, rhs = node.inputs
        lhs = lhs.data
        rhs = rhs.data
        lgrad = divide(Tensor.make_const(array_api.ones(rhs.shape)), rhs).data
        rgrad = negate(divide(lhs, power_scalar(rhs, 2))).data
        return (out_grad * lgrad, out_grad * rgrad)


def divide(a, b):
    return EWiseDiv()(a, b)


class DivScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar
    
    @logMetric(name='DivScalar_forward')
    def compute(self, a):
        return a / self.scalar
    
    @logMetric(name='DivScalar_backward')
    def gradient(self, out_grad, node):
        out_grad = out_grad.data
        return out_grad / self.scalar


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
        return self.inner_transpose(a)
    
    @logMetric(name='Transpose_backward')
    def gradient(self, out_grad, node):
        out_grad = out_grad.data
        return out_grad.transpose(self.axes)


def transpose(a, axes=None):
    return Transpose(axes)(a)


class Reshape(TensorOp):
    def __init__(self, shape):
        self.shape = shape
    
    @logMetric(name='Reshape_forward')
    def compute(self, a):
        return array_api.reshape(a, self.shape)
    
    @logMetric(name='Reshape_backward')
    def gradient(self, out_grad, node):
        out_grad = out_grad.data
        return out_grad.reshape(node.inputs[0].shape)


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


def broadcast_to(a, shape):
    return BroadcastTo(shape)(a)


class Summation(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        if (type(axes) == int):
            self.axes = tuple([axes])
        else:
            self.axes = axes

    @logMetric(name='Summation_forward')
    def compute(self, a):
        return array_api.sum(a, axis=self.axes)
    
    @logMetric(name='Summation_backward')
    def gradient(self, out_grad, node):
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
        res = out_grad.reshape(tuple(axes))
        res = res.broadcast_to(dest_shape)
        return res


def summation(a, axes=None):
    return Summation(axes)(a)


class MatMul(TensorOp):
    @logMetric(name='MatMul_forward')
    def compute(self, a, b):
        return array_api.matmul(a, b)
    
    @logMetric(name='MatMul_backward')    
    def gradient(self, out_grad, node):
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


def matmul(a, b):
    return MatMul()(a, b)


class Negate(TensorOp):
    @logMetric(name='Negate_forward')
    def compute(self, a):
        return array_api.negative(a)
    
    @logMetric(name='Negate_backward')
    def gradient(self, out_grad, node):
        out_grad = out_grad.data
        return negate(out_grad)


def negate(a):
    return Negate()(a)


class Log(TensorOp):
    @logMetric(name='Log_forward')
    def compute(self, a):
        return array_api.log(a)

    @logMetric(name='Log_backward')
    def gradient(self, out_grad, node):
        out_grad = out_grad.data
        input_node = node.inputs[0].data
        return divide(out_grad, input_node)


def log(a):
    return Log()(a)


class Exp(TensorOp):
    @logMetric(name='Exp_forward')
    def compute(self, a):
        return array_api.exp(a)

    @logMetric(name='Exp_backward')
    def gradient(self, out_grad, node):
        out_grad = out_grad.data
        return out_grad * node


def exp(a):
    return Exp()(a)


class ReLU(TensorOp):
    @logMetric(name='ReLU_forward')
    def compute(self, a):
        return array_api.maximum(a, 0)

    @logMetric(name='ReLU_backward')
    def gradient(self, out_grad, node):
        out_grad = out_grad.data
        input = node.inputs[0].data
        # TODO here might need to be fixed.
        mask = input.realize_cached_data() > 0
        return out_grad * Tensor.make_const(mask)


def relu(a):
    return ReLU()(a)


class LogSumExp(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        if (type(axes) == int):
            self.axes = tuple([axes])
        else:
            self.axes = axes

    @logMetric(name='LogSumExp_forward')
    def compute(self, Z):
        r_shape = [x for x in Z.shape]
        axes = self.axes
        if (axes == None):
          axes = []
        for i in axes:
          r_shape[i] = 1
        maxz = array_api.max(Z, self.axes)
        if(self.axes == None):
          r_shape=(1,)
        bmaxz = array_api.broadcast_to(maxz.reshape(r_shape), Z.shape)
        minus = Z - bmaxz
        expMinus = array_api.exp(minus)
        expSum = array_api.sum(expMinus, axis=self.axes)
        logExp = array_api.log(expSum)
        return logExp + maxz

    @logMetric(name='LogSumExp_backward')
    def gradient(self, out_grad, node):
        out_grad = out_grad.data
        input = node.inputs[0].data

        Z = input.realize_cached_data()
        r_shape = [x for x in Z.shape]
        axes = self.axes
        if (axes == None):
          axes = []
        for i in axes:
          r_shape[i] = 1
        maxz = array_api.max(Z, axis=self.axes)
        if(self.axes == None):
          r_shape=(1,)
        bmaxz = array_api.broadcast_to(maxz.reshape(r_shape), Z.shape)
        maxzTensor = Tensor.make_const(bmaxz)

        numerator = exp(input - maxzTensor).detach()
        denominator = summation(exp(input - maxzTensor), self.axes).detach()
        reshape_denominator = broadcast_to(denominator.reshape(r_shape), numerator.shape)
        local_grad = (numerator / reshape_denominator)
        reshape_out_grad = broadcast_to(out_grad.reshape(r_shape).detach(), local_grad.shape)
        return local_grad * reshape_out_grad


def logsumexp(a, axes=None):
    return LogSumExp(axes=axes)(a)


class Tanh(TensorOp):
    def compute(self, a):
        return (array_api.exp(a) - array_api.exp(-1 * a)) / (array_api.exp(a) + array_api.exp(-1 * a))

    def gradient(self, out_grad, node):
        out_grad = out_grad.data
        input = node.inputs[0].data
        return out_grad * (1 - ((exp(input) - exp(-input)) / (exp(input) + exp(-input))) ** 2)


def tanh(a):
    return Tanh()(a)


class Stack(TensorOp):
    def __init__(self, axis: int):
        """
        Concatenates a sequence of arrays along a new dimension.
        Parameters:
        axis - dimension to concatenate along
        All arrays need to be of the same size.
        """
        self.axis = axis

    def compute(self, args):
        shape = args[0].shape
        for i in args:
            if (shape != i.shape):
                raise ValueError("shapes are not matched")
        args_len = len(args)
        shape_list = [i for i in shape]
        shape_list.insert(self.axis, args_len)
        new_shape = tuple(shape_list)

        res = NDArray.make(new_shape, device = args[0].device)
        for i in range(len(args)):
            slices = []
            for j in range(len(new_shape)):
                shape_j_len = new_shape[j]
                if j == self.axis:
                    slices.append(slice(i, i+1, 1))
                else:
                    slices.append(slice(0, shape_j_len, 1))
            res.__setitem__(tuple(slices), args[i])
        
        return res

    def gradient(self, out_grad, node):
        out_grad_ndarray = out_grad.cached_data
        res = Split(self.axis).compute(out_grad_ndarray)
        res = [Tensor.make_const(i) for i in res]
        return make_tuple(*res)


def stack(args, axis):
    return Stack(axis)(make_tuple(*args))


class Split(TensorTupleOp):
    def __init__(self, axis: int):
        """
        Splits a tensor along an axis into a tuple of tensors.
        (The "inverse" of Stack)
        Parameters:
        axis - dimension to split
        """
        self.axis = axis

    def compute(self, A):
        shape = A.shape

        res = []
        new_shape = []
        for i in range(len(shape)):
            if self.axis != i:
                new_shape.append(shape[i])
        new_shape = tuple(new_shape)

        for i in range(shape[self.axis]):
            slices = []
            for j in range(len(shape)):
                if j == self.axis:
                    slices.append(slice(i, i+1, 1))
                else:
                    slices.append(slice(0, shape[j], 1))
            # According to the notes, this is a bug
            res.append(A.__getitem__(tuple(slices)).compact().reshape(new_shape))
        return tuple(res)

    def gradient(self, out_grad, node):
        out_grad_ndarrays = tuple([o.cached_data for o in out_grad])
        res = Stack(self.axis).compute(out_grad_ndarrays)
        return Tensor.make_const(res)


def split(a, axis):
    return Split(axis)(a)


class Flip(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION


def flip(a, axes):
    return Flip(axes)(a)



class Dilate(TensorOp):
    def __init__(self, axes: tuple, dilation: int):
        self.axes = axes
        self.dilation = dilation

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION


def dilate(a, axes, dilation):
    return Dilate(axes, dilation)(a)

class UnDilate(TensorOp):
    def __init__(self, axes: tuple, dilation: int):
        self.axes = axes
        self.dilation = dilation

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION


def undilate(a, axes, dilation):
    return UnDilate(axes, dilation)(a)


class Conv(TensorOp):
    def __init__(self, stride: Optional[int] = 1, padding: Optional[int] = 0):
        self.stride = stride
        self.padding = padding

    def compute(self, A, B):
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION


def conv(a, b, stride=1, padding=1):
    return Conv(stride, padding)(a, b)



