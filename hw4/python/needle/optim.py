"""Optimization module"""
import needle as ndl
import numpy as np
from needle import ops

class Optimizer:
    def __init__(self, params):
        self.params = params

    def step(self):
        raise NotImplementedError()

    def reset_grad(self):
        for p in self.params:
            p.grad = None


class SGD(Optimizer):
    def __init__(self, params, lr=0.01, momentum=0.0, weight_decay=0.0):
        super().__init__(params)
        self.lr = lr
        self.momentum = momentum
        self.u = {}
        self.weight_decay = weight_decay

    def step(self):
        ### BEGIN YOUR SOLUTION
        # print("before sgd optim step:{}".format(np.array(ndl.autograd.TENSOR_COUNTER)))
        length = len(self.params)
        for i in range(length):
          param = self.params[i]
          grad = param.grad.data
          if(self.weight_decay > 0):
            grad = grad + self.weight_decay * param.data
          if(i not in self.u):
            self.u[i] = ((1 - self.momentum) * grad).data
          else:
            self.u[i] = (self.momentum * self.u[i] + (1 - self.momentum) * grad).data
          #param.data = (1 - self.lr * self.weight_decay) * param.data +  (-self.lr * self.u[i])
          updated = -self.lr * self.u[i]
          # if (updated.realize_cached_data().dtype == 'float64'):
          #   updated = ndl.Tensor.make_const(updated.realize_cached_data().astype('float32'))
          if (updated.realize_cached_data().dtype == 'float64'):
              updated.astype('float32')
          param.data = (param + updated).data
        # print("after sgd optim step:{}".format(np.array(ndl.autograd.TENSOR_COUNTER)))
        ### END YOUR SOLUTION

    def clip_grad_norm(self, max_norm=0.25):
        """
        Clips gradient norm of parameters.
        """
        total_norm = np.linalg.norm(np.array([np.linalg.norm(p.grad.detach().numpy()).reshape((1,)) for p in self.params]))
        clip_coef = max_norm / (total_norm + 1e-6)
        clip_coef_clamped = min((np.asscalar(clip_coef), 1.0))
        for p in self.params:
            p.grad = p.grad.detach() * clip_coef_clamped


class Adam(Optimizer):
    def __init__(
        self,
        params,
        lr=0.01,
        beta1=0.9,
        beta2=0.999,
        eps=1e-8,
        weight_decay=0.0,
    ):
        super().__init__(params)
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weight_decay = weight_decay
        self.t = 0
        self.bias_correction = True

        self.m = {}
        self.v = {}

    def step(self):
        ### BEGIN YOUR SOLUTION
        # print("before adm optim step:{}".format(np.array(ndl.autograd.TENSOR_COUNTER)))
        length = len(self.params)
        self.t += 1
        for i in range(length):
          param = self.params[i]
          
          grad = param.grad.data
          if(self.weight_decay > 0):
            grad = grad + self.weight_decay * param.data
          if(i not in self.m):
            self.m[i] = ((1 - self.beta1) * grad).data
          else:
            self.m[i] = (self.beta1 * self.m[i] + (1 - self.beta1) * grad).data
          if(i not in self.v):
            self.v[i] = ((1 - self.beta2) * (grad**2.0)).data
          else:
            self.v[i] = (self.beta2 * self.v[i] + (1 - self.beta2) * (grad**2.0)).data
          
          m_i = self.m[i]
          v_i = self.v[i]
          if(self.bias_correction):
            m_i = self.m[i] / (1.0 - self.beta1**self.t)
            m_i.detach()
            v_i = self.v[i] / (1.0 - self.beta2**self.t)
            v_i.detach()
          
          minus = ops.negate(self.lr * m_i / (v_i**0.5 + self.eps))
          #param.data = (1 - self.lr * self.weight_decay) * param.data - minus
          param.data = (param + minus).data
        # print("after adm optim step:{}".format(np.array(ndl.autograd.TENSOR_COUNTER)))
        ### END YOUR SOLUTION
