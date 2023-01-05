"""The module.
"""
from typing import List
from needle.autograd import Tensor
from needle import ops
import needle.init as init
import numpy as np
import math


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
        return _unpack_params(self.__dict__)

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
    def __init__(
        self, in_features, out_features, bias=True, device=None, dtype="float32"
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.weight = Parameter(init.kaiming_uniform(in_features, out_features, device=device, dtype=dtype))
        if (bias):
          self.bias = Parameter(init.kaiming_uniform(out_features, 1, device=device, dtype=dtype).reshape((1, out_features)))

    def forward(self, X: Tensor) -> Tensor:
        mulA = ops.matmul(X, self.weight)
        if self.bias:
          return mulA + ops.broadcast_to(self.bias, mulA.shape)
        else:
          return mulA


class Flatten(Module):
    def forward(self, X):
        shape = X.shape
        dimension2 = 1
        for d in shape[1:]:
          dimension2 *= d
        return X.reshape((X.shape[0], dimension2))


class ReLU(Module):
    def forward(self, x: Tensor) -> Tensor:
        return ops.relu(x)

class LeakyReLU(Module):
    def __init__(self, negative_slope=0.01):
      self.negative_slope = negative_slope

    def forward(self, x: Tensor) -> Tensor:
        return ops.leakyrelu(x, self.negative_slope)


class Tanh(Module):
    def forward(self, x: Tensor) -> Tensor:
        return ops.tanh(x)


class Sigmoid(Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        return ops.sigmoid(x)


class Sequential(Module):
    def __init__(self, *modules):
        super().__init__()
        self.modules = modules

    def forward(self, x: Tensor) -> Tensor:
        start = x
        for module in self.modules:
          start = module.forward(start)
        return start


class SoftmaxLoss(Module):
    def forward(self, logits: Tensor, y: Tensor):
        one_hot_y = init.one_hot(logits.shape[1], y, device=logits.device)
        Zy = one_hot_y * logits
        temp_log_sum_exp = ops.logsumexp(logits, axes=(1,))
        numerator = (temp_log_sum_exp - Zy.sum(axes=(1,))).sum()
        res = numerator / np.float32(logits.shape[0])
        return res

class BCELoss(Module):
    def forward(self, p: Tensor, y: Tensor):
        return -1 * ops.summation(y * ops.log(p + 1e-12) \
        + (1 - y) * ops.log(1 - p + 1e-12)) / y.shape[0]

class BatchNorm1d(Module):
    def __init__(self, dim, eps=1e-5, momentum=0.1, device=None, dtype="float32"):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.momentum = momentum
        self.running_mean = init.zeros(dim, device=device, dtype=dtype)
        self.running_var = init.ones(dim, device=device, dtype=dtype)
        self.weight = Parameter(init.ones(dim, device=device, dtype=dtype))
        self.bias = Parameter(init.zeros(dim, device=device, dtype=dtype))

    def forward(self, x: Tensor) -> Tensor:
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


class BatchNorm2d(BatchNorm1d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x: Tensor):
        # nchw -> nhcw -> nhwc
        s = x.shape
        _x = x.transpose((1, 2)).transpose((2, 3)).reshape((s[0] * s[2] * s[3], s[1]))
        y = super().forward(_x).reshape((s[0], s[2], s[3], s[1]))
        return y.transpose((2,3)).transpose((1,2))


class LayerNorm1d(Module):
    def __init__(self, dim, eps=1e-5, device=None, dtype="float32"):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = Parameter(init.ones(dim, device=device, dtype=dtype))
        self.bias = Parameter(init.zeros(dim, device=device, dtype=dtype))

    def forward(self, x: Tensor) -> Tensor:
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

class Dropout(Module):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, x: Tensor) -> Tensor:
        if not self.training:
          return x
        random_tensor = init.randb(*x.shape, p = (1 - self.p), device=x.device, dtype=x.dtype) / (1-self.p)
        return x * (random_tensor.data)


class Residual(Module):
    def __init__(self, fn: Module):
        super().__init__()
        self.fn = fn

    def forward(self, x: Tensor) -> Tensor:
        return self.fn.forward(x) + x

class Conv(Module):
    """
    Multi-channel 2D convolutional layer
    IMPORTANT: Accepts inputs in NCHW format, outputs also in NCHW format
    Only supports padding=same
    No grouped convolution or dilation
    Only supports square kernels
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, bias=True, device=None, dtype="float32"):
        super().__init__()
        if isinstance(kernel_size, tuple):
            kernel_size = kernel_size[0]
        if isinstance(stride, tuple):
            stride = stride[0]
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride

        self.weight = Parameter(init.kaiming_uniform(kernel_size * kernel_size * in_channels, kernel_size * kernel_size * out_channels, \
        (kernel_size, kernel_size, in_channels, out_channels), device = device, dtype = dtype))

        bound = math.pow(kernel_size * kernel_size * in_channels, -1/2)
        self.bias = Parameter(init.rand(out_channels,low=-bound, high=bound, device = device, dtype = dtype)) if bias else None

    def forward(self, x: Tensor) -> Tensor:
        # NCHW -> NHCW -> NHWC
        x = ops.transpose(ops.transpose(x, axes=(1,2)), axes=(2,3))
        out = ops.conv(x, self.weight, stride=self.stride, padding=self.kernel_size // 2)
        out = ops.transpose(ops.transpose(out, axes=(2,3)), axes=(1,2))

        if self.bias:
            bias = ops.broadcast_to(ops.reshape(self.bias, shape=(1,self.out_channels,1,1)), out.shape)
            out += bias
        
        return out


class RNNCell(Module):
    def __init__(self, input_size, hidden_size, bias=True, nonlinearity='tanh', device=None, dtype="float32"):
        """
        Applies an RNN cell with tanh or ReLU nonlinearity.

        Parameters:
        input_size: The number of expected features in the input X
        hidden_size: The number of features in the hidden state h
        bias: If False, then the layer does not use bias weights
        nonlinearity: The non-linearity to use. Can be either 'tanh' or 'relu'.

        Variables:
        W_ih: The learnable input-hidden weights of shape (input_size, hidden_size).
        W_hh: The learnable hidden-hidden weights of shape (hidden_size, hidden_size).
        bias_ih: The learnable input-hidden bias of shape (hidden_size,).
        bias_hh: The learnable hidden-hidden bias of shape (hidden_size,).

        Weights and biases are initialized from U(-sqrt(k), sqrt(k)) where k = 1/hidden_size
        """
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.nonlinearity=nonlinearity

        val = math.sqrt(1 / hidden_size)
        self.W_ih = Parameter(init.rand(input_size, hidden_size, low=-val, high=val, device=device))
        self.W_hh = Parameter(init.rand(hidden_size, hidden_size, low=-val, high=val, device=device))
        self.bias_ih = Parameter(init.rand(hidden_size, low=-val, high=val, device=device))
        self.bias_hh = Parameter(init.rand(hidden_size, low=-val, high=val, device=device))

    def forward(self, X, h=None):
        """
        Inputs:
        X of shape (bs, input_size): Tensor containing input features
        h of shape (bs, hidden_size): Tensor containing the initial hidden state
            for each element in the batch. Defaults to zero if not provided.

        Outputs:
        h' of shape (bs, hidden_size): Tensor contianing the next hidden state
            for each element in the batch.
        """
        Z = X @ self.W_ih
        if (self.bias):
            # print("the biasih shape:{}, and Z shape:{}".format(self.bias_ih.shape, Z.shape))
            Z = Z + self.bias_ih.reshape((1, self.hidden_size)).broadcast_to(Z.shape)
        if (h != None):
            Z = Z + h @ self.W_hh
        if (self.bias):
            # print("the biashh shape:{}, and Z shape:{}".format(self.bias_hh.shape, Z.shape))
            Z = Z + self.bias_hh.reshape((1, self.hidden_size)).broadcast_to(Z.shape)
        if self.nonlinearity == 'relu':
            return ops.relu(Z)
        else:
            return ops.tanh(Z)


class RNN(Module):
    def __init__(self, input_size, hidden_size, num_layers=1, bias=True, nonlinearity='tanh', device=None, dtype="float32"):
        """
        Applies a multi-layer RNN with tanh or ReLU non-linearity to an input sequence.

        Parameters:
        input_size - The number of expected features in the input x
        hidden_size - The number of features in the hidden state h
        num_layers - Number of recurrent layers.
        nonlinearity - The non-linearity to use. Can be either 'tanh' or 'relu'.
        bias - If False, then the layer does not use bias weights.

        Variables:
        rnn_cells[k].W_ih: The learnable input-hidden weights of the k-th layer,
            of shape (input_size, hidden_size) for k=0. Otherwise the shape is
            (hidden_size, hidden_size).
        rnn_cells[k].W_hh: The learnable hidden-hidden weights of the k-th layer,
            of shape (hidden_size, hidden_size).
        rnn_cells[k].bias_ih: The learnable input-hidden bias of the k-th layer,
            of shape (hidden_size,).
        rnn_cells[k].bias_hh: The learnable hidden-hidden bias of the k-th layer,
            of shape (hidden_size,).
        """
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias
        self.nonlinearity = nonlinearity
        self.device = device
        self.dtype = dtype

        self.rnn_cells = []
        for i in range(num_layers):
            self.rnn_cells.append(
                RNNCell(self.input_size if i == 0 else self.hidden_size, self.hidden_size, bias = self.bias, nonlinearity = self.nonlinearity, device = self.device, dtype = self.dtype)
            )

    def forward(self, X, h0=None):
        """
        Inputs:
        X of shape (seq_len, bs, input_size) containing the features of the input sequence.
        h_0 of shape (num_layers, bs, hidden_size) containing the initial
            hidden state for each element in the batch. Defaults to zeros if not provided.

        Outputs
        output of shape (seq_len, bs, hidden_size) containing the output features
            (h_t) from the last layer of the RNN, for each t.
        h_n of shape (num_layers, bs, hidden_size) containing the final hidden state for each element in the batch.
        """
        seq_len, bs, hidden_size = X.shape
        input = ops.split(X, 0)
        if h0 == None:
            h0 = init.zeros(self.num_layers, bs, self.hidden_size, device=self.device, dtype=self.dtype)
        split_h_0 = ops.split(h0, 0)
        output = []
        split_h_x = split_h_0
        for i in range(seq_len):
            input_i = input[i]
            next_h_x = []
            for layer in range(self.num_layers):
                h_x = split_h_x[layer]
                input_i = self.rnn_cells[layer](input_i, h_x)
                next_h_x.append(input_i)
            split_h_x = next_h_x
            output.append(input_i)
        return ops.stack(output, 0), ops.stack(split_h_x, 0)

class LSTMCell(Module):
    def __init__(self, input_size, hidden_size, bias=True, device=None, dtype="float32"):
        """
        A long short-term memory (LSTM) cell.
        Parameters:
        input_size - The number of expected features in the input X
        hidden_size - The number of features in the hidden state h
        bias - If False, then the layer does not use bias weights
        Variables:
        W_ih - The learnable input-hidden weights, of shape (input_size, 4*hidden_size).
        W_hh - The learnable hidden-hidden weights, of shape (hidden_size, 4*hidden_size).
        bias_ih - The learnable input-hidden bias, of shape (4*hidden_size,).
        bias_hh - The learnable hidden-hidden bias, of shape (4*hidden_size,).
        Weights and biases are initialized from U(-sqrt(k), sqrt(k)) where k = 1/hidden_size
        """
        super().__init__()
        ### BEGIN YOUR SOLUTION
        self.dtype = dtype
        self.device = device
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.sigmoid_fn = Sigmoid()
        self.tanh_fn = Tanh()

        init_arg = math.sqrt(1/hidden_size)
        self.W_ih = Parameter(init.rand(input_size, hidden_size * 4, low=-init_arg, high=init_arg, device=device, dtype=dtype))
        self.W_hh = Parameter(init.rand(hidden_size, hidden_size * 4, low=-init_arg, high=init_arg, device=device, dtype=dtype))
        self.bias_ih = Parameter(init.rand(hidden_size * 4, low=-init_arg, high=init_arg, device=device, dtype=dtype)) if bias else None
        self.bias_hh = Parameter(init.rand(hidden_size * 4, low=-init_arg, high=init_arg, device=device, dtype=dtype)) if bias else None
        ### END YOUR SOLUTION


    def forward(self, X, h=None):
        """
        Inputs: X, h
        X of shape (batch, input_size): Tensor containing input features
        h, tuple of (h0, c0), with
            h0 of shape (bs, hidden_size): Tensor containing the initial hidden state
                for each element in the batch. Defaults to zero if not provided.
            c0 of shape (bs, hidden_size): Tensor containing the initial cell state
                for each element in the batch. Defaults to zero if not provided.
        Outputs: (h', c')
        h' of shape (bs, hidden_size): Tensor containing the next hidden state for each
            element in the batch.
        c' of shape (bs, hidden_size): Tensor containing the next cell state for each
            element in the batch.
        """
        ### BEGIN YOUR SOLUTION
        
        batch_size = X.shape[0]
        h0, c0 = h if h else (None, None)
        h_in = h0 or init.zeros(batch_size, self.hidden_size, device=self.device, dtype=self.dtype)
        c_in = c0 or init.zeros(batch_size, self.hidden_size, device=self.device, dtype=self.dtype)

        tmp = (h_in @ self.W_hh) # shape=(batch_size, 4*hidden_size)
        if self.bias_hh:
            tmp += ops.broadcast_to(ops.reshape(self.bias_hh, shape=(1,self.hidden_size * 4)), shape=tmp.shape)
        
        tmp += (X @ self.W_ih) # shape=(batch_size, 4*hidden_size)
        if self.bias_ih:
            tmp += ops.broadcast_to(ops.reshape(self.bias_ih, shape=(1,self.hidden_size * 4)), shape=tmp.shape)
        
        tmp_parts = ops.split(ops.reshape(tmp, (batch_size, 4, self.hidden_size)), axis=1)
        tmp_i = self.sigmoid_fn(tmp_parts[0])
        tmp_f = self.sigmoid_fn(tmp_parts[1])
        tmp_g = self.tanh_fn(tmp_parts[2])
        tmp_o = self.sigmoid_fn(tmp_parts[3])

        c_out = c_in * tmp_f + tmp_i * tmp_g
        h_out = self.tanh_fn(c_out) * tmp_o
        
        return h_out, c_out


class LSTM(Module):
    def __init__(self, input_size, hidden_size, num_layers=1, bias=True, device=None, dtype="float32"):
        super().__init__()
        """
        Applies a multi-layer long short-term memory (LSTM) RNN to an input sequence.
        Parameters:
        input_size - The number of expected features in the input x
        hidden_size - The number of features in the hidden state h
        num_layers - Number of recurrent layers.
        bias - If False, then the layer does not use bias weights.
        Variables:
        lstm_cells[k].W_ih: The learnable input-hidden weights of the k-th layer,
            of shape (input_size, 4*hidden_size) for k=0. Otherwise the shape is
            (hidden_size, 4*hidden_size).
        lstm_cells[k].W_hh: The learnable hidden-hidden weights of the k-th layer,
            of shape (hidden_size, 4*hidden_size).
        lstm_cells[k].bias_ih: The learnable input-hidden bias of the k-th layer,
            of shape (4*hidden_size,).
        lstm_cells[k].bias_hh: The learnable hidden-hidden bias of the k-th layer,
            of shape (4*hidden_size,).
        """
        ### BEGIN YOUR SOLUTION
        self.dtype = dtype
        self.device = device
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm_cells = [
            LSTMCell(
                input_size if layer_idx == 0 else hidden_size, 
                hidden_size, 
                bias, 
                device, 
                dtype
            ) for layer_idx in range(num_layers)]
        ### END YOUR SOLUTION

    def forward(self, X, h=None):
        """
        Inputs: X, h
        X of shape (seq_len, bs, input_size) containing the features of the input sequence.
        h, tuple of (h0, c0) with
            h_0 of shape (num_layers, bs, hidden_size) containing the initial
                hidden state for each element in the batch. Defaults to zeros if not provided.
            c0 of shape (num_layers, bs, hidden_size) containing the initial
                hidden cell state for each element in the batch. Defaults to zeros if not provided.
        Outputs: (output, (h_n, c_n))
        output of shape (seq_len, bs, hidden_size) containing the output features
            (h_t) from the last layer of the LSTM, for each t.
        tuple of (h_n, c_n) with
            h_n of shape (num_layers, bs, hidden_size) containing the final hidden state for each element in the batch.
            h_n of shape (num_layers, bs, hidden_size) containing the final hidden cell state for each element in the batch.
        """
        ### BEGIN YOUR SOLUTION

        batch_size = X.shape[1]
        (H0, C0) = h if h else (None, None)
        H0 = H0 or init.zeros(self.num_layers, batch_size, self.hidden_size, device=self.device, dtype=self.dtype)
        C0 = C0 or init.zeros(self.num_layers, batch_size, self.hidden_size, device=self.device, dtype=self.dtype)
        H0 = list(ops.split(H0, axis=0))
        C0 = list(ops.split(C0, axis=0))

        H_t = H0
        C_t = C0
        output = []
        for t, X_t in enumerate(list(ops.split(X, axis=0))):
            H_next = []
            C_next = []
            for layer_idx, (h_t, c_t) in enumerate(zip(H_t, C_t)):
                h_next, c_next = self.lstm_cells[layer_idx](X_t, (h_t, c_t))
                X_t = h_next
                H_next.append(h_next)
                C_next.append(c_next)
            output.append(h_next)
            H_t = H_next
            C_t = C_next

        H_n = ops.stack(tuple(H_t), axis=0)
        C_n = ops.stack(tuple(C_t), axis=0)
        output = ops.stack(tuple(output), axis=0)
        return output, (H_n, C_n)


class Embedding(Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype="float32"):
        super().__init__()
        """
        Maps one-hot word vectors from a dictionary of fixed size to embeddings.

        Parameters:
        num_embeddings (int) - Size of the dictionary
        embedding_dim (int) - The size of each embedding vector

        Variables:
        weight - The learnable weights of shape (num_embeddings, embedding_dim)
            initialized from N(0, 1).
        """
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.device = device
        self.dtype = dtype
        self.weight = Parameter(init.randn(num_embeddings, embedding_dim, device=device, dtype=dtype))

    def forward(self, x: Tensor) -> Tensor:
        """
        Maps word indices to one-hot vectors, and projects to embedding vectors

        Input:
        x of shape (seq_len, bs)

        Output:
        output of shape (seq_len, bs, embedding_dim)
        """
        seq_len, bs = x.shape
        one_hot = init.one_hot(self.num_embeddings, x.cached_data.flat, device=self.device, dtype=self.dtype)
        res = (one_hot @ self.weight).reshape((seq_len, bs, self.embedding_dim))
        return res

class DiscriminatorNet(Module):
    """
    A three hidden-layer discriminative neural network
    """
    def __init__(self, input_size=784, device=None, dtype="float32"):
        super().__init__()
        self.input_size=input_size
        
        self.model = Sequential(
            Linear(input_size, 200, device=device, dtype=dtype),
            LeakyReLU(0.02),
            LayerNorm1d(200),
            Linear(200, 1, device=device, dtype=dtype),
            Sigmoid()
        )

    def forward(self, x):
        x = self.model(x)
        return x

class GeneratorNet(Module):
    """
    A three hidden-layer generative neural network
    """
    def __init__(self, input_size=100, output_size=784, device=None, dtype="float32"):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size

        self.model = Sequential(
            Linear(input_size, 200, device=device, dtype=dtype),
            LeakyReLU(0.02),
            LayerNorm1d(200),
            Linear(200, output_size, device=device, dtype=dtype),
            Sigmoid(),
        )

    def forward(self, x):
        x = self.model(x)
        return x