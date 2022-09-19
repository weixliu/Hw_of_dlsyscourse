#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cmath>
#include <iostream>

namespace py = pybind11;

void matmul(const float *X, float *theta, float *Z,
unsigned start, unsigned z_dimension, unsigned int k, unsigned int n) {
  for(unsigned int i = 0;i < z_dimension;i++) {
    Z[i] = 0;

    auto row = i / k + start;
    auto col = i % k;

    for(unsigned int j = 0;j < n;j++) {
      Z[i] += X[row * n + j] * theta[j * k + col];
    }
  }
}

void normalization_by_row(float *Z, unsigned int batch, unsigned int k) {
  for(unsigned int i = 0;i < batch;i++) {
    float sum = 0;
    for(unsigned int j = 0;j < k;j++) {
      sum += Z[i * k + j];
    }
    for(unsigned int j = 0;j < k;j++) {
      if (sum > 0) {
        Z[i * k + j] /= sum;
      }
    }
  }
}

void debug_array(float *array, int r, int c) {
  //debug
  std::cout<<"debug Z ==>"<<std::endl;
  for(int i = 0;i < r;i++) {
    for(int j = 0;j < c;j++) {
      std::cout<<array[i * c + j]<<" ";
    }
    std::cout<<std::endl;
  }
}

void debug_array(unsigned char *array, int r, int c) {
  //debug
  std::cout<<"debug Iy ==>"<<std::endl;
  for(int i = 0;i < r;i++) {
    for(int j = 0;j < c;j++) {
      std::cout<<(unsigned int)array[i * c + j]<<" ";
    }
    std::cout<<std::endl;
  }
}

void softmax_regression_epoch_cpp(const float *X, const unsigned char *y,
								  float *theta, size_t m, size_t n, size_t k,
								  float lr, size_t batch)
{
    /**
     * A C++ version of the softmax regression epoch code.  This should run a
     * single epoch over the data defined by X and y (and sizes m,n,k), and
     * modify theta in place.  Your function will probably want to allocate
     * (and then delete) some helper arrays to store the logits and gradients.
     *
     * Args:
     *     X (const float *): pointer to X data, of size m*n, stored in row
     *          major (C) format
     *     y (const unsigned char *): pointer to y data, of size m
     *     theta (foat *): pointer to theta data, of size n*k, stored in row
     *          major (C) format
     *     m (size_t): number of exmaples
     *     n (size_t): input dimension
     *     k (size_t): number of classes
     *     lr (float): learning rate / SGD step size
     *     batch (int): SGD minibatch size
     *
     * Returns:
     *     (None)
     */

    /// BEGIN YOUR CODE
    unsigned int z_dimension = batch * k;
    unsigned int g_dimension = n * k;
    float *Z = new float[z_dimension];
    unsigned char *I_y = new unsigned char[z_dimension];
    float *gradient = new float[g_dimension];
    for(unsigned int start = 0 ; start < m ; start += batch) {
      //clean
      for(unsigned int i = 0;i < z_dimension;i++) {
        Z[i] = 0;
        I_y[i] = 0;
      }
      for(unsigned int i = 0;i < g_dimension;i++) {
        gradient[i] = 0;
      }

      matmul(X, theta, Z, start, z_dimension, k, n);
      for(unsigned int i = 0;i < z_dimension;i++) {
        Z[i] = exp(Z[i]);
      }
      normalization_by_row(Z, batch, k);

      //debug_array(Z, batch, k);

      //Iy
      for(unsigned int i = start;i < start + batch;i++) {
        unsigned char y_idx = y[i];
        int I_y_row = i - start;
        I_y[I_y_row * k + y_idx] = 1;
      }

      //debug_array(I_y, batch, k);

      for(unsigned int i = 0;i < z_dimension;i++) {
        Z[i] -= I_y[i];
      }

      for(unsigned int i = 0;i < g_dimension;i++) {
        gradient[i] = 0.0;
        int row = i / k;
        int col = i % k;

        for(unsigned int j = 0;j < batch;j++) {
          gradient[i] += X[(j + start) * n + row] * Z[j * k + col];
        }
      }

      for(unsigned int i = 0;i < g_dimension;i++) {
        gradient[i] /= batch;
      }

      for(unsigned int i = 0;i < n * k;i++) {
        theta[i] -= lr * gradient[i];
      }
    }
    delete [] gradient;
    delete [] I_y;
    delete [] Z;
    /// END YOUR CODE
}


/**
 * This is the pybind11 code that wraps the function above.  It's only role is
 * wrap the function above in a Python module, and you do not need to make any
 * edits to the code
 */
PYBIND11_MODULE(simple_ml_ext, m) {
    m.def("softmax_regression_epoch_cpp",
    	[](py::array_t<float, py::array::c_style> X,
           py::array_t<unsigned char, py::array::c_style> y,
           py::array_t<float, py::array::c_style> theta,
           float lr,
           int batch) {
        softmax_regression_epoch_cpp(
        	static_cast<const float*>(X.request().ptr),
            static_cast<const unsigned char*>(y.request().ptr),
            static_cast<float*>(theta.request().ptr),
            X.request().shape[0],
            X.request().shape[1],
            theta.request().shape[1],
            lr,
            batch
           );
    },
    py::arg("X"), py::arg("y"), py::arg("theta"),
    py::arg("lr"), py::arg("batch"));
}
