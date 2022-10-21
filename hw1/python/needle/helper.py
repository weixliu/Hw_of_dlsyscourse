import sys
import numpy as array_api
def utils_compute_gradient(f, *args, **kwargs):
  eps = 1e-4
  numerical_grads = [array_api.zeros(a.shape) for a in args]
  for i in range(len(args)):
    for j in range(args[i].realize_cached_data().size):
      args[i].realize_cached_data().flat[j] += eps
      f1 = float(f(*args, **kwargs).numpy().sum())
      args[i].realize_cached_data().flat[j] -= 2 * eps
      f2 = float(f(*args, **kwargs).numpy().sum())
      args[i].realize_cached_data().flat[j] += eps
      numerical_grads[i].flat[j] = (f1 - f2) / (2 * eps)
  print(numerical_grads)
  return None
