# Testing JAX capability of taking gradients with respect to nested dictionaries

Pytrees play a central role in JAX: https://jax.readthedocs.io/en/latest/pytrees.html

Pytrees are a good fit to implement flexible tensors with tree-shaped indices (**V-values**) described in Sections 3 and 5.3 of 
"Dataflow Matrix Machines and V-values: a Bridge between Programs and Neural Nets", https://arxiv.org/abs/1712.07447

JAX is capable of taking gradients with respect to variables accumulated within pytrees
(see e.g. the last section ("Linear regression with Pytrees") of
https://flax.readthedocs.io/en/latest/notebooks/jax_for_the_impatient.html)

Here is a simple test performed on February 22, 2022 (Ubuntu 20.04.3 LTS, Python 3.8.10, JAX 0.3.0, tree-math 0.1.0):

It is convenient to use the https://github.com/google/tree-math/ library.

We take a gradient of sum of `tree_map(relu, x)` with respect to a nested dictionary `x`.

Note that despite the JAX reputation of being "static", we can change the shape of the dictionary `x` on the fly and `grad_f` keeps working correctly.

```python
>>> from jax.nn import relu
>>> from jax.tree_util import tree_map
>>> from jax import numpy as jnp
>>> import tree_math as tm
>>> from jax import grad

>>> d = {}
>>> d["x"] = jnp.array([3., -4])
>>> d["y"] = jnp.array([8., 0])
>>> d
{'x': DeviceArray([ 3., -4.], dtype=float32), 'y': DeviceArray([8., 0.], dtype=float32)}
>>> tree_map(relu, d)
{'x': DeviceArray([3., 0.], dtype=float32), 'y': DeviceArray([8., 0.], dtype=float32)}

>>> def f(x):
...   return tm.Vector(tree_map(relu, x)).sum()
... 
>>> f(d)
DeviceArray(11., dtype=float32)

>>> grad_f = grad(f)
>>> grad_f(d)
{'x': DeviceArray([1., 0.], dtype=float32), 'y': DeviceArray([1., 0.], dtype=float32)}

>>> d["deeper"] = {"inner": jnp.array([-7, 13, 0.])}
>>> d
{'x': DeviceArray([ 3., -4.], dtype=float32), 'y': DeviceArray([8., 0.], dtype=float32), 'deeper': {'inner': DeviceArray([-7., 13.,  0.], dtype=float32)}}
>>> tree_map(relu, d)
{'deeper': {'inner': DeviceArray([ 0., 13.,  0.], dtype=float32)}, 'x': DeviceArray([3., 0.], dtype=float32), 'y': DeviceArray([8., 0.], dtype=float32)}
>>> f(d)
DeviceArray(24., dtype=float32)
>>> grad_f(d)
{'deeper': {'inner': DeviceArray([0., 1., 0.], dtype=float32)}, 'x': DeviceArray([1., 0.], dtype=float32), 'y': DeviceArray([1., 0.], dtype=float32)}
```

