### August 2023

I am exploring a possibility to resume my JAX work in parallel with my Julia work.

In particular, I am pondering an option of porting Julia code from

https://github.com/anhinga/late-2022-julia-drafts/tree/main/dmm-port-from-clojure/using-Zygote

to JAX using https://github.com/google/tree-math

---

I used the newly available ability to install JAX on Windows 10 in CPU-only mode:

`pip install --upgrade "jax[cpu]"`

and reproduced the example from the README of this repository (JAX 0.4.14, tree-math 0.2.0)

`pip install tree-math`

```python
(base) PS C:\Users\anhin> python
Python 3.9.12 (main, Apr  4 2022, 05:22:27) [MSC v.1916 64 bit (AMD64)] :: Anaconda, Inc. on win32
Type "help", "copyright", "credits" or "license" for more information.
>>> from jax.nn import relu
>>> from jax.tree_util import tree_map
>>> from jax import numpy as jnp
>>> import tree_math as tm
>>> from jax import grad
>>> d = {}
>>> d["x"] = jnp.array([3., -4])
No GPU/TPU found, falling back to CPU. (Set TF_CPP_MIN_LOG_LEVEL=0 and rerun for more info.)
>>> d["y"] = jnp.array([8., 0])
>>> d
{'x': Array([ 3., -4.], dtype=float32), 'y': Array([8., 0.], dtype=float32)}
>>> tree_map(relu, d)
{'x': Array([3., 0.], dtype=float32), 'y': Array([8., 0.], dtype=float32)}
>>> def f(x):
...   return tm.Vector(tree_map(relu, x)).sum()
...
>>> f(d)
Array(11., dtype=float32)
>>> grad_f = grad(f)
>>> grad_f(d)
{'x': Array([1., 0.], dtype=float32), 'y': Array([1., 0.], dtype=float32)}
>>> d["deeper"] = {"inner": jnp.array([-7, 13, 0.])}
>>> d
{'x': Array([ 3., -4.], dtype=float32), 'y': Array([8., 0.], dtype=float32), 'deeper': {'inner': Array([-7., 13.,  0.], dtype=float32)}}
>>> tree_map(relu, d)
{'deeper': {'inner': Array([ 0., 13.,  0.], dtype=float32)}, 'x': Array([3., 0.], dtype=float32), 'y': Array([8., 0.], dtype=float32)}
>>> f(d)
Array(24., dtype=float32)
>>> grad_f(d)
{'deeper': {'inner': Array([0., 1., 0.], dtype=float32)}, 'x': Array([1., 0.], dtype=float32), 'y': Array([1., 0.], dtype=float32)}
>>>
```
