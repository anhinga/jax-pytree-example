### August 2023

I am exploring a possibility to resume my JAX work in parallel with my Julia work.

In particular, I am pondering an option of porting Julia code from

https://github.com/anhinga/late-2022-julia-drafts/tree/main/dmm-port-from-clojure/using-Zygote

to JAX using https://github.com/google/tree-math

---

Work in progress:

[immutable_ops.py](immutable_ops.py) - V-value operations (a nicely compact code, looks good)

[immutable_engine.py](immutable_engine.py) - two-stroke cycle

[immutable_machine.py](immutable_machine.py) - a self-referential machine (works fine)

[testing-some-gradients.py](testing-some-gradients.py) - gradient computations which were broken in mutable code for Zygote.jl work fine here (we should consider creating immutable Julia project of this kind); not a Python script, but *.py-approppriate coloring is making it easier to read

---
---
---

### Auxiliary

PyTree resources:

Working with Pytrees: https://jax.readthedocs.io/en/latest/jax-101/05.1-pytrees.html

Pytrees: https://jax.readthedocs.io/en/latest/pytrees.html

`jax.tree_util` module: https://jax.readthedocs.io/en/latest/jax.tree_util.html

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

---

### Work files

  * [immutable_ops.py](immutable_ops.py) - V-value operations (a nicely compact code, looks good)

     * The following auxiliary skethces were used in making it

     * [first_sketches.py](first_sketches.py) - a nice immutable port of `mult-v-value`

     * [GPT-4/second-conversation.md](GPT-4/second-conversation.md) - GPT-4 created what is effectively a nice immutable port of `add-v-values`
   
        * [GPT-4/extras/continue-second-conversation.md](GPT-4/extras/continue-second-conversation.md) - a variadic version

     * [add_v_values.py](add_v_values.py) - old two-argument version and a better variadic version of `add_v_values`
   
     * [mult_mask_and_grad.py](mult_mask_and_grad.py) - `mult-mask-v-value` and a gradient computation
