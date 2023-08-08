from jax.tree_util import tree_map

def mult_v_value(multiplier, v_value):
    return tree_map(lambda x: multiplier*x, v_value)

d = {"x": 8.0, "y": -3.0}
d["a"]={"t":7.0, "v":-9.0}

mult_v_value(5.0, d)
