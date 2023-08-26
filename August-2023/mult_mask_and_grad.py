import tree_math as tm
from jax import grad
from jax.tree_util import tree_map

tree = {'a': 1.0, 'b': {'c': 2.0, 'd': -3.0}, 'e': -4.0}

def sum_tree(x):
    return tm.Vector(x).sum()

sum_tree(tree)

# Array(-4., dtype=float32)

grad_sum_tree = grad(sum_tree)
grad_sum_tree(tree)

# {'a': Array(1., dtype=float32, weak_type=True), 'b': {'c': Array(1., dtype=float32, weak_type=True), 'd': Array(1., dtype=float32, weak_type=True)}, 'e': Array(1., dtype=float32, weak_type=True)}

def mult_v_value(multiplier, v_value):
    return tree_map(lambda x: multiplier*x, v_value)

def mult_mask_v_value(mult_mask, v_value):
    joint_keys = frozenset(mult_mask.keys()) & frozenset(v_value.keys())
    number_keys = {key for key in joint_keys if not isinstance(mult_mask[key], dict) and not isinstance(v_value[key], dict)}
    dict_keys = {key for key in joint_keys if isinstance(mult_mask[key], dict) and isinstance(v_value[key], dict)}
    mult_keys = {key for key in (joint_keys - number_keys - dict_keys) if isinstance(v_value[key], dict)}
    # the remaining case does not generate values and is omitted
    number_key_values = {key: (mult_mask[key]*v_value[key]) for key in number_keys}
    dict_key_values = {key: mult_mask_v_value(mult_mask[key], v_value[key]) for key in dict_keys}
    mult_key_values = {key: mult_v_value(mult_mask[key], v_value[key]) for key in mult_keys} 
    return {**number_key_values, **dict_key_values, **mult_key_values}

mult_mask_v_value(tree, tree)
	
# {'a': 1.0, 'e': 16.0, 'b': {'d': 9.0, 'c': 4.0}}

def sum_square_tree(x):
    return tm.Vector(mult_mask_v_value(x, x)).sum()

sum_square_tree(tree)

# Array(30., dtype=float32)

grad_sum_square_tree = grad(sum_square_tree)

grad_sum_square_tree(tree)

# {'a': Array(2., dtype=float32, weak_type=True), 'b': {'c': Array(4., dtype=float32, weak_type=True), 'd': Array(-6., dtype=float32, weak_type=True)}, 'e': Array(-8., dtype=float32, weak_type=True)}
