def add_v_values(a_v_value, b_v_value):
    if not isinstance(a_v_value, dict) and not isinstance(b_v_value, dict): # two numbers
        return a_v_value + b_v_value    
    if isinstance(a_v_value, dict) and isinstance(b_v_value, dict):
        joint_keys = set(a_v_value.keys()) & set(b_v_value.keys())
        a_keys = set(a_v_value.keys()) - joint_keys
        b_keys = set(b_v_value.keys()) - joint_keys
        joint_key_values = {key: add_v_values(a_v_value[key], b_v_value[key]) for key in joint_keys}
        a_key_values = {key: a_v_value[key] for key in a_keys}
        b_key_values = {key: b_v_value[key] for key in b_keys}
        return {**joint_key_values, **a_key_values, **b_key_values}
    if isinstance(a_v_value, dict): # b_v_value is a leaf node
        return add_v_values(a_v_value, {':number': b_v_value})
    if isinstance(b_v_value, dict): # a_v_value is a leaf
        return add_v_values(b_v_value, {':number': a_v_value})

# Example usage:
tree1 = {'a': 1, 'b': {'c': 2, 'd': 3}, 'e': 4}
tree2 = {'a': {'z': 0}, 'b': {'c': 6}, 'f': 7}
tree3 = {'b': 7}

#>>> add_v_values(tree1, tree2)
#{'a': {'z': 0, ':number': 1}, 'b': {'c': 8, 'd': 3}, 'e': 4, 'f': 7}
#>>> add_v_values(tree1, tree3)
#{'b': {'c': 2, 'd': 3, ':number': 7}, 'e': 4, 'a': 1}
#>>> add_v_values(tree2, tree3)
#{'b': {'c': 6, ':number': 7}, 'a': {'z': 0}, 'f': 7}
