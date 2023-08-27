# down-movement

def apply_v_valued_matrix(v_valued_matrix, v_valued_args, level):
    if level==0:
        return mult_mask_lin_comb(v_valued_matrix, v_valued_args)
    return {key: apply_v_valued_matrix(v_valued_matrix[key], v_valued_args, level-1) for key in v_values_matrix.keys()}

# activations

def accum_add_args(all_inputs):
    return {'result': add_v_values(all_inputs['accum'], all_inputs['delta'])}
	
activation_functions = {'accum_add_args': accum_add_args}

# superfluid version of up-movement

def up_movement_helper(input_tree):
    dict_of_functions = input_tree[':function']
    partial_results = [mult_v_value(dict_of_functions[k], activation_functions[k](input_tree)) for k in dict_of_functions.keys()]
    result = add_v_values(*partial_results)
    return {**{':function': dict_of_functions}, **result}


def up_movement(all_input_trees):
    return {neuron_name: up_movement_helper(all_input_trees[neuron_name]) for neuron_name in all_input_trees.keys()}
