
def matrix_element(to_neuron, to_input, from_neuron, from_output, value = 1.0):
    return {to_neuron: {to_input: {from_neuron: {from_output: value}}}}

# activations

# accumulator with two arguments (used everywhere)

def accum_add_args(all_inputs):
    return {'result': add_v_values(getv(all_inputs, 'accum'), getv(all_inputs, 'delta'))}
    
# constant update matrices for the experiment of a self-referential machine
# running waves pattern through its own wave-connectivity matrix_element
# as described in Appendix B.2 of https://arxiv.org/abs/1706.00648 

def update_1(all_inputs):
    return {'result': add_v_values(matrix_element('self', 'delta', 'update-1', 'result', -1.0),
                                   matrix_element('self', 'delta', 'update-2', 'result'))}

def update_2(all_inputs):
    return {'result': add_v_values(matrix_element('self', 'delta', 'update-2', 'result', -1.0),
                                   matrix_element('self', 'delta', 'update-3', 'result'))}

def update_3(all_inputs):
    return {'result': add_v_values(matrix_element('self', 'delta', 'update-3', 'result', -1.0),
                                   matrix_element('self', 'delta', 'update-1', 'result'))}

# activations needed for Section 3 of https://arxiv.org/abs/1606.09470

def max_norm_dict(all_inputs):
    return {'result': max_norm(getv(all_inputs, 'dict'))}

def dot_product(all_inputs):
    result = mult_mask_lin_comb(getv(all_inputs, 'x'), getv(all_inputs, 'y'))
    return  {'result': {':number': 0.0}, 'warning': {':number': 1.0}} if isinstance(result, dict) else {'result': {':number': result}, 'warning': {':number': 0.0}}

def compare_scalars(all_inputs):
    x_dict, y_dict = getv(all_inputs, 'x'), getv(all_inputs, 'y')
    x = getn(x_dict)
    y = getn(y_dict)
    return {'true': {':number': relu(x-y)}, 'false': {':number': relu(y-x)}}

activation_functions = {'accum_add_args': accum_add_args,
                        'update_1': update_1,
                        'update_2': update_2,
                        'update_3': update_3}

