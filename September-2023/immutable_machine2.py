
# handcrafted duplicate detector

# we are likely to want to distinguish between hard links and soft links

# we'll need to do this before optimization experiments, but now
# we are only trying to compute gradients, so perhaps we don't need to do that yet

init_matrix_elements = [matrix_element("self", "accum", "self", "result"),
                        matrix_element("timer", "timer", "timer", "timer"),
                        matrix_element("input", "timer", "timer", "timer"),
                        matrix_element("accum", "dict-1", "accum", "dict"),
                        matrix_element("accum", "dict-2", "input", "char"),
                        matrix_element("norm", "dict", "accum", "dict"),
                        matrix_element("compare", "dict-1", "norm", "norm"),
                        matrix_element("compare", "dict-2", "const_1", "const_1"),
                        matrix_element("dot", "dict-1", "accum", "dict"),
                        matrix_element("dot", "dict-2", "eos", "char"),
                        matrix_element("output", "dict-1", "compare", "true"),
                        matrix_element("output", "dict-2", "dot", "dot"),
                        matrix_element("self", ":function", "self", ":function"),
                        matrix_element("timer", ":function", "timer", ":function"),
                        matrix_element("input", ":function", "input", ":function"),
                        matrix_element("accum", ":function", "accum", ":function"),
                        matrix_element("norm", ":function", "norm", ":function"),
                        matrix_element("const_1", ":function", "const_1", ":function"), 
                        matrix_element("eos", ":function", "eos", ":function"),                         
                        matrix_element("compare", ":function", "compare", ":function"), 
                        matrix_element("dot", ":function", "dot", ":function"), 
                        matrix_element("output", ":function", "output", ":function")]

init_matrix = {'result': add_v_values(*init_matrix_elements)}

# let's initialize carefully (this might be excessive)

initial_output = {'self': add_v_values(init_matrix, {':function': {'accum_add_args': 1.0}}),
                  'timer': add_v_values({'timer': {':number': 0.0}}, {':function': {'timer_add_one': 1.0}}),
                  'input': {':function': {'input_dummy': 1.0}},
                  'accum': {':function': {'accum_add_args': 1.0}},
                  'norm': {':function': {'max_norm_dict': 1.0}},
                  'const_1': add_v_values({'const_1': 1.0}, {':function': {'const_1': 1.0}}),
                  'eos': add_v_values({'const_end': 1.0}, {':function': {'const_end': 1.0}}),
                  'compare': {':function': {'compare_scalars': 1.0}},
                  'dot': {':function': {'dot_product': 1.0}},
                  'output': {':function': {'output_dummy': 1.0}}}

from pprint import pprint

from functools import reduce

def square(x):
    return x * x

def one_cycle(state):
    return two_stroke_cycle(state['output'])

def one_iteration(accum, y):
    previous_dict, previous_state = accum
    new_state = one_cycle(previous_state)
    new_dict = {**previous_dict, **{y: new_state}}
    return (new_dict, new_state)

result, last_state = reduce(one_iteration, range(150), ({}, {'input': {}, 'output': initial_output}))

# >>> [key for key in result]
# [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149]
