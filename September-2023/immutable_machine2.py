
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

pprint(init_matrix)
# {'result': {'self': {':function': {'self': {':function': 1.0}},
#                      'accum': {'self': {'result': 1.0}},
#                      'delta': {'update-1': {'result': 1.0}}},
#             'update-1': {':function': {'update-1': {':function': 1.0}}},
#             'update-2': {':function': {'update-2': {':function': 1.0}}},
#             'update-3': {':function': {'update-3': {':function': 1.0}}}}}

pprint(initial_output)
# {'self': {':function': {'accum_add_args': 1.0},
#           'result': {'self': {':function': {'self': {':function': 1.0}},
#                               'accum': {'self': {'result': 1.0}},
#                               'delta': {'update-1': {'result': 1.0}}},
#                      'update-1': {':function': {'update-1': {':function': 1.0}}},
#                      'update-2': {':function': {'update-2': {':function': 1.0}}},
#                      'update-3': {':function': {'update-3': {':function': 1.0}}}}},
#  'update-1': {':function': {'update_1': 1.0},
#               'result': {'self': {'delta': {'update-1': {'result': -1.0},
#                                             'update-2': {'result': 1.0}}}}},
#  'update-2': {':function': {'update_2': 1.0},
#               'result': {'self': {'delta': {'update-2': {'result': -1.0},
#                                             'update-3': {'result': 1.0}}}}},
#  'update-3': {':function': {'update_3': 1.0},
#               'result': {'self': {'delta': {'update-1': {'result': 1.0},
#                                             'update-3': {'result': -1.0}}}}}}

step1 = two_stroke_cycle(initial_output)
pprint(step1)
step2 = two_stroke_cycle(step1['output'])
pprint(step2)
step3 = two_stroke_cycle(step2['output'])
pprint(step3)
step4 = two_stroke_cycle(step3['output'])
pprint(step4)
