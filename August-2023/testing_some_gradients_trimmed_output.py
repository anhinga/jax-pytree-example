# Using the following barbaric trick at the moment:

cat immutable_utils.py immutable_ops.py immutable_engine.py immutable_machine.py > main.py

# and then:

python -i main.py

# Gradients actually work nicely, bugs noted with Zygote.jl handling are not present 

(base) C:\Users\anhin\Desktop\GitHub\jax-pytree-example\August-2023>python -i main.py
No GPU/TPU found, falling back to CPU. (Set TF_CPP_MIN_LOG_LEVEL=0 and rerun for more info.)
{'result': {'self': {':function': {'self': {':function': 1.0}},
                     'accum': {'self': {'result': 1.0}},
                     'delta': {'update-1': {'result': 1.0}}},
            'update-1': {':function': {'update-1': {':function': 1.0}}},
            'update-2': {':function': {'update-2': {':function': 1.0}}},
            'update-3': {':function': {'update-3': {':function': 1.0}}}}}
{'self': {':function': {'accum_add_args': 1.0},
          'result': {'self': {':function': {'self': {':function': 1.0}},
                              'accum': {'self': {'result': 1.0}},
                              'delta': {'update-1': {'result': 1.0}}},
                     'update-1': {':function': {'update-1': {':function': 1.0}}},
                     'update-2': {':function': {'update-2': {':function': 1.0}}},
                     'update-3': {':function': {'update-3': {':function': 1.0}}}}},
 'update-1': {':function': {'update_1': 1.0},
              'result': {'self': {'delta': {'update-1': {'result': -1.0},
                                            'update-2': {'result': 1.0}}}}},
 'update-2': {':function': {'update_2': 1.0},
              'result': {'self': {'delta': {'update-2': {'result': -1.0},
                                            'update-3': {'result': 1.0}}}}},
 'update-3': {':function': {'update_3': 1.0},
              'result': {'self': {'delta': {'update-1': {'result': 1.0},
                                            'update-3': {'result': -1.0}}}}}}
{'input': {'self': {':function': {'accum_add_args': 1.0},
                    'accum': {'self': {':function': {'self': {':function': 1.0}},
                                       'accum': {'self': {'result': 1.0}},
                                       'delta': {'update-1': {'result': 1.0}}},
                              'update-1': {':function': {'update-1': {':function': 1.0}}},
                              'update-2': {':function': {'update-2': {':function': 1.0}}},
                              'update-3': {':function': {'update-3': {':function': 1.0}}}},
                    'delta': {'self': {'delta': {'update-1': {'result': -1.0},
                                                 'update-2': {'result': 1.0}}}}},
           'update-1': {':function': {'update_1': 1.0}},
           'update-2': {':function': {'update_2': 1.0}},
           'update-3': {':function': {'update_3': 1.0}}},
 'output': {'self': {':function': {'accum_add_args': 1.0},
                     'result': {'self': {':function': {'self': {':function': 1.0}},
                                         'accum': {'self': {'result': 1.0}},
                                         'delta': {'update-1': {'result': 0.0},
                                                   'update-2': {'result': 1.0}}},
                                'update-1': {':function': {'update-1': {':function': 1.0}}},
                                'update-2': {':function': {'update-2': {':function': 1.0}}},
                                'update-3': {':function': {'update-3': {':function': 1.0}}}}},
            'update-1': {':function': {'update_1': 1.0},
                         'result': {'self': {'delta': {'update-1': {'result': -1.0},
                                                       'update-2': {'result': 1.0}}}}},
            'update-2': {':function': {'update_2': 1.0},
                         'result': {'self': {'delta': {'update-2': {'result': -1.0},
                                                       'update-3': {'result': 1.0}}}}},
            'update-3': {':function': {'update_3': 1.0},
                         'result': {'self': {'delta': {'update-1': {'result': 1.0},
                                                       'update-3': {'result': -1.0}}}}}}}
{'input': {'self': {':function': {'accum_add_args': 1.0},
                    'accum': {'self': {':function': {'self': {':function': 1.0}},
                                       'accum': {'self': {'result': 1.0}},
                                       'delta': {'update-1': {'result': 0.0},
                                                 'update-2': {'result': 1.0}}},
                              'update-1': {':function': {'update-1': {':function': 1.0}}},
                              'update-2': {':function': {'update-2': {':function': 1.0}}},
                              'update-3': {':function': {'update-3': {':function': 1.0}}}},
                    'delta': {'self': {'delta': {'update-1': {'result': 0.0},
                                                 'update-2': {'result': -1.0},
                                                 'update-3': {'result': 1.0}}}}},
           'update-1': {':function': {'update_1': 1.0}},
           'update-2': {':function': {'update_2': 1.0}},
           'update-3': {':function': {'update_3': 1.0}}},
 'output': {'self': {':function': {'accum_add_args': 1.0},
                     'result': {'self': {':function': {'self': {':function': 1.0}},
                                         'accum': {'self': {'result': 1.0}},
                                         'delta': {'update-1': {'result': 0.0},
                                                   'update-2': {'result': 0.0},
                                                   'update-3': {'result': 1.0}}},
                                'update-1': {':function': {'update-1': {':function': 1.0}}},
                                'update-2': {':function': {'update-2': {':function': 1.0}}},
                                'update-3': {':function': {'update-3': {':function': 1.0}}}}},
            'update-1': {':function': {'update_1': 1.0},
                         'result': {'self': {'delta': {'update-1': {'result': -1.0},
                                                       'update-2': {'result': 1.0}}}}},
            'update-2': {':function': {'update_2': 1.0},
                         'result': {'self': {'delta': {'update-2': {'result': -1.0},
                                                       'update-3': {'result': 1.0}}}}},
            'update-3': {':function': {'update_3': 1.0},
                         'result': {'self': {'delta': {'update-1': {'result': 1.0},
                                                       'update-3': {'result': -1.0}}}}}}}
{'input': {'self': {':function': {'accum_add_args': 1.0},
                    'accum': {'self': {':function': {'self': {':function': 1.0}},
                                       'accum': {'self': {'result': 1.0}},
                                       'delta': {'update-1': {'result': 0.0},
                                                 'update-2': {'result': 0.0},
                                                 'update-3': {'result': 1.0}}},
                              'update-1': {':function': {'update-1': {':function': 1.0}}},
                              'update-2': {':function': {'update-2': {':function': 1.0}}},
                              'update-3': {':function': {'update-3': {':function': 1.0}}}},
                    'delta': {'self': {'delta': {'update-1': {'result': 1.0},
                                                 'update-2': {'result': 0.0},
                                                 'update-3': {'result': -1.0}}}}},
           'update-1': {':function': {'update_1': 1.0}},
           'update-2': {':function': {'update_2': 1.0}},
           'update-3': {':function': {'update_3': 1.0}}},
 'output': {'self': {':function': {'accum_add_args': 1.0},
                     'result': {'self': {':function': {'self': {':function': 1.0}},
                                         'accum': {'self': {'result': 1.0}},
                                         'delta': {'update-1': {'result': 1.0},
                                                   'update-2': {'result': 0.0},
                                                   'update-3': {'result': 0.0}}},
                                'update-1': {':function': {'update-1': {':function': 1.0}}},
                                'update-2': {':function': {'update-2': {':function': 1.0}}},
                                'update-3': {':function': {'update-3': {':function': 1.0}}}}},
            'update-1': {':function': {'update_1': 1.0},
                         'result': {'self': {'delta': {'update-1': {'result': -1.0},
                                                       'update-2': {'result': 1.0}}}}},
            'update-2': {':function': {'update_2': 1.0},
                         'result': {'self': {'delta': {'update-2': {'result': -1.0},
                                                       'update-3': {'result': 1.0}}}}},
            'update-3': {':function': {'update_3': 1.0},
                         'result': {'self': {'delta': {'update-1': {'result': 1.0},
                                                       'update-3': {'result': -1.0}}}}}}}
{'input': {'self': {':function': {'accum_add_args': 1.0},
                    'accum': {'self': {':function': {'self': {':function': 1.0}},
                                       'accum': {'self': {'result': 1.0}},
                                       'delta': {'update-1': {'result': 1.0},
                                                 'update-2': {'result': 0.0},
                                                 'update-3': {'result': 0.0}}},
                              'update-1': {':function': {'update-1': {':function': 1.0}}},
                              'update-2': {':function': {'update-2': {':function': 1.0}}},
                              'update-3': {':function': {'update-3': {':function': 1.0}}}},
                    'delta': {'self': {'delta': {'update-1': {'result': -1.0},
                                                 'update-2': {'result': 1.0},
                                                 'update-3': {'result': 0.0}}}}},
           'update-1': {':function': {'update_1': 1.0}},
           'update-2': {':function': {'update_2': 1.0}},
           'update-3': {':function': {'update_3': 1.0}}},
 'output': {'self': {':function': {'accum_add_args': 1.0},
                     'result': {'self': {':function': {'self': {':function': 1.0}},
                                         'accum': {'self': {'result': 1.0}},
                                         'delta': {'update-1': {'result': 0.0},
                                                   'update-2': {'result': 1.0},
                                                   'update-3': {'result': 0.0}}},
                                'update-1': {':function': {'update-1': {':function': 1.0}}},
                                'update-2': {':function': {'update-2': {':function': 1.0}}},
                                'update-3': {':function': {'update-3': {':function': 1.0}}}}},
            'update-1': {':function': {'update_1': 1.0},
                         'result': {'self': {'delta': {'update-1': {'result': -1.0},
                                                       'update-2': {'result': 1.0}}}}},
            'update-2': {':function': {'update_2': 1.0},
                         'result': {'self': {'delta': {'update-2': {'result': -1.0},
                                                       'update-3': {'result': 1.0}}}}},
            'update-3': {':function': {'update_3': 1.0},
                         'result': {'self': {'delta': {'update-1': {'result': 1.0},
                                                       'update-3': {'result': -1.0}}}}}}}
>>> step4
{'input': {'update-2': {':function': {'update_2': 1.0}}, 'self': {':function': {'accum_add_args': 1.0}, 'delta': {'self': {'delta': {'update-2': {'result': 1.0}, 'update-3': {'result': 0.0}, 'update-1': {'result': -1.0}}}}, 'accum': {'update-2': {':function': {'update-2': {':function': 1.0}}}, 'self': {':function': {'self': {':function': 1.0}}, 'delta': {'update-2': {'result': 0.0}, 'update-3': {'result': 0.0}, 'update-1': {'result': 1.0}}, 'accum': {'self': {'result': 1.0}}}, 'update-1': {':function': {'update-1': {':function': 1.0}}}, 'update-3': {':function': {'update-3': {':function': 1.0}}}}}, 'update-1': {':function': {'update_1': 1.0}}, 'update-3': {':function': {'update_3': 1.0}}}, 'output': {'update-2': {':function': {'update_2': 1.0}, 'result': {'self': {'delta': {'update-2': {'result': -1.0}, 'update-3': {'result': 1.0}}}}}, 'self': {':function': {'accum_add_args': 1.0}, 'result': {'update-2': {':function': {'update-2': {':function': 1.0}}}, 'self': {':function': {'self': {':function': 1.0}}, 'delta': {'update-2': {'result': 1.0}, 'update-3': {'result': 0.0}, 'update-1': {'result': 0.0}}, 'accum': {'self': {'result': 1.0}}}, 'update-1': {':function': {'update-1': {':function': 1.0}}}, 'update-3': {':function': {'update-3': {':function': 1.0}}}}}, 'update-1': {':function': {'update_1': 1.0}, 'result': {'self': {'delta': {'update-2': {'result': 1.0}, 'update-1': {'result': -1.0}}}}}, 'update-3': {':function': {'update_3': 1.0}, 'result': {'self': {'delta': {'update-3': {'result': -1.0}, 'update-1': {'result': 1.0}}}}}}}
>>> pprint(step4)
{'input': {'self': {':function': {'accum_add_args': 1.0},
                    'accum': {'self': {':function': {'self': {':function': 1.0}},
                                       'accum': {'self': {'result': 1.0}},
                                       'delta': {'update-1': {'result': 1.0},
                                                 'update-2': {'result': 0.0},
                                                 'update-3': {'result': 0.0}}},
                              'update-1': {':function': {'update-1': {':function': 1.0}}},
                              'update-2': {':function': {'update-2': {':function': 1.0}}},
                              'update-3': {':function': {'update-3': {':function': 1.0}}}},
                    'delta': {'self': {'delta': {'update-1': {'result': -1.0},
                                                 'update-2': {'result': 1.0},
                                                 'update-3': {'result': 0.0}}}}},
           'update-1': {':function': {'update_1': 1.0}},
           'update-2': {':function': {'update_2': 1.0}},
           'update-3': {':function': {'update_3': 1.0}}},
 'output': {'self': {':function': {'accum_add_args': 1.0},
                     'result': {'self': {':function': {'self': {':function': 1.0}},
                                         'accum': {'self': {'result': 1.0}},
                                         'delta': {'update-1': {'result': 0.0},
                                                   'update-2': {'result': 1.0},
                                                   'update-3': {'result': 0.0}}},
                                'update-1': {':function': {'update-1': {':function': 1.0}}},
                                'update-2': {':function': {'update-2': {':function': 1.0}}},
                                'update-3': {':function': {'update-3': {':function': 1.0}}}}},
            'update-1': {':function': {'update_1': 1.0},
                         'result': {'self': {'delta': {'update-1': {'result': -1.0},
                                                       'update-2': {'result': 1.0}}}}},
            'update-2': {':function': {'update_2': 1.0},
                         'result': {'self': {'delta': {'update-2': {'result': -1.0},
                                                       'update-3': {'result': 1.0}}}}},
            'update-3': {':function': {'update_3': 1.0},
                         'result': {'self': {'delta': {'update-1': {'result': 1.0},
                                                       'update-3': {'result': -1.0}}}}}}}
>>> pprint(trim_v_value(step4, 0.0))
{'input': {'self': {':function': {'accum_add_args': 1.0},
                    'accum': {'self': {':function': {'self': {':function': 1.0}},
                                       'accum': {'self': {'result': 1.0}},
                                       'delta': {'update-1': {'result': 1.0}}},
                              'update-1': {':function': {'update-1': {':function': 1.0}}},
                              'update-2': {':function': {'update-2': {':function': 1.0}}},
                              'update-3': {':function': {'update-3': {':function': 1.0}}}},
                    'delta': {'self': {'delta': {'update-1': {'result': -1.0},
                                                 'update-2': {'result': 1.0}}}}},
           'update-1': {':function': {'update_1': 1.0}},
           'update-2': {':function': {'update_2': 1.0}},
           'update-3': {':function': {'update_3': 1.0}}},
 'output': {'self': {':function': {'accum_add_args': 1.0},
                     'result': {'self': {':function': {'self': {':function': 1.0}},
                                         'accum': {'self': {'result': 1.0}},
                                         'delta': {'update-2': {'result': 1.0}}},
                                'update-1': {':function': {'update-1': {':function': 1.0}}},
                                'update-2': {':function': {'update-2': {':function': 1.0}}},
                                'update-3': {':function': {'update-3': {':function': 1.0}}}}},
            'update-1': {':function': {'update_1': 1.0},
                         'result': {'self': {'delta': {'update-1': {'result': -1.0},
                                                       'update-2': {'result': 1.0}}}}},
            'update-2': {':function': {'update_2': 1.0},
                         'result': {'self': {'delta': {'update-2': {'result': -1.0},
                                                       'update-3': {'result': 1.0}}}}},
            'update-3': {':function': {'update_3': 1.0},
                         'result': {'self': {'delta': {'update-1': {'result': 1.0},
                                                       'update-3': {'result': -1.0}}}}}}}
>>> def loss(state):
...     return state["output"]["self"]["result"]["self"]["accum"]["self"]["result"]
...
>>> loss(step4)
1.0
>>> trim_v_value(grad(loss(step4), 0.0)
...
KeyboardInterrupt
>>> trim_v_value(grad(loss)(step4), 0.0)
{'output': {'self': {'result': {'self': {'accum': {'self': {'result': Array(1., dtype=float32, weak_type=True)}}}}}}}
>>> def loss2(state):
...     l = state["output"]["self"]["result"]["self"]["accum"]["self"]["result"]
...     return l*l
...
>>> trim_v_value(grad(loss2)(step4), 0.0)
{'output': {'self': {'result': {'self': {'accum': {'self': {'result': Array(2., dtype=float32, weak_type=True)}}}}}}}
>>>
>>> def loss4(state):
...     new_state = two_stroke_cycle(state["output"])
...     l1 = state["output"]["self"]["result"]["self"]["accum"]["self"]["result"]
...     l1 = square(l1)
...     l = new_state["output"]["self"]["result"]["self"]["accum"]["self"]["result"]
...     l = square(l)
...     return l + l1
...
>>>
>>> loss4(step4)
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
  File "<stdin>", line 4, in loss4
NameError: name 'square' is not defined
>>> def square(x):
...     return x*x
...
>>> loss4(step4)
2.0
>>> pprint(trim_v_value(grad(loss4)(step4), 0.0))
{'output': {'self': {':function': {'accum_add_args': Array(2., dtype=float32, weak_type=True)},
                     'result': {'self': {':function': {'self': {':function': Array(2., dtype=float32, weak_type=True)}},
                                         'accum': {'self': {'result': Array(6., dtype=float32, weak_type=True)}}}}}}}
>>>
>>> def loss5(state):
...     #new_state = two_stroke_cycle(state["output"])
...     current_output = state["output"]
...     new_input = apply_v_valued_matrix(current_output["self"]["result"], current_output, 2)
...     l1 = state["output"]["self"]["result"]["self"]["accum"]["self"]["result"]
...     l1 = square(l1)
...     l = new_input["self"]["accum"]["self"]["accum"]["self"]["result"]
...     l = square(l)
...     return l + l1
...
>>> loss5(step4)
2.0
>>> pprint(trim_v_value(grad(loss5)(step4), 0.0))
{'output': {'self': {'result': {'self': {'accum': {'self': {'result': Array(6., dtype=float32, weak_type=True)}}}}}}}
>>>
>>> def loss7(state):
...     #new_state = two_stroke_cycle(state["output"])
...     current_output = state["output"]
...     new_input = apply_v_valued_matrix(current_output["self"]["result"], current_output, 2)
...     l1 = state["output"]["self"]["result"]["self"]["accum"]["self"]["result"]
...     l1 = square(l1)
...     l = new_input["self"]["accum"]["self"]["accum"]["self"]["result"]
...     l = square(l)
...     #new_output = up_movement(new_input)
...     #l2 = new_output["self"]["result"]["self"]["accum"]["self"]["result"]
...     new_self = accum_add_args(new_input["self"])
...     l2 = new_self["result"]["self"]["accum"]["self"]["result"]
...     l2 = square(l2)
...     return l + l1 + l2
...
>>> loss7(step4)
3.0
>>> pprint(trim_v_value(grad(loss7)(step4), 0.0))
{'output': {'self': {'result': {'self': {'accum': {'self': {'result': Array(10., dtype=float32, weak_type=True)}}}}}}}
>>>
