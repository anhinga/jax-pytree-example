def max_norm(v_value): # we'll also have a slightly different activation function of this kind: v_value -> {'result': max_norm(v_value)} 
    return abs(v_value) if not isinstance(v_value, dict) else max([max_norm(v_value[key]) for key in v_value.keys()])

def trim_v_value(v_value, threshold): # keeping paths with max_norm strictly above treshold; keeping scalars
    return v_value if not isinstance(v_value, dict) else {key: trim_v_value(v_value[key], threshold) for key in v_value.keys() if max_norm(v_value[key]) > threshold}
