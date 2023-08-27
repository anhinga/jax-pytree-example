init_matrix_elements = [matrix_element("self", "accum", "self", "result"),
                        matrix_element("self", "delta", "update-1", "result"),
                        matrix_element("self", ":function", "self", ":function"), 
                        matrix_element("update-1", ":function", "update-1", ":function"), 
                        matrix_element("update-2", ":function", "update-2", ":function"), 
                        matrix_element("update-3", ":function", "update-3", ":function")]

init_matrix = {'result': add_v_values(*init_matrix_elements)}


