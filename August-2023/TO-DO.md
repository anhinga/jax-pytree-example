* do normal unit testing (in a low key fashion, no frameworks, but test resulting dicts for equality)

   * probably via `max_norm(x-y)`, that is `max_norm(add_v_values(x, mult_v_value(-1.0, y)))` 

* watch for possible tension between default Python 64-bit float and default JAX 32-bit float
