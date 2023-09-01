
# numerical gradient for occasional testing of correctness of autograd computations

# thanks to GPT-4 for helping me to think about this: https://chat.openai.com/share/fd9fae68-278d-4fbe-8637-8ff83fd345d1

from jax.tree_util import tree_map_with_path

def num_grad(loss, tree):
    # for JAX paths which are DictKey(key='string') and for our "tree"
    def update_tree_at_path(path, func_leaf):
        if len(path) == 1:
            return {k: func_leaf(v) if k == path[0].key else v for k, v in tree.items()}
        else:
            key, *rest_path = path
            return {k: update_tree_at_path(v, rest_path, func_leaf) if k == key.key else v for k, v in tree.items()}
    # compute one partial derivative for our "loss" (should we parametrize this to compute one-sided ones also?)
    def compute_partial_derivative(path, leaf):
        delta = 0.01
        tree1 = update_tree_at_path(path, lambda v: v+delta)
        tree2 = update_tree_at_path(path, lambda v: v-delta)
        return (loss(tree1) - loss(tree2))/(2*delta)
    return tree_map_with_path(loss, tree)


        

