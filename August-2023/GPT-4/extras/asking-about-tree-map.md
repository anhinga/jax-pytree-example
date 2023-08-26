`tree_map` is truly elegant and rather cryptic:

https://jax.readthedocs.io/en/latest/_autosummary/jax.tree_util.tree_map.html

https://jax.readthedocs.io/en/latest/_modules/jax/_src/tree_util.html#tree_map

```python
def tree_map(f: Callable[..., Any],
             tree: Any,
             *rest: Any,
             is_leaf: Callable[[Any], bool] | None = None) -> Any:
  leaves, treedef = tree_flatten(tree, is_leaf)
  all_leaves = [leaves] + [treedef.flatten_up_to(r) for r in rest]
  return treedef.unflatten(f(*xs) for xs in zip(*all_leaves))
```

so, I asked GPT-4 to comment on `g(f(*xs) for xs in zip(*all_leaves))`

https://chat.openai.com/share/c4185499-d41a-4cad-b247-f619023bfef3
