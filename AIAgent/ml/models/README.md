# ML Models

Model classes for the state-prediction GNN. A model lives here only if it
conforms to the forward contract and is registered.

## Layout

- `NorthernPenguin/` — the registered model (`StateModelEncoder`).
- `modelop/` — model utilities (`filemanager.save_model`).
- `registry.py` — the model registry: the single selection path for model
  classes (name → class, reverse lookup by instance or class).

## Forward contract

`ml/protocols.py` defines the `Model` protocol: the 11-parameter `forward`
signature that `ml/inference.py:infer()` calls, returning a
`(num_states, 1)` tensor. Every production forward call goes through
`infer()`, so a model that does not conform to the contract cannot be
trained, exported, or deployed. The registry tests
(`AIAgent/tests/unit/ml/test_registry.py`) pin each registered model's
`forward` signature against the contract, and the smoke tests
(`AIAgent/tests/unit/ml/test_models_forward.py`) forward a dummy batch
through `infer()` for every registered model.

## Registering a new model

1. Add the model class under a new directory (e.g. `MyModel/model.py`).
2. Conform its `forward` to the `Model` contract.
3. Register it in `registry.py` (`MODELS["MyModel"] = MyModelEncoder`).
4. Extend the smoke test's per-model init-kwargs table with the model's
   small kwargs — its coverage assertion fails if the table and the
   registry drift apart.
