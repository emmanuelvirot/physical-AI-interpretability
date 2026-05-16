# PR justification: ACT attention visualization fixes

## Summary of changes

This change set fixes several runtime issues in the ACT attention visualization example and makes it more robust with current LeRobot policy implementations and multi-camera datasets.

## 1. Load the local checkout instead of an installed stale package

`examples/visualise_original_data_attention.py` now inserts the repository root into `sys.path` before importing `physical_ai_interpretability`.

### Why

When running the example script directly, Python starts its import path from `examples/`. That caused the script to import the installed `physical_ai_interpretability` package from `site-packages` instead of the local edited source.

### What it fixes

This prevents the script from silently using stale installed code, including the old mapper implementation that still referenced `model.vision_encoder`.

## 2. Support current and older ACT vision backbone layouts

`ACTPolicyWithAttention._get_image_spatial_shapes()` now supports multiple ACT model layouts:

- `policy.model.vision_encoder.resnet_feature_extractor`
- `policy.model.backbone`
- `policy.model.backbones`

It also handles backbone outputs that are either dictionaries or tensors.

### Why

Current LeRobot ACT models expose the ResNet feature extractor as `model.backbone`, while the previous mapper assumed `model.vision_encoder.resnet_feature_extractor`.

### What it fixes

This resolves:

```text
AttributeError: 'ACT' object has no attribute 'vision_encoder'
```

## 3. Load policy weights from checkpoint config without resizing the action head

Policy loading now uses the policy class directly:

```python
policy_cls = get_policy_class(policy_cfg.type)
policy = policy_cls.from_pretrained(policy_path, config=policy_cfg)
```

instead of `make_policy(policy_cfg, ds_meta=...)`.

### Why

`make_policy()` rewrites `cfg.output_features` from dataset metadata before loading weights. If the metadata action dimension differs from the checkpoint action dimension, the model is instantiated with the wrong output size.

### What it fixes

This resolves checkpoint load failures such as:

```text
size mismatch for model.vae_encoder_action_input_proj.weight
size mismatch for model.action_head.weight
size mismatch for model.action_head.bias
```

## 4. Use the analyzed dataset metadata for preprocessing stats

The script no longer loads the hardcoded combined training dataset list. It now passes the analyzed dataset metadata into `load_policy()`:

```python
policy, policy_cfg = load_policy(
    args.policy_path,
    dataset.meta,
    args.policy_overrides,
)
```

### Why

The combined training metadata could include a different state/action schema from the dataset being analyzed. In practice, this created 7D normalizer stats while the actual observation state was 6D.

### What it fixes

This resolves normalization errors such as:

```text
RuntimeError: The size of tensor a (6) must match the size of tensor b (7)
```

It also avoids fetching many unrelated datasets before running a single episode analysis.

## 5. Batch and cast proprioceptive state observations

`prepare_observation_for_policy()` now ensures state tensors are:

- converted to the requested model dtype
- given a batch dimension when they are 1D

### Why

Image observations were batched, but state observations were not. ACT expects batched tensors.

### What it fixes

This prevents downstream shape mismatches when passing observations through the policy and preprocessor.

## 6. Make combined attention video generation work with 2+ valid cameras

Added `create_combined_attention_frame()` and updated combined-video generation.

The combined frame now:

- uses any 2 or more valid camera views
- skips `None` camera visualizations instead of dropping the whole frame
- resizes views to a common height before concatenating
- only writes the combined video if combined frames exist

### Why

The old logic required every camera view to be non-`None` and exactly the same height. With three cameras, one missing or differently sized view prevented the combined output from being written.

### What it fixes

Combined video generation now works for multi-camera setups beyond the previous strict same-height/all-cameras case.

## 7. Minor lint cleanup

Removed a few unnecessary constant f-strings and ensured the edited files pass focused lint checks.

## Validation performed

Focused checks passed:

```bash
ruff check examples/visualise_original_data_attention.py physical_ai_interpretability/attention_maps/act_attention_mapper.py
python -m py_compile examples/visualise_original_data_attention.py physical_ai_interpretability/attention_maps/act_attention_mapper.py
conda run -n attentionmap_env python examples/visualise_original_data_attention.py --help
```

`pytest` was also tried after installing the missing pytest plugins, but the repository currently has no tests under the configured `tests` path, so pytest exits with `collected 0 items`.
