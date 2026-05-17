# PR Change Summary - 2026-05-17

## Summary

This change extends the ACT attention visualization path so it can output multiple attention sources from the same policy rollout. In addition to the existing default decoder cross-attention map, it now captures the first encoder self-attention layer and writes separate videos for each attention source.

## Files changed

- `physical_ai_interpretability/attention_maps/act_attention_mapper.py`
- `examples/visualise_original_data_attention.py`

## 1. Capture first encoder self-attention

`ACTPolicyWithAttention` now hooks:

```python
policy.model.encoder.layers[0].self_attn
```

and labels that output as:

```text
encoder_first
```

### Why

Some interpretability examples discuss visualizing early transformer attention, especially encoder attention over observation tokens. The existing mapper only observed decoder cross-attention. For policies with only one decoder layer, there is no earlier decoder layer to inspect, so the encoder is the useful place to look for a first-layer attention view.

### What it fixes or enables

This enables output videos such as:

```text
attention_ep9_encoder_first_cam0_....mp4
attention_ep9_encoder_first_combined_....mp4
```

These show first encoder self-attention mapped back onto the camera feature maps.

## 2. Keep the default decoder cross-attention output

The mapper still captures the existing default attention target:

```python
policy.model.decoder.layers[-1].multihead_attn
```

and labels it as:

```text
default
```

### Why

Decoder cross-attention remains the most direct view of action-query tokens attending to observation tokens, so it should remain available for comparison with encoder attention.

### What it fixes or enables

The script now writes separate default decoder outputs such as:

```text
attention_ep9_default_cam0_....mp4
attention_ep9_default_combined_....mp4
```

## 3. Avoid duplicate first/default decoder outputs for one-layer decoders

The mapper now prints encoder and decoder layer counts at startup. If the decoder has only one layer, it does not create a duplicate `decoder_first` output, because:

```python
decoder.layers[0] == decoder.layers[-1]
```

### Why

For one-layer ACT decoders, `first` and `default` refer to the same `MultiheadAttention` module and produce identical videos.

### What it fixes

Avoids writing two identical decoder attention outputs when the checkpoint only has one decoder layer.

Expected print for a one-layer decoder:

```text
ACT encoder self-attention layers available: N. Capturing encoder_first layer (index 0).
ACT decoder cross-attention layers available: 1. The first and default final decoder layers are the same module, so only 'default' decoder attention will be written.
```

## 4. Support multi-output attention maps in the example script

`examples/visualise_original_data_attention.py` now treats visualizations as a dictionary keyed by attention-source label:

```text
encoder_first
default
decoder_first  # only when the decoder has more than one layer
```

It stores separate per-camera and combined video buffers for each label.

### Why

The mapper can now return more than one attention map set per action. The example script needed to save each set independently instead of assuming a single list of camera visualizations.

### What it fixes or enables

The script now writes clearly named videos per attention source, for example:

```text
attention_ep9_encoder_first_cam0_....mp4
attention_ep9_encoder_first_combined_....mp4
attention_ep9_default_cam0_....mp4
attention_ep9_default_combined_....mp4
```

## 5. Encoder attention mapping method

A new `_map_encoder_attention_to_images()` method maps encoder self-attention back to image feature maps.

For encoder self-attention, observation tokens are both queries and keys/values. The implementation averages over all encoder query tokens and keeps the attention paid to each camera image-token region.

### Why

Decoder cross-attention maps naturally from action queries to observation tokens. Encoder self-attention is different: there is no action query. Averaging query tokens gives a broad view of how much the first encoder layer attends to each image region.

### Interpretation note

`encoder_first` should not be interpreted exactly the same way as decoder cross-attention:

- `encoder_first`: observation tokens attending to observation tokens
- `default`: action-query tokens attending to observation tokens

So `encoder_first` is useful for early visual-token structure, while `default` is closer to action-conditioned visual relevance.

## Validation performed

Focused checks passed:

```bash
ruff check physical_ai_interpretability/attention_maps/act_attention_mapper.py examples/visualise_original_data_attention.py
python -m py_compile physical_ai_interpretability/attention_maps/act_attention_mapper.py examples/visualise_original_data_attention.py
conda run -n attentionmap_env python examples/visualise_original_data_attention.py --help
```
