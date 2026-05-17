
# Commands ran with attention maps

# 01 - Plushie in the box - recording - episode 09
python examples/visualise_original_data_attention.py \
    --dataset-repo-id emmanuel-v/2026-02-11_so101_plushie_02 \
    --episode-id 9 \
    --policy-path emmanuel-v/policy_2026-02-11_so101_plushie_02e \
    --output-dir ./output/attention_analysis_results

# 02 - Detection test - eval - episode 09
python examples/visualise_original_data_attention.py \
    --dataset-repo-id jogarulfop/eval_2026-04-27_dragontactile_test_detection_vibration_metal_exitation_v3 \
    --episode-id 9 \
    --policy-path jogarulfop/2026-04-27_dragontactile_test_detection_vibration \
    --output-dir ./output/attention_analysis_results

# 03 - Detection test - eval - episode 05
python examples/visualise_original_data_attention.py \
    --dataset-repo-id jogarulfop/eval_2026-04-27_dragontactile_test_detection_vibration_metal_exitation_v3 \
    --episode-id 5 \
    --policy-path jogarulfop/2026-04-27_dragontactile_test_detection_vibration \
    --output-dir ./output/attention_analysis_results

# --------------------------------------------------------------------

# Detection tests, True or False
* 00 - 01 - 02 - 03 - 04 - 05 - 06 - 07 - 08 - 09
* No - No - No - No - No - Yes- Yes- Yes- Yes- Yes (recordings)
* No/- Yes- No - No - Yes- Yes- No - Yes- No + perturbations - Yes + perturbations (evals)


# --------------------------------------------------------------------


# 04 - Detection test - recording - episode 05 (clear tap tap and noise constant afterwards)
python examples/visualise_original_data_attention.py \
    --dataset-repo-id jogarulfop/2026-04-27_dragontactile_test_detection_vibration_metal_exitation_v3 \
    --episode-id 5 \
    --policy-path jogarulfop/2026-04-27_dragontactile_test_detection_vibration \
    --output-dir ./output/attention_analysis_results

# 05 - Plushie in the box - recording - episode 07
# https://huggingface.co/spaces/lerobot/visualize_dataset?path=%2Femmanuel-v%2F2026-02-11_so101_plushie_02%2Fepisode_7
python examples/visualise_original_data_attention.py \
    --dataset-repo-id emmanuel-v/2026-02-11_so101_plushie_02 \
    --episode-id 7 \
    --policy-path emmanuel-v/policy_2026-02-11_so101_plushie_02e \
    --output-dir ./output/attention_analysis_results

# 06 - Plushie in the box - recording - episode 06
python examples/visualise_original_data_attention.py \
    --dataset-repo-id emmanuel-v/2026-02-11_so101_plushie_02 \
    --episode-id 6 \
    --policy-path emmanuel-v/policy_2026-02-11_so101_plushie_02e \
    --output-dir ./output/attention_analysis_results


# --------------------------------------------------------------------


# Training a model from the dataset "2026-05-13_chakeitup_alubox_20260513_153802"
# https://huggingface.co/spaces/lerobot/visualize_dataset?path=jogarulfop%2F2026-05-13_chakeitup_alubox_20260513_153802
```bash
lerobot-train \
  --dataset.repo_id="jogarulfop/2026-05-13_chakeitup_alubox_20260513_153802" \
  --policy.type=act \
  --output_dir=outputs/train/act_2026-05-13_chakeitup_alubox_d \
  --job_name=act_2026-05-13_chakeitup_alubox_d \
  --policy.device=cuda \
  --wandb.enable=true \
  --policy.repo_id="emmanuel-v/policy_2026-05-13_chakeitup_alubox_d" \
  --save_freq=10_000 \
  --batch_size=16
  # --steps=20_000 \
```
  
# Detection tests, True or False
* 00     - 01    - 02     - 03     - 04    - 05   - 06    - 07    - 08    - 09    - ...
* full?  - full! - empty! - empty? - full? - full - full  - full  - full  - full! - ...
We test out on the 09 for full and 02 for empty.

# 07 - shakeitup alu box - recording - episode 09 (full)
python examples/visualise_original_data_attention.py \
    --dataset-repo-id jogarulfop/2026-05-13_chakeitup_alubox_20260513_153802 \
    --episode-id 9 \
    --policy-path emmanuel-v/policy_2026-05-13_chakeitup_alubox_d \
    --output-dir ./output/attention_analysis_results

# 08 - shakeitup alu box - recording - episode 01 (empty)
python examples/visualise_original_data_attention.py \
    --dataset-repo-id jogarulfop/2026-05-13_chakeitup_alubox_20260513_153802 \
    --episode-id 1 \
    --policy-path emmanuel-v/policy_2026-05-13_chakeitup_alubox_d \
    --output-dir ./output/attention_analysis_results



