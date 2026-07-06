"""
Fix: Add missing config parameters for the fairseq-to-HF conversion.
The model has layernorm_embedding weights but the config doesn't enable them.
"""
import json

config_path = "config.json"
with open(config_path, "r") as f:
    cfg = json.load(f)

# Add missing parameters for proper fairseq conversion
cfg["encoder_layernorm_embedding"] = True
cfg["decoder_layernorm_embedding"] = True
cfg["normalize_before"] = True  # pre-norm (fairseq convention)

with open(config_path, "w") as f:
    json.dump(cfg, f, indent=2)

print("Updated config.json with missing normalization parameters:")
print("  encoder_layernorm_embedding: true")
print("  decoder_layernorm_embedding: true")
print("  normalize_before: true")