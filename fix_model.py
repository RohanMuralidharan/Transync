"""
Simple fix: set tie_word_embeddings=true in config.json.
With tied embeddings, the shared embedding is used for both input AND output projection,
bypassing the randomly initialized lm_head.weight in the safetensors file.
"""
import json

config_path = "config.json"
with open(config_path, "r") as f:
    config = json.load(f)

config["tie_word_embeddings"] = True

with open(config_path, "w") as f:
    json.dump(config, f, indent=2)

print("Fixed! Config updated: tie_word_embeddings: true")
print("HF will now use shared embeddings for output projection (weight tying).")