"""
The decoder has its OWN embedding (model.decoder.embed_tokens.weight) which differs
from the shared embedding (model.shared.weight). This means the model was trained
with SEPARATE encoder and decoder embeddings.

The lm_head should project from the DECODER embedding space, not the shared/encoder one.
"""
import torch
import json
from safetensors.torch import save_file, load_file
import os, shutil

print("Loading original model...")
state = load_file("model.safetensors")

shared = state["model.shared.weight"]
dec_emb = state["model.decoder.embed_tokens.weight"]
lm_head = state["lm_head.weight"]

print(f"Shared: mean={shared.mean():.6f}, std={shared.std():.6f}")
print(f"Decoder embed: mean={dec_emb.mean():.6f}, std={dec_emb.std():.6f}")
print(f"LM head: mean={lm_head.mean():.6f}, std={lm_head.std():.6f}")

# The lm_head should map from decoder embedding space to vocab
# Since the decoder has its OWN embedding, use decoder.embed_tokens.weight
print("\nCopying decoder.embed_tokens.weight -> lm_head.weight")
state["lm_head.weight"] = state["model.decoder.embed_tokens.weight"].clone()

# Set tie_word_embeddings=false so HF doesn't override with shared
config_path = "config.json"
with open(config_path, "r") as f:
    config = json.load(f)
config["tie_word_embeddings"] = False
with open(config_path, "w") as f:
    json.dump(config, f, indent=2)
print("Updated config.json: tie_word_embeddings = false")

# Save backup
if not os.path.exists("model.safetensors.bak2"):
    print("Creating backup model.safetensors.bak2...")
    shutil.copy2("model.safetensors", "model.safetensors.bak2")

print("Saving fixed model...")
save_file(state, "model.safetensors")
print("Done! LM head now matches decoder embeddings.")