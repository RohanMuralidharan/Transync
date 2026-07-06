import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import torch
from transformers import MBartForConditionalGeneration
import sentencepiece as spm
import json

# Load model
print("Loading model...")
model = MBartForConditionalGeneration.from_pretrained(".", local_files_only=True)
model.eval()

# Check current state
shared = model.model.shared.weight
dec_emb = model.model.decoder.embed_tokens.weight
lm_head = model.lm_head.weight

print(f"Before tie_weights():")
print(f"  Shared == Decoder embed: {torch.allclose(shared, dec_emb)}")
print(f"  Shared == LM head: {torch.allclose(shared, lm_head)}")
print(f"  Decoder embed == LM head: {torch.allclose(dec_emb, lm_head)}")

# Try tying weights properly
print("\nCalling model.tie_weights()...")
model.tie_weights()

# Check after
shared2 = model.model.shared.weight
dec_emb2 = model.model.decoder.embed_tokens.weight
lm_head2 = model.lm_head.weight

print(f"After tie_weights():")
print(f"  Shared == Decoder embed: {torch.allclose(shared2, dec_emb2)}")
print(f"  Shared == LM head: {torch.allclose(shared2, lm_head2)}")
print(f"  Decoder embed == LM head: {torch.allclose(dec_emb2, lm_head2)}")

# Now test translation
from hf_inference import translate_onemt

print("\n--- Testing translation ---")
result1 = translate_onemt("Hello, how are you?", "eng", "hin")
print(f"eng→hin: {result1}")

result2 = translate_onemt("నమస్కారం, మీరు ఎలా ఉన్నారు?", "tel", "eng")
print(f"tel→eng: {result2}")