import json, torch, sentencepiece as spm
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

sp = spm.SentencePieceProcessor(model_file="onemtv3b_spm.model")
with open("fairseq_dict.json", encoding="utf-8") as f:
    _fs = json.load(f)
src_sym2id = _fs["src"]
tgt_id2sym = {v: k for k, v in _fs["tgt"].items()}
EOS_ID, PAD_ID, BOS_ID, UNK_ID = _fs["special"]["eos"], _fs["special"]["pad"], _fs["special"]["bos"], _fs["special"]["unk"]

from transformers import MBartForConditionalGeneration
model = MBartForConditionalGeneration.from_pretrained(".", local_files_only=True)
model.eval()

print(f"max_position_embeddings: {model.config.max_position_embeddings}")
print(f"model.generation_config: decoder_start_token_id={model.generation_config.decoder_start_token_id}, forced_bos={model.generation_config.forced_bos_token_id}")
print(f"model.generation_config: forced_eos={model.generation_config.forced_eos_token_id}")

# Encode
tagged = "###eng_Latn-to-hin_Deva### Hello, how are you?"
pieces = sp.encode(tagged, out_type=str)
ids = [src_sym2id.get(p, UNK_ID) for p in pieces] + [EOS_ID]
input_ids = torch.tensor([ids], dtype=torch.long)
attn_mask = torch.ones_like(input_ids)

# Test 1: Let HuggingFace handle decoder_start_token_id entirely
with torch.no_grad():
    gen = model.generate(
        input_ids=input_ids,
        attention_mask=attn_mask,
        max_new_tokens=50,
    )
decoded = gen[0].tolist()
clean = [t for t in decoded if t not in (BOS_ID, PAD_ID, EOS_ID)]
pieces_out = [tgt_id2sym.get(t, f"<UNK:{t}>") for t in clean]
print(f"\nTest 1 - No special params:\n  IDs: {decoded}\n  Text: {sp.decode(pieces_out)}")

# Test 2: Set forced_bos_token_id to None and decoder_start to EOS
with torch.no_grad():
    gen = model.generate(
        input_ids=input_ids,
        attention_mask=attn_mask,
        decoder_start_token_id=EOS_ID,
        forced_bos_token_id=None,
        max_new_tokens=50,
    )
decoded = gen[0].tolist()
clean = [t for t in decoded if t not in (BOS_ID, PAD_ID, EOS_ID)]
pieces_out = [tgt_id2sym.get(t, f"<UNK:{t}>") for t in clean]
print(f"\nTest 2 - EOS decoder_start, forced_bos=None:\n  IDs: {decoded}\n  Text: {sp.decode(pieces_out)}")

# Test 3: With forced_bos_token_id=EOS_ID 
with torch.no_grad():
    gen = model.generate(
        input_ids=input_ids,
        attention_mask=attn_mask,
        decoder_start_token_id=EOS_ID,
        forced_bos_token_id=EOS_ID,
        max_new_tokens=50,
    )
decoded = gen[0].tolist()
clean = [t for t in decoded if t not in (BOS_ID, PAD_ID, EOS_ID)]
pieces_out = [tgt_id2sym.get(t, f"<UNK:{t}>") for t in clean]
print(f"\nTest 3 - Both forced_bos=EOS:\n  IDs: {decoded}\n  Text: {sp.decode(pieces_out)}")

# Test 4: Using the generation_config as is (load from generation_config.json)
with torch.no_grad():
    gen = model.generate(
        input_ids=input_ids,
        attention_mask=attn_mask,
        max_new_tokens=50,
        num_beams=1,
        do_sample=False,
    )
decoded = gen[0].tolist()
clean = [t for t in decoded if t not in (BOS_ID, PAD_ID, EOS_ID)]
pieces_out = [tgt_id2sym.get(t, f"<UNK:{t}>") for t in clean]
print(f"\nTest 4 - No beam, greedy:\n  IDs: {decoded}\n  Text: {sp.decode(pieces_out)}")