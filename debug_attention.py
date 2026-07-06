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

# Load with tied embeddings
model = MBartForConditionalGeneration.from_pretrained(".", local_files_only=True, tie_word_embeddings=True)
model.eval()

# Test if model responds differently to different inputs
for input_text in [
    "###eng_Latn-to-hin_Deva### Hello, how are you?",
    "###eng_Latn-to-hin_Deva### Good morning.",
]:
    pieces = sp.encode(input_text, out_type=str)
    ids = [src_sym2id.get(p, UNK_ID) for p in pieces] + [EOS_ID]
    input_ids = torch.tensor([ids], dtype=torch.long)
    attn_mask = torch.ones_like(input_ids)
    
    # Get encoder output
    with torch.no_grad():
        encoder_outputs = model.model.encoder(input_ids=input_ids, attention_mask=attn_mask)
        enc_last = encoder_outputs.last_hidden_state
    
    # Check if encoder output is different
    print(f"\nInput: {repr(input_text[:40])}...")
    print(f"  Encoder output mean: {enc_last.mean().item():.6f}")
    print(f"  Encoder output std: {enc_last.std().item():.6f}")
    
    # Generate with beam=1 and no repetition penalty
    with torch.no_grad():
        gen = model.generate(
            input_ids=input_ids,
            attention_mask=attn_mask,
            decoder_start_token_id=EOS_ID,
            max_new_tokens=30,
            num_beams=1,
            do_sample=False,
        )
    
    decoded_ids = gen[0].tolist()
    clean = [t for t in decoded_ids if t not in (BOS_ID, PAD_ID, EOS_ID)]
    pieces_out = [tgt_id2sym.get(t, f"<UNK:{t}>") for t in clean]
    text_out = sp.decode(pieces_out)
    print(f"  Output: {text_out}")