import json
import sentencepiece as spm
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

sp = spm.SentencePieceProcessor(model_file='onemtv3b_spm.model')

with open('fairseq_dict.json', encoding='utf-8') as f:
    _fs = json.load(f)

src_sym2id = _fs['src']
tgt_id2sym = {v: k for k, v in _fs['tgt'].items()}

print("SPM vocab size:", sp.GetPieceSize())
print("Fairseq src dict size:", len(src_sym2id))
print("Fairseq tgt dict size:", len(_fs['tgt']))

# Check id2piece
id2piece = _fs.get('id2piece', {})
print(f"\nid2piece field present: {bool(id2piece)}")
print(f"id2piece has {len(id2piece)} entries")
if id2piece:
    keys = sorted(id2piece.keys(), key=lambda x: int(x))[:10]
    for k in keys:
        print(f"  id2piece[{k}] = {repr(id2piece[k])}")

# Critical: Are SPM's native IDs the same as fairseq IDs?
spm_id_0 = sp.IdToPiece(0)  # SPM says ID 0 = ?
print(f"\nSPM IdToPiece(0) = {repr(spm_id_0)}")
print(f"SPM IdToPiece(1) = {repr(sp.IdToPiece(1))}")
print(f"SPM IdToPiece(2) = {repr(sp.IdToPiece(2))}")

# Compare: the id2piece in fairseq dict maps fairseq_id -> piece
# This tells us what piece the MODEL expects at each embedding index
if id2piece:
    print("\nVerifying: for fairseq_id 0-10, what piece does id2piece say?")
    for i in range(10):
        piece_via_id2piece = id2piece.get(str(i), 'MISSING')
        print(f"  fairseq_id {i} -> piece={repr(piece_via_id2piece)} | src_sym2id[piece]={src_sym2id.get(piece_via_id2piece, 'N/A')}")

# Now encode a test sentence
tagged = "###eng_Latn-to-hin_Deva### Hello, how are you?"
pieces = sp.encode(tagged, out_type=str)

# Map using src_sym2id (fairseq ordering)
our_ids = [src_sym2id.get(p, 3) for p in pieces]

print(f"\nTagged: {repr(tagged)}")
print(f"SPM pieces ({len(pieces)}): {pieces}")
print(f"Our fairseq IDs: {our_ids}")

# Now decode: what do these IDs correspond to in the model's id2piece?
if id2piece:
    print("\nWhat our IDs map to via id2piece (model's expected ordering):")
    model_pieces = []
    for i, pid in enumerate(our_ids):
        piece = id2piece.get(str(pid), f'<MISSING:{pid}>')
        model_pieces.append(piece)
    print(f"  {model_pieces}")
    
# Check SPM native IDs for same pieces
spm_native_ids = [sp.PieceToId(p) for p in pieces]
print(f"\nSPM native IDs: {spm_native_ids}")
print(f"Fairseq IDs:     {our_ids}")
print(f"Are they same? {spm_native_ids == our_ids}")
if spm_native_ids != our_ids:
    print("DIFFERENT ORDERING! This means fairseq IDs != SPM IDs")