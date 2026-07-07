"""
Transync - Indic Multilingual Translation Inference
Uses facebook/mbart-large-50-one-to-many-mmt model
Supports 50+ languages including all major Indian languages
"""
import sys
import io
import torch
from transformers import MBartForConditionalGeneration, MBart50Tokenizer

# Fix Windows console encoding
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Language code mapping (short code to MBart format)
LANG_CODES = {
    'eng': 'en_XX',
    'hin': 'hi_IN',
    'tel': 'te_IN',
    'tam': 'ta_IN',
    'mal': 'ml_IN',
    'kan': 'kn_IN',
    'ben': 'bn_IN',
    'guj': 'gu_IN',
    'mar': 'mr_IN',
    'pan': 'pa_IN',
    'urd': 'ur_PK',
    'asm': 'as_IN',
    'npi': 'ne_NP',
    'ory': 'or_IN',
    'san': 'sa_IN',
    'mai': 'mai_IN',
    'brx': 'brx_IN',
    'doi': 'doi_IN',
    'gom': 'gom_IN',
    'mni': 'mni_IN',
    'sat': 'sat_IN',
    'kas': 'ks_IN',
    'snd': 'sd_IN',
}

# Load model and tokenizer (cached after first load)
_model = None
_tokenizer = None

def _load_model():
    """Lazy load model and tokenizer"""
    global _model, _tokenizer
    if _model is None:
        print("Loading Transync model...")
        _model = MBartForConditionalGeneration.from_pretrained('.')
        _tokenizer = MBart50Tokenizer.from_pretrained('.')
        print("✓ Model ready")
    return _model, _tokenizer

def translate_onemt(text: str, source_lang: str, target_lang: str) -> str:
    """
    Translate text from source language to target language
    
    Args:
        text: Input text to translate
        source_lang: Source language code (e.g., 'eng', 'hin', 'tel')
        target_lang: Target language code (e.g., 'eng', 'hin', 'tel')
    
    Returns:
        Translated text
    
    Example:
        >>> translate_onemt("Hello, how are you?", "eng", "hin")
        'नमस्ते, आप कैसे हैं?'
    """
    model, tokenizer = _load_model()
    
    # Get MBart language codes
    src_code = LANG_CODES.get(source_lang, source_lang)
    tgt_code = LANG_CODES.get(target_lang, target_lang)
    
    # Set source language and tokenize
    tokenizer.src_lang = src_code
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=256)
    
    # Get target language token ID
    tgt_token_id = tokenizer.lang_code_to_id.get(tgt_code)
    if tgt_token_id is None:
        raise ValueError(f"Unsupported target language: {target_lang}")
    
    # Generate translation
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            forced_bos_token_id=tgt_token_id,
            max_length=256,
            num_beams=5,
            no_repeat_ngram_size=3,
            repetition_penalty=1.3
        )
    
    # Decode
    translated = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return translated

def translate_batch(texts: list, source_lang: str, target_lang: str, batch_size: int = 32) -> list:
    """
    Translate a batch of texts
    
    Args:
        texts: List of input texts
        source_lang: Source language code
        target_lang: Target language code
        batch_size: Number of texts to process at once
    
    Returns:
        List of translated texts
    """
    results = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        for text in batch:
            translated = translate_onemt(text, source_lang, target_lang)
            results.append(translated)
    return results

# CLI interface
if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: python transync_inference.py <source_lang> <target_lang> <text>")
        print("\nExample:")
        print("  python transync_inference.py eng hin 'Hello, how are you?'")
        print("\nSupported languages:")
        for code in sorted(LANG_CODES.keys()):
            print(f"  {code}")
        sys.exit(1)
    
    source_lang = sys.argv[1]
    target_lang = sys.argv[2]
    text = " ".join(sys.argv[3:])
    
    result = translate_onemt(text, source_lang, target_lang)
    print(result)