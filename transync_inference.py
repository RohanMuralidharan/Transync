""
Transync - Indic Multilingual Translation Inference
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
    # Additional ML50 languages
    'ara': 'ar_AR',
    'ces': 'cs_CZ',
    'deu': 'de_DE',
    'spa': 'es_XX',
    'est': 'et_EE',
    'fin': 'fi_FI',
    'fra': 'fr_XX',
    'heb': 'he_IL',
    'hrv': 'hr_HR',
    'ind': 'id_ID',
    'ita': 'it_IT',
    'jpn': 'ja_XX',
    'kat': 'ka_GE',
    'kaz': 'kk_KZ',
    'khm': 'km_KH',
    'kor': 'ko_KR',
    'lit': 'lt_LT',
    'lav': 'lv_LV',
    'mkd': 'mk_MK',
    'mon': 'mn_MN',
    'mya': 'my_MM',
    'nld': 'nl_XX',
    'pol': 'pl_PL',
    'pus': 'ps_AF',
    'por': 'pt_XX',
    'ron': 'ro_RO',
    'rus': 'ru_RU',
    'sin': 'si_LK',
    'slk': 'sl_SI',
    'swe': 'sv_SE',
    'swa': 'sw_KE',
    'tha': 'th_TH',
    'tgl': 'tl_XX',
    'tur': 'tr_TR',
    'ukr': 'uk_UA',
    'vie': 'vi_VN',
    'xho': 'xh_ZA',
    'zho': 'zh_CN',
    'aze': 'az_AZ',
    'fas': 'fa_IR',
    'glg': 'gl_ES',
    'afr': 'af_ZA',
}

# Reverse mapping for display
CODE_TO_LANG = {
    'eng': 'English', 'hin': 'Hindi', 'tel': 'Telugu', 'tam': 'Tamil',
    'mal': 'Malayalam', 'kan': 'Kannada', 'ben': 'Bengali', 'guj': 'Gujarati',
    'mar': 'Marathi', 'pan': 'Punjabi', 'urd': 'Urdu', 'asm': 'Assamese',
    'npi': 'Nepali', 'ory': 'Odia', 'san': 'Sanskrit', 'mai': 'Maithili',
    'brx': 'Bodo', 'doi': 'Dogri', 'gom': 'Konkani', 'mni': 'Manipuri',
    'sat': 'Santali', 'kas': 'Kashmiri', 'snd': 'Sindhi',
    'ara': 'Arabic', 'ces': 'Czech', 'deu': 'German', 'spa': 'Spanish',
    'est': 'Estonian', 'fin': 'Finnish', 'fra': 'French', 'heb': 'Hebrew',
    'hrv': 'Croatian', 'ind': 'Indonesian', 'ita': 'Italian', 'jpn': 'Japanese',
    'kat': 'Georgian', 'kaz': 'Kazakh', 'khm': 'Khmer', 'kor': 'Korean',
    'lit': 'Lithuanian', 'lav': 'Latvian', 'mkd': 'Macedonian', 'mon': 'Mongolian',
    'mya': 'Burmese', 'nld': 'Dutch', 'pol': 'Polish', 'pus': 'Pashto',
    'por': 'Portuguese', 'ron': 'Romanian', 'rus': 'Russian', 'sin': 'Sinhala',
    'slk': 'Slovak', 'swe': 'Swedish', 'swa': 'Swahili', 'tha': 'Thai',
    'tgl': 'Tagalog', 'tur': 'Turkish', 'ukr': 'Ukrainian', 'vie': 'Vietnamese',
    'xho': 'Xhosa', 'zho': 'Chinese', 'aze': 'Azerbaijani', 'fas': 'Persian',
    'glg': 'Galician', 'afr': 'Afrikaans',
}

# Load model and tokenizer (cached after first load)
_model = None
_tokenizer = None
_device = None


def _get_device() -> str:
    """Detect and return the best available device (CUDA/CPU)."""
    global _device
    if _device is None:
        if torch.cuda.is_available():
            _device = "cuda"
            print(f"✓ Using GPU: {torch.cuda.get_device_name(0)}")
        else:
            _device = "cpu"
            print("ℹ Using CPU (CUDA not available)")
    return _device


def _load_model():
    """Lazy load model and tokenizer with device optimization."""
    global _model, _tokenizer
    if _model is None:
        device = _get_device()
        print("Loading Transync model...")
        _model = MBartForConditionalGeneration.from_pretrained('.').to(device)
        _tokenizer = MBart50Tokenizer.from_pretrained('.')
        if device == "cuda":
            _model = _model.half()  # Use FP16 for faster inference on GPU
        print("✓ Model ready")
    return _model, _tokenizer


def translate_onemt(
    text: str,
    source_lang: str,
    target_lang: str,
    max_length: int = 256,
    num_beams: int = 5,
    temperature: float = 1.0,
    repetition_penalty: float = 1.3,
    no_repeat_ngram_size: int = 3,
) -> str:
    """
    Translate text from source language to target language.

    Args:
        text: Input text to translate.
        source_lang: Source language code (e.g., 'eng', 'hin', 'tel').
        target_lang: Target language code (e.g., 'eng', 'hin', 'tel').
        max_length: Maximum length of generated translation.
        num_beams: Number of beams for beam search.
        temperature: Sampling temperature (higher = more diverse).
        repetition_penalty: Penalty for repeating tokens.
        no_repeat_ngram_size: Size of n-grams to avoid repeating.

    Returns:
        Translated text.

    Raises:
        ValueError: If an unsupported language code is provided.

    Example:
        >>> translate_onemt("Hello, how are you?", "eng", "hin")
        'नमस्ते, आप कैसे हैं?'
    """
    if not text or not text.strip():
        return ""

    model, tokenizer = _load_model()

    # Get MBart language codes
    src_code = LANG_CODES.get(source_lang, source_lang)
    tgt_code = LANG_CODES.get(target_lang, target_lang)

    # Validate source language
    if src_code not in tokenizer.lang_code_to_id:
        valid_codes = sorted(LANG_CODES.keys())
        raise ValueError(
            f"Unsupported source language: '{source_lang}'. "
            f"Supported codes: {', '.join(valid_codes)}"
        )

    # Validate target language
    tgt_token_id = tokenizer.lang_code_to_id.get(tgt_code)
    if tgt_token_id is None:
        valid_codes = sorted(LANG_CODES.keys())
        raise ValueError(
            f"Unsupported target language: '{target_lang}'. "
            f"Supported codes: {', '.join(valid_codes)}"
        )

    # Set source language and tokenize
    tokenizer.src_lang = src_code
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
        padding=True,
    ).to(_device)

    # Generate translation
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            forced_bos_token_id=tgt_token_id,
            max_length=max_length,
            num_beams=num_beams,
            no_repeat_ngram_size=no_repeat_ngram_size,
            repetition_penalty=repetition_penalty,
            temperature=temperature,
            early_stopping=True,
        )

    # Decode
    translated = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return translated


def translate_batch(
    texts: list,
    source_lang: str,
    target_lang: str,
    batch_size: int = 32,
    max_length: int = 256,
    num_beams: int = 5,
    show_progress: bool = True,
) -> list:
    """
    Translate a batch of texts efficiently using optimized batching.

    Args:
        texts: List of input texts to translate.
        source_lang: Source language code.
        target_lang: Target language code.
        batch_size: Number of texts to process at once (default: 32).
        max_length: Maximum length of generated translation.
        num_beams: Number of beams for beam search.
        show_progress: Whether to show a progress bar.

    Returns:
        List of translated texts.

    Example:
        >>> translate_batch(["Hello", "How are you?"], "eng", "hin")
        ['नमस्ते', 'आप कैसे हैं?']
    """
    if not texts:
        return []

    model, tokenizer = _load_model()
    tgt_code = LANG_CODES.get(target_lang, target_lang)
    tgt_token_id = tokenizer.lang_code_to_id.get(tgt_code)

    if tgt_token_id is None:
        raise ValueError(f"Unsupported target language: {target_lang}")

    results = []
    total_batches = (len(texts) + batch_size - 1) // batch_size

    if show_progress:
        try:
            from tqdm import tqdm
            iterator = tqdm(
                range(0, len(texts), batch_size),
                desc="Translating",
                unit="batch",
                total=total_batches,
            )
        except ImportError:
            iterator = range(0, len(texts), batch_size)
            print(f"Translating {len(texts)} texts in {total_batches} batches...")
    else:
        iterator = range(0, len(texts), batch_size)

    tokenizer.src_lang = LANG_CODES.get(source_lang, source_lang)

    for i in iterator:
        batch_texts = texts[i:i + batch_size]

        # Tokenize batch
        inputs = tokenizer(
            batch_texts,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
            padding=True,
        ).to(_device)

        # Generate batch
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                forced_bos_token_id=tgt_token_id,
                max_length=max_length,
                num_beams=num_beams,
                no_repeat_ngram_size=3,
                repetition_penalty=1.3,
                early_stopping=True,
            )

        # Decode batch results
        batch_results = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        results.extend(batch_results)

    return results


def list_languages(category: str = "all") -> None:
    """
    Print available languages and their codes.

    Args:
        category: Filter by category ('all', 'indic', 'other').
    """
    indic_langs = {
        'asm': 'Assamese', 'ben': 'Bengali', 'brx': 'Bodo', 'doi': 'Dogri',
        'gom': 'Konkani', 'guj': 'Gujarati', 'hin': 'Hindi', 'kan': 'Kannada',
        'kas': 'Kashmiri', 'mai': 'Maithili', 'mal': 'Malayalam', 'mar': 'Marathi',
        'mni': 'Manipuri', 'npi': 'Nepali', 'ory': 'Odia', 'pan': 'Punjabi',
        'san': 'Sanskrit', 'sat': 'Santali', 'snd': 'Sindhi', 'tam': 'Tamil',
        'tel': 'Telugu', 'urd': 'Urdu',
    }

    print("\n╔══════════════════════════════════════════════╗")
    print("║        Transync - Supported Languages        ║")
    print("╚══════════════════════════════════════════════╝")

    if category in ("all", "indic"):
        print(f"\n📚 Indian Languages ({len(indic_langs)}):")
        print("─" * 45)
        for code in sorted(indic_langs):
            print(f"  {code:6s} → {indic_langs[code]}")

    if category in ("all", "other"):
        other = {k: v for k, v in sorted(CODE_TO_LANG.items()) if k not in indic_langs}
        print(f"\n🌍 Other Languages ({len(other)}):")
        print("─" * 45)
        for code, name in other.items():
            print(f"  {code:6s} → {name}")

    print()


# CLI interface
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Transync - Indic Multilingual Translation Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python transync_inference.py eng hin "Hello, how are you?"
  python transync_inference.py hin tel "नमस्ते, आप कैसे हैं?" --beams 3
  python transync_inference.py --batch eng hin -f input.txt -o output.txt
  python transync_inference.py --list-langs
        """,
    )

    parser.add_argument(
        "source_lang",
        nargs="?",
        help="Source language code (e.g., 'eng', 'hin', 'tel')",
    )
    parser.add_argument(
        "target_lang",
        nargs="?",
        help="Target language code (e.g., 'eng', 'hin', 'tel')",
    )
    parser.add_argument(
        "text",
        nargs="*",
        help="Text to translate",
    )
    parser.add_argument(
        "--beams",
        type=int,
        default=5,
        help="Number of beams for beam search (default: 5)",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=256,
        help="Maximum translation length (default: 256)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Sampling temperature (default: 1.0)",
    )
    parser.add_argument(
        "--list-langs",
        action="store_true",
        help="List all supported languages and exit",
    )
    parser.add_argument(
        "--batch",
        action="store_true",
        help="Batch translation mode (requires --file)",
    )
    parser.add_argument(
        "-f", "--file",
        type=str,
        help="Input file path for batch translation",
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        help="Output file path for batch translation",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for batch translation (default: 32)",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Hide progress bar during batch translation",
    )

    args = parser.parse_args()

    # List languages mode
    if args.list_langs:
        list_languages()
        sys.exit(0)

    # Validate required arguments
    if not args.source_lang or not args.target_lang:
        parser.print_help()
        print("\n❌ Error: source_lang and target_lang are required.")
        print("   Use --list-langs to see all supported language codes.")
        sys.exit(1)

    # Batch translation from file
    if args.batch or args.file:
        if not args.file:
            print("❌ Error: --file is required for batch translation mode.")
            sys.exit(1)

        try:
            with open(args.file, "r", encoding="utf-8") as f:
                texts = [line.strip() for line in f if line.strip()]
        except FileNotFoundError:
            print(f"❌ Error: File not found: {args.file}")
            sys.exit(1)

        if not texts:
            print("❌ Error: Input file is empty.")
            sys.exit(1)

        print(f"📖 Loaded {len(texts)} texts from {args.file}")
        results = translate_batch(
            texts,
            args.source_lang,
            args.target_lang,
            batch_size=args.batch_size,
            max_length=args.max_length,
            num_beams=args.beams,
            show_progress=not args.no_progress,
        )

        if args.output:
            with open(args.output, "w", encoding="utf-8") as f:
                for result in results:
                    f.write(result + "\n")
            print(f"✓ Results written to {args.output}")
        else:
            for i, (orig, trans) in enumerate(zip(texts, results), 1):
                print(f"\n[{i}]")
                print(f"  Input:  {orig}")
                print(f"  Output: {trans}")

    # Single translation
    elif args.text:
        text = " ".join(args.text)
        src_name = CODE_TO_LANG.get(args.source_lang, args.source_lang)
        tgt_name = CODE_TO_LANG.get(args.target_lang, args.target_lang)

        print(f"\n🔤 {src_name} → {tgt_name}")
        print(f"   Input:  {text}")

        result = translate_onemt(
            text,
            args.source_lang,
            args.target_lang,
            max_length=args.max_length,
            num_beams=args.beams,
            temperature=args.temperature,
        )

        print(f"   Output: {result}")

    else:
        print("❌ Error: No text provided for translation.")
        print("   Usage: python transync_inference.py <source> <target> <text>")
        sys.exit(1)