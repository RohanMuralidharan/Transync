"""Language code helpers for Transync."""

from __future__ import annotations

LANGUAGE_CODE_MAP: dict[str, str] = {
    "asm": "asm_Beng",
    "ben": "ben_Beng",
    "ban": "ben_Beng",
    "brx": "brx_Deva",
    "doi": "doi_Deva",
    "gom": "gom_Deva",
    "guj": "guj_Gujr",
    "hin": "hin_Deva",
    "kan": "kan_Knda",
    "kas_arab": "kas_Arab",
    "kas": "kas_Arab",
    "kas_deva": "kas_Deva",
    "mai": "mai_Deva",
    "mal": "mal_Mlym",
    "mar": "mar_Deva",
    "mni_beng": "mni_Beng",
    "mni": "mni_Beng",
    "mni_mtei": "mni_Mtei",
    "npi": "npi_Deva",
    "ory": "ory_Orya",
    "ori": "ory_Orya",
    "odi": "ory_Orya",
    "pan": "pan_Guru",
    "pun": "pan_Guru",
    "san": "san_Deva",
    "sat": "sat_Olck",
    "snd_arab": "snd_Arab",
    "snd_deva": "snd_Deva",
    "snd": "snd_Arab",
    "tam": "tam_Taml",
    "tel": "tel_Telu",
    "urd_arab": "urd_Arab",
    "urd": "urd_Arab",
    "eng": "eng_Latn",
}

PRIMARY_LANGUAGES: tuple[tuple[str, str, str], ...] = (
    ("asm", "Assamese", "asm_Beng"),
    ("ben", "Bengali", "ben_Beng"),
    ("brx", "Bodo", "brx_Deva"),
    ("doi", "Dogri", "doi_Deva"),
    ("gom", "Konkani", "gom_Deva"),
    ("guj", "Gujarati", "guj_Gujr"),
    ("hin", "Hindi", "hin_Deva"),
    ("kan", "Kannada", "kan_Knda"),
    ("kas", "Kashmiri", "kas_Arab"),
    ("kas_deva", "Kashmiri", "kas_Deva"),
    ("mai", "Maithili", "mai_Deva"),
    ("mal", "Malayalam", "mal_Mlym"),
    ("mar", "Marathi", "mar_Deva"),
    ("mni", "Meitei", "mni_Beng"),
    ("mni_mtei", "Meitei", "mni_Mtei"),
    ("npi", "Nepali", "npi_Deva"),
    ("ory", "Odia", "ory_Orya"),
    ("pan", "Punjabi", "pan_Guru"),
    ("san", "Sanskrit", "san_Deva"),
    ("sat", "Santali", "sat_Olck"),
    ("snd", "Sindhi", "snd_Arab"),
    ("snd_deva", "Sindhi", "snd_Deva"),
    ("tam", "Tamil", "tam_Taml"),
    ("tel", "Telugu", "tel_Telu"),
    ("urd", "Urdu", "urd_Arab"),
    ("eng", "English", "eng_Latn"),
)


def format_supported_language_codes() -> str:
    """Return a comma-separated list of accepted language codes."""

    return ", ".join(sorted(LANGUAGE_CODE_MAP))


def resolve_language_code(code: str) -> str:
    """Resolve a user-facing language code to the FLORES-style model tag."""

    normalized = code.strip().lower()
    try:
        return LANGUAGE_CODE_MAP[normalized]
    except KeyError as exc:
        supported = format_supported_language_codes()
        raise ValueError(
            f"Unsupported language code '{code}'. Supported codes: {supported}"
        ) from exc
