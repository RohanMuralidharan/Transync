import pytest

from transync.languages import resolve_language_code


def test_resolve_language_code_accepts_primary_code() -> None:
    assert resolve_language_code("tel") == "tel_Telu"


def test_resolve_language_code_accepts_alias() -> None:
    assert resolve_language_code("ori") == "ory_Orya"


def test_resolve_language_code_rejects_unknown_code() -> None:
    with pytest.raises(ValueError, match="Unsupported language code"):
        resolve_language_code("unknown")
