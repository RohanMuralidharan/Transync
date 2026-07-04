from pathlib import Path
from types import SimpleNamespace

from transync.config import TransyncConfig
from transync.translator import _model_source


def test_config_resolves_default_artifact_paths() -> None:
    config = TransyncConfig.from_file(Path("configs/transync.json"))

    assert config.model_dir.name == "transync"
    assert config.hf_repo == "facebook/mbart-large-50-many-to-many-mmt"
    assert config.sentencepiece_model.name == "transync_spm.model"
    assert config.dictionary_path.name == "fairseq_dict.json"
    assert config.generation.beam_size == 5
    assert config.generation.max_new_tokens == 256


def test_model_source_prefers_local_directory_when_available(tmp_path: Path) -> None:
    config = SimpleNamespace(model_dir=tmp_path, hf_repo="facebook/mbart-large-50-many-to-many-mmt")

    assert _model_source(config) == tmp_path
