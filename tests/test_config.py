from pathlib import Path

from transync.config import TransyncConfig


def test_config_resolves_default_artifact_paths() -> None:
    config = TransyncConfig.from_file(Path("configs/transync.json"))

    assert config.model_dir.name == "transync"
    assert config.sentencepiece_model.name == "onemtv3b_spm.model"
    assert config.dictionary_path.name == "fairseq_dict.json"
    assert config.generation.beam_size == 5
    assert config.generation.max_new_tokens == 256
