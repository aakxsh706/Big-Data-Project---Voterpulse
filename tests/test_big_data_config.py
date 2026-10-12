import pytest
from pydantic import ValidationError

from voterpulse.config import load_config


def test_big_data_is_opt_in_and_environment_is_validated():
    assert not load_config(env={}).settings.big_data.enabled
    cfg = load_config(env={
        "VOTERPULSE_BIG_DATA_ENABLED": "1",
        "VOTERPULSE_MONGO_URI": "mongodb://reader:password@localhost:27017",
        "VOTERPULSE_MONGO_DATABASE": "news",
        "VOTERPULSE_SPARK_PARTITIONS": "8",
    })
    assert cfg.settings.big_data.enabled
    assert cfg.settings.big_data.spark_partitions == 8
    assert cfg.settings.big_data.mongo_database == "news"
    assert "password" not in repr(cfg)


@pytest.mark.parametrize("env", [
    {"VOTERPULSE_MONGO_DATABASE": "same", "VOTERPULSE_MONGO_DEMO_DATABASE": "same"},
    {"VOTERPULSE_SPARK_PARTITIONS": "0"},
    {"VOTERPULSE_SYNC_BATCH_SIZE": "-2"},
    {"VOTERPULSE_MONGO_URI": "http://localhost"},
    {"VOTERPULSE_MONGO_DATABASE": "invalid/name"},
])
def test_invalid_big_data_environment_is_rejected(env):
    with pytest.raises(ValidationError):
        load_config(env=env)


def test_invalid_uri_is_not_exposed_in_validation_message():
    with pytest.raises(ValidationError) as raised:
        load_config(env={"VOTERPULSE_MONGO_URI": "http://person:private-password@localhost"})
    assert "private-password" not in str(raised.value)
