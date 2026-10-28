"""Unit tests for the pure seams of ``derive_dataset_improvement_config``.

The MLflow-dependent paths are covered by the integration tests
(``AIAgent/tests/integration/test_derive_dataset_improvement_config.py``);
here we pin the ``main()`` CLI seam and the config validation that fails
before an ``MlflowClient`` is created — no MLflow store needed.
"""

import pytest
import yaml

import derive_dataset_improvement_config
from derive_dataset_improvement_config import derive_config

pytestmark = pytest.mark.unit


def test_main_dispatches_to_derive_config(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    base = tmp_path / "base.yml"
    output = tmp_path / "derived" / "config.yml"
    calls: list[tuple] = []
    monkeypatch.setattr(
        derive_dataset_improvement_config,
        "derive_config",
        lambda base_config, output_config: calls.append((base_config, output_config)),
    )
    monkeypatch.setattr(
        "sys.argv",
        [
            "derive_dataset_improvement_config.py",
            "--base-config",
            str(base),
            "--output-config",
            str(output),
        ],
    )

    derive_dataset_improvement_config.main()

    assert calls == [(base, output)]


def test_derive_config_without_experiment_name_exits(tmp_path) -> None:
    # No tracking_uri in the section, so the failure happens before any
    # MLflow call.
    base = tmp_path / "base.yml"
    with open(base, "w") as file:
        yaml.safe_dump({"MLFlowConfig": {}}, file)
    output = tmp_path / "derived.yml"

    with pytest.raises(SystemExit, match="has no MLFlowConfig.experiment_name"):
        derive_config(base, output)

    assert not output.exists()
