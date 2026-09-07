from __future__ import annotations

import copy
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import pytest

import desksense.tapness_v2_provisional as provisional
from desksense.tapness import stage1_policy_metadata
from desksense.tapness_v2_research import (
    TAPNESS_V2_RESEARCH_FEATURE_NAMES,
    tapness_v2_research_feature_schema,
)


ARTIFACT_PATH = Path(
    "research_baselines/lenovo-tapness-v2-provisional-ab.json"
)
SESSION_A_PATH = Path("datasets/20260901T131308.362205Z-f9e2b1ec")
SESSION_B_PATH = Path("datasets/20260904T171453.458221Z-4ecb16ad")


@pytest.fixture
def artifact() -> dict:
    return provisional.load_provisional_tapness_v2_artifact(ARTIFACT_PATH)


def test_tracked_artifact_is_explicitly_provisional_research_only(
    artifact: dict,
) -> None:
    assert artifact["artifact_type"] == (
        "desksense_provisional_tapness_v2_research_baseline"
    )
    assert artifact["research_status"] == {
        "development_only": True,
        "provisional": True,
        "not_production": True,
        "not_external_validated": True,
        "not_live_default": True,
    }
    assert artifact["target"]["spatial_inputs_used"] is False
    assert artifact["evidence_boundary"][
        "development_replication_r1_used_for_fitting"
    ] is False


def test_artifact_freezes_exact_two_feature_schema_and_stage1_policy(
    artifact: dict,
) -> None:
    assert artifact["feature_schema"] == tapness_v2_research_feature_schema()
    assert artifact["model"]["ordered_feature_names"] == list(
        TAPNESS_V2_RESEARCH_FEATURE_NAMES
    )
    assert artifact["required_stage1_policy"] == stage1_policy_metadata()
    assert [
        item["transformation"] for item in artifact["feature_schema"]["features"]
    ] == ["natural_log_with_fixed_epsilon", "natural_log_with_fixed_epsilon"]
    assert [item["log_epsilon"] for item in artifact["feature_schema"]["features"]] == [
        1.0e-12,
        1.0e-12,
    ]
    forbidden = {
        "peak_ratio_db_ch2_minus_ch1",
        "predicted_zone",
        "spatial_margin",
    }
    assert forbidden.isdisjoint(artifact["model"]["ordered_feature_names"])


def test_artifact_binds_exact_ab_sources_and_183_memberships(artifact: dict) -> None:
    sources = artifact["source_development_datasets"]
    assert [(item["session_id"], item["dataset_fingerprint_sha256"]) for item in sources] == [
        (provisional.SESSION_A_ID, provisional.SESSION_A_FINGERPRINT),
        (provisional.SESSION_B_ID, provisional.SESSION_B_FINGERPRINT),
    ]
    membership = artifact["training_membership"]
    assert membership["positive_candidate_count"] == 59
    assert membership["negative_candidate_count"] == 124
    assert membership["total_candidate_count"] == 183
    assert len(set(membership["positive_candidate_ids"] + membership["negative_candidate_ids"])) == 183
    assert membership["r1_candidates_used"] is False


def test_artifact_freezes_expected_all_ab_fit_and_threshold(artifact: dict) -> None:
    model = artifact["model"]
    assert model["l2_regularization"] == 0.01
    assert model["standardization"]["means"] == pytest.approx(
        [5.698220898643634, -4.055024572592979]
    )
    assert model["standardization"]["scales"] == pytest.approx(
        [1.229464637607929, 1.0538236386850006]
    )
    assert model["coefficients"] == pytest.approx(
        [-2.3350101420675204, -2.1605692696663525]
    )
    assert model["intercept"] == pytest.approx(-3.1820820485112873)
    assert model["decision_threshold"] == pytest.approx(0.6786616454344989)
    selection = model["threshold_selection"]
    assert selection["minimum_positive_candidate_recall"] == 0.95
    assert selection["selected_positive_candidates"] == 59
    assert selection["selected_negative_false_accepts"] == 3
    assert all(
        fold["training_only_standardization"]["means"]
        and fold["training_only_standardization"]["scales"]
        for fold in artifact["development_selection"]["grouped_oof_folds"]
    )


def test_artifact_generation_reproduces_semantics_from_exact_local_ab(
    artifact: dict,
) -> None:
    if not SESSION_A_PATH.exists() or not SESSION_B_PATH.exists():
        pytest.skip("Local ignored Phase 3B development datasets are unavailable.")
    created = datetime.fromisoformat(artifact["created_at_utc"].replace("Z", "+00:00"))

    regenerated = provisional.create_provisional_tapness_v2_artifact(
        SESSION_A_PATH,
        SESSION_B_PATH,
        now_fn=lambda: created,
    )

    assert regenerated == artifact


def test_provisional_inference_is_deterministic_and_tie_inclusive(
    artifact: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    features = {
        "spectral_bandwidth_hz": 1000.0,
        "post_0_100_zero_crossing_rate": 0.1,
    }
    first = provisional.classify_with_provisional_tapness_v2(features, artifact)
    second = provisional.classify_with_provisional_tapness_v2(features, artifact)
    assert first == second
    assert first["score_is_calibrated_probability"] is False

    monkeypatch.setattr(
        provisional.ResearchLogisticModel,
        "score",
        lambda self, values: __import__("numpy").array(
            [artifact["model"]["decision_threshold"]]
        ),
    )
    tied = provisional.classify_with_provisional_tapness_v2(features, artifact)
    assert tied["tap_accepted"] is True
    assert tied["predicted_label"] == "TAP"


@pytest.mark.parametrize(
    "mutation",
    [
        lambda value: value.update(artifact_type="wrong"),
        lambda value: value["feature_schema"].update(version=999),
        lambda value: value["required_stage1_policy"].update(version=999),
        lambda value: value["model"].update(coefficients=[1.0]),
        lambda value: value["model"].update(decision_threshold=float("nan")),
        lambda value: value["training_membership"].update(total_candidate_count=182),
        lambda value: value["research_status"].update(not_live_default=False),
    ],
)
def test_malformed_artifacts_are_rejected(artifact: dict, mutation) -> None:
    malformed = copy.deepcopy(artifact)
    mutation(malformed)
    with pytest.raises(provisional.ProvisionalTapnessV2Error):
        provisional.validate_provisional_tapness_v2_artifact(malformed)


def test_exclusive_artifact_write_refuses_overwrite(
    tmp_path: Path, artifact: dict
) -> None:
    path = tmp_path / "provisional.json"
    provisional.write_provisional_tapness_v2_artifact(artifact, path)
    first = path.read_bytes()
    assert first == ARTIFACT_PATH.read_bytes()
    with pytest.raises(provisional.ProvisionalTapnessV2Error, match="overwrite"):
        provisional.write_provisional_tapness_v2_artifact(artifact, path)
    assert path.read_bytes() == first


def test_exact_source_guard_rejects_replication_and_unrelated_external_roles() -> None:
    session_a = {"session_id": provisional.SESSION_A_ID, "evidence_role": "development"}
    session_b = {
        "session_id": provisional.SESSION_B_ID,
        "evidence_role": "external_validation",
    }
    provisional._validate_exact_training_sources(session_a, session_b)

    replication = dict(session_a, evidence_role="development_replication")
    with pytest.raises(provisional.ProvisionalTapnessV2Error):
        provisional._validate_exact_training_sources(replication, session_b)
    unrelated = dict(session_b, session_id="some-other-external-session")
    with pytest.raises(provisional.ProvisionalTapnessV2Error):
        provisional._validate_exact_training_sources(session_a, unrelated)


def test_importing_provisional_module_does_not_load_sounddevice() -> None:
    code = (
        "import sys; import desksense.tapness_v2_provisional; "
        "print('sounddevice' in sys.modules)"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        capture_output=True,
        text=True,
    )
    assert result.stdout.strip() == "False"


def test_tracked_artifact_is_finite_json_without_waveform_arrays() -> None:
    text = ARTIFACT_PATH.read_text(encoding="utf-8")
    value = json.loads(text)
    json.dumps(value, allow_nan=False)
    assert '"candidate_window"' not in text
    assert '"capture"' not in text
    assert '"raw_audio"' not in text
