from types import SimpleNamespace

import pytest
import torch

from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.utils.constants import OBS_STATE


def _policy_for_prepare_state(*, tactile_feature_keys=(), max_state_dim=32):
    policy = SmolVLAPolicy.__new__(SmolVLAPolicy)
    policy.config = SimpleNamespace(
        tactile_feature_keys=tactile_feature_keys,
        max_state_dim=max_state_dim,
    )
    return policy


def test_prepare_state_concatenates_configured_tactile_features():
    policy = _policy_for_prepare_state(
        tactile_feature_keys=("observation.tactile.force",),
        max_state_dim=10,
    )
    batch = {
        OBS_STATE: torch.tensor(
            [
                [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
                [[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]],
            ]
        ),
        "observation.tactile.force": torch.tensor(
            [
                [[0.1, 0.2], [0.3, 0.4]],
                [[0.5, 0.6], [0.7, 0.8]],
            ]
        ),
    }

    state = policy.prepare_state(batch)

    expected = torch.tensor(
        [
            [4.0, 5.0, 6.0, 0.3, 0.4, 0.0, 0.0, 0.0, 0.0, 0.0],
            [10.0, 11.0, 12.0, 0.7, 0.8, 0.0, 0.0, 0.0, 0.0, 0.0],
        ]
    )
    assert torch.allclose(state, expected)


def test_prepare_state_requires_configured_tactile_features():
    policy = _policy_for_prepare_state(
        tactile_feature_keys=("observation.tactile.force",),
        max_state_dim=10,
    )

    with pytest.raises(ValueError, match="missing from the batch"):
        policy.prepare_state({OBS_STATE: torch.ones(2, 3)})


def test_prepare_state_rejects_tactile_features_that_exceed_max_state_dim():
    policy = _policy_for_prepare_state(
        tactile_feature_keys=("observation.tactile.force",),
        max_state_dim=4,
    )
    batch = {
        OBS_STATE: torch.ones(2, 3),
        "observation.tactile.force": torch.ones(2, 2),
    }

    with pytest.raises(ValueError, match="exceeds config.max_state_dim"):
        policy.prepare_state(batch)
