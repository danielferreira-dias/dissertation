"""Tests for the locally-testable parts of model_loader.

The full load_model_and_processor path requires HF + GPU and is verified
on a pod. These tests only cover `freeze_vision_tower` which is pure
string matching on parameter names.
"""
import torch

from fine_tune.model_loader import freeze_vision_tower


def make_fake_model_with_params(names: list[str]):
    class Fake:
        def __init__(self):
            self._params = {n: torch.nn.Parameter(torch.zeros(1)) for n in names}

        def named_parameters(self):
            return self._params.items()

    return Fake()


def test_freeze_vision_tower_disables_grad_on_matching_names():
    model = make_fake_model_with_params([
        "vision_tower.encoder.layers.0.weight",
        "vision_model.embeddings.patch_embedding.weight",
        "visual.blocks.0.attn.weight",
        "language_model.layers.0.self_attn.q_proj.weight",
        "model.embed_tokens.weight",
    ])
    frozen = freeze_vision_tower(model)
    # Three of the five match vision prefixes
    assert frozen == 3
    for name, p in model.named_parameters():
        if any(pref in name for pref in ("vision_tower", "vision_model", "visual")):
            assert not p.requires_grad, f"expected {name} frozen"
        else:
            assert p.requires_grad, f"expected {name} trainable"


def test_freeze_vision_tower_on_model_without_vision_params():
    """No-op if there are no vision-prefixed params (e.g., text-only model)."""
    model = make_fake_model_with_params([
        "model.embed_tokens.weight",
        "model.layers.0.self_attn.q_proj.weight",
    ])
    frozen = freeze_vision_tower(model)
    assert frozen == 0
    for _, p in model.named_parameters():
        assert p.requires_grad


def test_freeze_vision_tower_idempotent():
    """Calling twice doesn't break anything."""
    model = make_fake_model_with_params([
        "visual.blocks.0.attn.weight",
        "model.layers.0.self_attn.q_proj.weight",
    ])
    freeze_vision_tower(model)
    freeze_vision_tower(model)
    params = dict(model.named_parameters())
    assert not params["visual.blocks.0.attn.weight"].requires_grad
    assert params["model.layers.0.self_attn.q_proj.weight"].requires_grad
