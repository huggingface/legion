"""flock.registry: every harness and compute entry is complete and its code imports."""

import inspect

import pytest
from pydantic import ValidationError

from flock.controller.main import build
from flock.models import ComputeSpec, Harness, TypeSpec
from flock.registry import COMPUTE, HARNESSES, load


@pytest.mark.parametrize("name", HARNESSES)
def test_harness_entry_is_complete(name):
    h = HARNESSES[name]
    # harbor's also takes the worker's `host` (flock.worker.harbor)
    assert list(inspect.signature(load(h.adapter)).parameters)[:3] == ["workdir", "name", "spec"]
    assert h.instructions_file.endswith(".md")
    assert Harness(name) == name


@pytest.mark.parametrize("kind", COMPUTE)
def test_compute_entry_is_complete(kind):
    assert list(inspect.signature(load(COMPUTE[kind])).parameters) == ["env", "dep", "data_dir"]
    assert ComputeSpec(kind=kind).kind == kind


def test_unknown_names_keep_their_errors():
    with pytest.raises(ValidationError,
                       match="Input should be 'llm' or 'harbor'"):
        TypeSpec.model_validate({"harness": "nope"})
    with pytest.raises(ValidationError, match="Input should be 'hf-jobs', 'local' or 'slurm'"):
        ComputeSpec(kind="nope")
    with pytest.raises(SystemExit, match=r"FLOCK_COMPUTE='nope' is not one of hf-jobs, local, "
                                         r"slurm \(unset: hf-jobs\)"):
        build({"FLOCK_TOKEN": "t", "FLOCK_COMPUTE": "nope"})
