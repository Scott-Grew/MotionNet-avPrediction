"""Tests for the training loop, covering the learning-rate schedule, resuming
and the Kaggle kernel's command line.
"""

import ast
import contextlib
import io
import unittest.mock
from pathlib import Path

import numpy as np
import pytest
import torch
from reference_implementations import unit_anchor_offsets_per_type

import train
from womd import checkpoint, contract, model
from womd.loader import MapArrays, SceneArrays, SceneBatch, TargetArrays
from womd.model import MotionPredictor

KERNEL_PATH = (
    Path(__file__).parent.parent
    / "data"
    / "kaggle_upload"
    / "kernel"
    / "run.py"
)


def build_synthetic_map_rows(scene_count, dot_count):
    """Random map rows whose two boundary-crossing columns hold valid codes."""
    map_rows = torch.randn(scene_count, dot_count, contract.MAP_FEATURE_DIM)
    code_shape = (scene_count, dot_count, 2)
    codes = torch.randint(0, contract.NUM_BOUNDARY_CROSSING_CODES, code_shape)
    map_rows[..., contract.MAP_LEFT_BOUNDARY_CROSSING :] = codes.float()
    return map_rows


def synthetic_batch():
    """Two scenes of three agents and ten map dots in two chunks, with one
    target each, agent 0 of its scene.
    """
    history = contract.HISTORY_STEPS
    signals = contract.NUM_TRAFFIC_SIGNAL_STATES
    dot_chunk = (torch.arange(10) // 5).expand(2, -1).clone()
    return SceneBatch(
        scene=SceneArrays(
            agent_history=torch.randn(
                2, 3, history, contract.AGENT_FEATURE_DIM
            ),
            agent_history_mask=torch.ones(2, 3, history, dtype=torch.bool),
            agent_signal_history=torch.zeros(2, 3, history, signals),
        ),
        map=MapArrays(
            rows=build_synthetic_map_rows(2, 10),
            dot_present=torch.ones(2, 10, dtype=torch.bool),
            dot_chunk=dot_chunk,
            chunk_pose=torch.randn(2, 2, 4),
            chunk_signal_history=torch.zeros(2, 2, history, signals),
        ),
        targets=TargetArrays(
            scene_index=torch.arange(2),
            track_index=torch.zeros(2, dtype=torch.long),
            agent_history=torch.randn(2, history, contract.AGENT_FEATURE_DIM),
            agent_history_mask=torch.ones(2, history, dtype=torch.bool),
            agent_signal_history=torch.zeros(2, history, signals),
            future_positions=torch.randn(2, contract.FUTURE_STEPS, 2),
            future_mask=torch.ones(2, contract.FUTURE_STEPS, dtype=torch.bool),
        ),
    )


def linear_training_objects():
    """A one-weight model with its optimiser and scaler, for checkpoints."""
    predictor = torch.nn.Linear(1, 1)
    optimizer = torch.optim.AdamW(predictor.parameters())
    scaler = train.GradScaler(enabled=False)
    return checkpoint.TrainingObjects(predictor, optimizer, scaler)


def test_warmup_rises_then_holds():
    """The rate climbs through warm-up, holds at the chosen rate, and reaches
    zero on the last step.
    """
    warmup_steps = 20
    schedule = train.Schedule(train.LEARNING_RATE, warmup_steps, None, None)
    rates = [schedule.at(step) for step in range(200)]
    assert rates[0] < rates[warmup_steps - 1]
    rising = zip(rates[:warmup_steps], rates[1:warmup_steps], strict=False)
    assert all(later >= earlier for earlier, later in rising)
    assert all(rate == train.LEARNING_RATE for rate in rates[warmup_steps:])
    lowered = train.Schedule(1e-4, warmup_steps, None, None)
    assert lowered.at(warmup_steps + 5) == 1e-4


def test_rate_holds_then_falls_to_zero():
    """Zero is reached only on the last step."""
    schedule = train.Schedule(1e-3, 20, 100, 200)
    rates = [schedule.at(step) for step in range(200)]
    assert all(rate == 1e-3 for rate in rates[20:100])
    falling = zip(rates[100:199], rates[101:200], strict=True)
    assert all(later < earlier for earlier, later in falling)
    assert rates[150] < 1e-3 / 2
    assert rates[198] > 0.0
    assert rates[199] == 0.0


def test_resuming_never_skips_a_completed_epoch():
    """A resumed run starts at the first epoch its checkpoint has not done."""
    trained = linear_training_objects()
    two_done = checkpoint.Progress(seed=0, completed_epochs=2, epoch_plan=None)
    three_done = checkpoint.Progress(
        seed=0, completed_epochs=3, epoch_plan=None
    )
    interrupted = checkpoint.checkpoint_of(trained, two_done)
    finished = checkpoint.checkpoint_of(trained, three_done)
    assert list(train.remaining_epochs(interrupted["completed_epochs"], 5)) == [
        2,
        3,
        4,
    ]
    assert list(train.remaining_epochs(finished["completed_epochs"], 5)) == [
        3,
        4,
    ]
    assert list(train.remaining_epochs(1, 1)) == []


def test_checkpoint_round_trips(tmp_path):
    """A checkpoint with a tampered version is rejected."""
    trained = linear_training_objects()
    progress = checkpoint.Progress(seed=0, completed_epochs=1, epoch_plan=None)
    state = checkpoint.checkpoint_of(trained, progress)
    checkpoint_path = tmp_path / "checkpoint.pt"
    torch.save(state, checkpoint_path)
    reloaded_state = checkpoint.load_checkpoint(checkpoint_path)
    assert reloaded_state["code_version"] == contract.STAGING_CODE_VERSION
    assert torch.equal(
        reloaded_state["model_state"]["weight"],
        trained.predictor.state_dict()["weight"],
    )
    tampered_state = dict(state)
    tampered_state["code_version"] = "not-a-real-version"
    tampered_path = tmp_path / "tampered.pt"
    torch.save(tampered_state, tampered_path)
    with pytest.raises(ValueError):
        checkpoint.load_checkpoint(tampered_path)


def test_one_training_step_runs_end_to_end():
    """Forward, loss, backward and optimiser step."""
    torch.manual_seed(0)
    predictor = MotionPredictor(unit_anchor_offsets_per_type())
    optimizer = torch.optim.AdamW(
        train.parameter_groups(predictor), lr=train.LEARNING_RATE
    )
    step = train.training_step(predictor, synthetic_batch())
    total = step.losses.total
    optimizer.zero_grad()
    total.backward()
    optimizer.step()
    assert torch.isfinite(total)
    assert any(
        parameter.grad is not None and parameter.grad.abs().sum() > 0.0
        for parameter in predictor.parameters()
    )


def literal_run_arguments(call):
    """The literal arguments of one subprocess.run([...]) call, or None."""
    is_run = isinstance(call.func, ast.Attribute) and call.func.attr == "run"
    if not is_run or not call.args:
        return None
    argument_list = call.args[0]
    if not isinstance(argument_list, ast.List):
        return None
    literals = [
        element.value
        for element in argument_list.elts
        if isinstance(element, ast.Constant)
    ]
    return literals, len(argument_list.elts)


def kernel_train_arguments():
    """The literal train.py arguments in the Kaggle kernel script, and how many
    arguments the call has in total.
    """
    if not KERNEL_PATH.exists():
        pytest.skip(f"no Kaggle kernel script at {KERNEL_PATH}")
    kernel_tree = ast.parse(KERNEL_PATH.read_text())
    for node in ast.walk(kernel_tree):
        if not isinstance(node, ast.Call):
            continue
        found = literal_run_arguments(node)
        if found is None:
            continue
        literals, element_count = found
        if "train.py" in literals:
            launcher_count = literals.index("train.py") - 1
            return literals[launcher_count:], element_count - launcher_count
    raise AssertionError(f"no train.py invocation found in {KERNEL_PATH}")


def completed_flags(flag_arguments, value_of_flag):
    """The kernel's flags with the test's own values filled in for the flags
    in value_of_flag.
    """
    completed = []
    skip_next = False
    for position, argument in enumerate(flag_arguments):
        if skip_next:
            skip_next = False
            continue
        if argument in value_of_flag:
            completed.extend([argument, value_of_flag[argument]])
            has_value = position + 1 < len(flag_arguments)
            if has_value and not flag_arguments[position + 1].startswith("--"):
                skip_next = True
            continue
        completed.append(argument)
    return completed


def test_kernel_command_line_parses(tmp_path):
    """Parsed by train.py's real argument parser."""
    literal_arguments, element_count = kernel_train_arguments()
    flag_arguments = literal_arguments[2:]
    while flag_arguments and not flag_arguments[0].startswith("--"):
        flag_arguments = flag_arguments[1:]
    anchors_path = tmp_path / "training_anchors.npz"
    anchor_shape = (contract.NUM_OBJECT_TYPES, model.QUERY_COUNT, 2)
    np.savez(
        anchors_path,
        unit_anchors=np.zeros(anchor_shape, dtype=np.float32),
        provenance=contract.artifact_provenance(
            "test", "kernel-interface-test"
        ),
    )
    value_of_flag = {
        "--anchors": str(anchors_path),
        "--stop-after-seconds": "1.0",
    }
    completed_arguments = completed_flags(flag_arguments, value_of_flag)
    non_literal_count = element_count - len(literal_arguments)
    if "--anchors" not in flag_arguments:
        completed_arguments.extend(["--anchors", str(anchors_path)])
    if "--stop-after-seconds" not in flag_arguments and non_literal_count:
        completed_arguments.extend(["--stop-after-seconds", "1.0"])
    argv = [
        "train.py",
        str(tmp_path),
        str(tmp_path / "checkpoint.pt"),
    ] + completed_arguments

    patched_argv = unittest.mock.patch("sys.argv", argv)
    patched_planning = unittest.mock.patch.object(train, "steps_per_epoch")
    with patched_argv, patched_planning as blocked:
        blocked.side_effect = AssertionError(
            "parsing must fail before any work starts"
        )
        error_output = io.StringIO()
        try:
            with contextlib.redirect_stderr(error_output):
                train.TrainingRun(train.parse_arguments()).run()
        except AssertionError as expected_stop:
            assert "parsing must fail before any work starts" in str(
                expected_stop
            )
        except SystemExit as parse_failure:
            raise AssertionError(
                "the kernel's train.py command line does not parse: "
                f"{error_output.getvalue()}"
            ) from parse_failure
