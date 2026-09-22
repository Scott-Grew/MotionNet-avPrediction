"""Tests for the training loop, covering the learning-rate schedule, resuming
and the Kaggle kernel's command line.
"""
from pathlib import Path

import pytest
import torch

import train
from reference_implementations import unit_anchor_offsets_per_type
from womd import checkpoint, contract, model
from womd.loader import MapArrays, SceneArrays, SceneBatch, TargetArrays
from womd.model import MotionPredictor


def build_synthetic_map_rows(*, dot_count):
    """Random map rows whose two boundary-crossing columns hold valid codes."""
    map_rows = torch.randn(dot_count, contract.MAP_FEATURE_DIM)
    map_rows[:, contract.MAP_LEFT_BOUNDARY_CROSSING:] = torch.randint(
        0, contract.NUM_BOUNDARY_CROSSING_CODES, (dot_count, 2)).float()
    return map_rows


def synthetic_batch():
    """A random two-target batch with three scene agents and four map chunks."""
    return SceneBatch(
        scene=SceneArrays(
            agent_history=torch.randn(2, 3, contract.HISTORY_STEPS,
                                      contract.AGENT_FEATURE_DIM),
            agent_history_mask=torch.ones(2,
                                          3,
                                          contract.HISTORY_STEPS,
                                          dtype=torch.bool),
            agent_signal_history=torch.zeros(
                2,
                3,
                contract.HISTORY_STEPS,
                contract.NUM_TRAFFIC_SIGNAL_STATES,
            ),
        ),
        map=MapArrays(
            rows=build_synthetic_map_rows(dot_count=20),
            dot_chunk_slot=torch.arange(20) // 5,
            chunk_signal_history=torch.zeros(
                2,
                4,
                contract.HISTORY_STEPS,
                contract.NUM_TRAFFIC_SIGNAL_STATES,
            ),
        ),
        targets=TargetArrays(
            scene_index=torch.arange(2),
            agent_history=torch.randn(2, contract.HISTORY_STEPS,
                                      contract.AGENT_FEATURE_DIM),
            agent_history_mask=torch.ones(2,
                                          contract.HISTORY_STEPS,
                                          dtype=torch.bool),
            agent_signal_history=torch.zeros(
                2,
                contract.HISTORY_STEPS,
                contract.NUM_TRAFFIC_SIGNAL_STATES,
            ),
            token_visible=torch.ones(2, 3 + 4, dtype=torch.bool),
            token_pose=torch.randn(2, 3 + 4, 4),
            future_positions=torch.randn(2, contract.FUTURE_STEPS, 2),
            future_mask=torch.ones(2, contract.FUTURE_STEPS, dtype=torch.bool),
        ),
    )


def test_warmup_rises_then_holds():
    """The rate climbs through warm-up, holds at the chosen rate, and reaches
    zero on the last step.
    """
    warmup_steps = 20
    rates = [
        train.scheduled_learning_rate(step, warmup_steps=warmup_steps)
        for step in range(200)
    ]
    assert rates[0] < rates[warmup_steps - 1]
    assert all(
        later >= earlier
        for earlier, later in zip(rates[:warmup_steps], rates[1:warmup_steps]))
    assert all(rate == train.LEARNING_RATE for rate in rates[warmup_steps:])
    assert (train.scheduled_learning_rate(warmup_steps + 5,
                                          warmup_steps=warmup_steps,
                                          learning_rate=1e-4) == 1e-4)


def test_rate_holds_then_falls_to_zero():
    """Zero is reached only on the last step."""
    rates = [
        train.scheduled_learning_rate(
            step,
            warmup_steps=20,
            learning_rate=1e-3,
            decay_start_step=100,
            decay_end_step=200,
        ) for step in range(200)
    ]
    assert all(rate == 1e-3 for rate in rates[20:100])
    assert all(later < earlier
               for earlier, later in zip(rates[100:199], rates[101:200]))
    assert rates[150] < 1e-3 / 2
    assert rates[198] > 0.0
    assert rates[199] == 0.0


def test_resuming_never_skips_a_completed_epoch():
    """A resumed run starts at the first epoch the checkpoint has not finished.
    """
    predictor = torch.nn.Linear(1, 1)
    optimizer = torch.optim.AdamW(predictor.parameters())
    scaler = train.GradScaler(enabled=False)
    interrupted = checkpoint.checkpoint_state(predictor,
                                              optimizer,
                                              scaler,
                                              seed=0,
                                              completed_epochs=2)
    finished = checkpoint.checkpoint_state(predictor,
                                           optimizer,
                                           scaler,
                                           seed=0,
                                           completed_epochs=3)
    assert list(
        train.epochs_left_to_train(
            completed_epochs=interrupted["completed_epochs"],
            requested_epochs=5)) == [2, 3, 4]
    assert list(
        train.epochs_left_to_train(
            completed_epochs=finished["completed_epochs"],
            requested_epochs=5)) == [3, 4]
    assert list(
        train.epochs_left_to_train(completed_epochs=1,
                                   requested_epochs=1)) == []


def test_checkpoint_round_trips(tmp_path,):
    """A checkpoint with a tampered version is rejected."""
    predictor = torch.nn.Linear(1, 1)
    optimizer = torch.optim.AdamW(predictor.parameters())
    scaler = train.GradScaler(enabled=False)
    state = checkpoint.checkpoint_state(predictor,
                                        optimizer,
                                        scaler,
                                        seed=0,
                                        completed_epochs=1)
    checkpoint_path = tmp_path / "checkpoint.pt"
    torch.save(state, checkpoint_path)
    reloaded_state = checkpoint.load_checkpoint_state(checkpoint_path)
    assert reloaded_state["code_version"] == contract.STAGING_CODE_VERSION
    assert torch.equal(
        reloaded_state["model_state"]["weight"],
        predictor.state_dict()["weight"],
    )
    tampered_state = dict(state)
    tampered_state["code_version"] = "not-a-real-version"
    tampered_path = tmp_path / "tampered.pt"
    torch.save(tampered_state, tampered_path)
    with pytest.raises(AssertionError):
        checkpoint.load_checkpoint_state(tampered_path)


def test_one_training_step_runs_end_to_end():
    """Forward, loss, backward and optimiser step."""
    torch.manual_seed(0)
    predictor = MotionPredictor(unit_anchor_offsets_per_type())
    optimizer = torch.optim.AdamW(train.parameter_groups(predictor),
                                  lr=train.LEARNING_RATE)
    total, _, _, _, _ = train.training_losses(predictor, synthetic_batch())
    optimizer.zero_grad()
    total.backward()
    optimizer.step()
    assert torch.isfinite(total)
    assert any(parameter.grad is not None and parameter.grad.abs().sum() > 0.0
               for parameter in predictor.parameters())


def kernel_train_arguments():
    """The literal train.py arguments in the Kaggle kernel script, and how many
    arguments the call has in total.
    """
    import ast as ast_module

    kernel_path = (Path(__file__).parent.parent / "data" / "kaggle_upload" /
                   "kernel" / "run.py")
    if not kernel_path.exists():
        pytest.skip(f"no Kaggle kernel script at {kernel_path}")
    kernel_tree = ast_module.parse(kernel_path.read_text())
    for node in ast_module.walk(kernel_tree):
        if not isinstance(node, ast_module.Call):
            continue
        if not (isinstance(node.func, ast_module.Attribute) and
                node.func.attr == "run"):
            continue
        argument_list = node.args[0]
        if not isinstance(argument_list, ast_module.List):
            continue
        literal_arguments = [
            element.value
            for element in argument_list.elts
            if isinstance(element, ast_module.Constant)
        ]
        if "train.py" in literal_arguments:
            launcher_count = literal_arguments.index("train.py") - 1
            return (literal_arguments[launcher_count:],
                    len(argument_list.elts) - launcher_count)
    raise AssertionError(f"no train.py invocation found in {kernel_path}")


def test_kernel_command_line_parses(tmp_path,):
    """Parsed by train.py's real argument parser."""
    literal_arguments, element_count = kernel_train_arguments()
    flag_arguments = literal_arguments[2:]
    while flag_arguments and not flag_arguments[0].startswith("--"):
        flag_arguments = flag_arguments[1:]
    anchors_path = tmp_path / "training_anchors.npz"
    import numpy as np

    np.savez(
        anchors_path,
        unit_anchors=np.zeros(
            (contract.NUM_OBJECT_TYPES, model.QUERY_COUNT, 2),
            dtype=np.float32,
        ),
        provenance=contract.artifact_provenance("test",
                                                "kernel-interface-test"),
    )
    completed_arguments = []
    value_of_flag = {
        "--anchors": str(anchors_path),
        "--stop-after-seconds": "1.0",
    }
    skip_next = False
    for position, argument in enumerate(flag_arguments):
        if skip_next:
            skip_next = False
            continue
        if argument in value_of_flag:
            completed_arguments.extend([argument, value_of_flag[argument]])
            if position + 1 < len(flag_arguments) and not flag_arguments[
                    position + 1].startswith("--"):
                skip_next = True
            continue
        completed_arguments.append(argument)
    non_literal_count = element_count - len(literal_arguments)
    if "--anchors" not in flag_arguments:
        completed_arguments.extend(["--anchors", str(anchors_path)])
    if "--stop-after-seconds" not in flag_arguments and non_literal_count:
        completed_arguments.extend(["--stop-after-seconds", "1.0"])
    parser_arguments = [
        str(tmp_path),
        str(tmp_path / "checkpoint.pt"),
    ] + completed_arguments
    import unittest.mock

    with unittest.mock.patch("sys.argv", ["train.py"] +
                             parser_arguments), unittest.mock.patch.object(
                                 train, "optimiser_steps_per_epoch") as blocked:
        blocked.side_effect = AssertionError(
            "parsing must fail before any work starts")
        import io, contextlib

        error_output = io.StringIO()
        try:
            with contextlib.redirect_stderr(error_output):
                train.main()
        except AssertionError as expected_stop:
            assert "parsing must fail before any work starts" in str(
                expected_stop)
        except SystemExit as parse_failure:
            raise AssertionError(
                "the kernel's train.py command line does not parse: "
                f"{error_output.getvalue()}") from parse_failure
