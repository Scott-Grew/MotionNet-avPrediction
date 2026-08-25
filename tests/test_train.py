from pathlib import Path

import pytest
import torch

import train
from womd import contract, model
from womd.model import MotionPredictor, unit_anchor_offsets_per_type


def build_synthetic_map_rows(dot_count):
    map_rows = torch.randn(dot_count, contract.MAP_FEATURE_DIM)
    map_rows[:, contract.MAP_LEFT_BOUNDARY_CROSSING:] = torch.randint(
        0, contract.NUM_BOUNDARY_CROSSING_CODES, (dot_count, 2)
    ).float()
    return map_rows


def synthetic_batch():
    return {
        "agent_history": torch.randn(2, contract.HISTORY_STEPS, contract.AGENT_FEATURE_DIM),
        "agent_history_mask": torch.ones(2, contract.HISTORY_STEPS, dtype=torch.bool),
        "neighbour_history": torch.randn(2, 3, contract.HISTORY_STEPS, contract.AGENT_FEATURE_DIM),
        "neighbour_history_mask": torch.ones(2, 3, contract.HISTORY_STEPS, dtype=torch.bool),
        "agent_signal_history": torch.zeros(2, contract.HISTORY_STEPS, contract.NUM_TRAFFIC_SIGNAL_STATES),
        "neighbour_signal_history": torch.zeros(
            2, 3, contract.HISTORY_STEPS, contract.NUM_TRAFFIC_SIGNAL_STATES
        ),
        "map_rows": build_synthetic_map_rows(20),
        "map_dot_polyline_slot": torch.arange(20) // 5,
        "map_chunk_signal_history": torch.zeros(
            2, 4, contract.HISTORY_STEPS, contract.NUM_TRAFFIC_SIGNAL_STATES
        ),
        "max_polylines_in_batch": torch.tensor(4),
        "future_positions": torch.randn(2, contract.FUTURE_STEPS, 2),
        "future_mask": torch.ones(2, contract.FUTURE_STEPS, dtype=torch.bool),
    }


def test_warmup_rises_to_the_learning_rate_then_holds_it():
    warmup_steps = 20
    rates = [train.scheduled_learning_rate(step, warmup_steps) for step in range(200)]
    assert rates[0] < rates[warmup_steps - 1]
    assert all(later >= earlier for earlier, later in zip(rates[:warmup_steps], rates[1:warmup_steps]))
    assert all(rate == train.LEARNING_RATE for rate in rates[warmup_steps:])


def test_resuming_never_skips_a_completed_epoch():
    predictor = torch.nn.Linear(1, 1)
    optimizer = torch.optim.AdamW(predictor.parameters())
    scaler = train.GradScaler(enabled=False)
    interrupted = train.checkpoint_state(predictor, optimizer, scaler, 0, 2, 137)
    finished = train.checkpoint_state(predictor, optimizer, scaler, 0, 3, None)
    assert list(train.epochs_left_to_train(interrupted["completed_epochs"], 5)) == [2, 3, 4]
    assert list(train.epochs_left_to_train(finished["completed_epochs"], 5)) == [3, 4]
    assert list(train.epochs_left_to_train(1, 1)) == []


def test_checkpoint_state_round_trips_and_rejects_a_tampered_version(tmp_path):
    predictor = torch.nn.Linear(1, 1)
    optimizer = torch.optim.AdamW(predictor.parameters())
    scaler = train.GradScaler(enabled=False)
    state = train.checkpoint_state(predictor, optimizer, scaler, 0, 1, None)
    checkpoint_path = tmp_path / "checkpoint.pt"
    torch.save(state, checkpoint_path)
    reloaded_state = model.load_checkpoint_state(checkpoint_path)
    assert reloaded_state["code_version"] == contract.STAGING_CODE_VERSION
    assert torch.equal(reloaded_state["model_state"]["weight"], predictor.state_dict()["weight"])
    tampered_state = dict(state)
    tampered_state["code_version"] = "not-a-real-version"
    tampered_path = tmp_path / "tampered.pt"
    torch.save(tampered_state, tampered_path)
    with pytest.raises(AssertionError):
        model.load_checkpoint_state(tampered_path)


def test_one_training_step_runs_forward_loss_backward_and_optimizer_step():
    torch.manual_seed(0)
    predictor = MotionPredictor(unit_anchor_offsets_per_type())
    optimizer = torch.optim.AdamW(train.parameter_groups(predictor), lr=train.LEARNING_RATE)
    total, _, _, _, _ = train.training_losses(predictor, synthetic_batch())
    optimizer.zero_grad()
    total.backward()
    optimizer.step()
    assert torch.isfinite(total)
    assert any(
        parameter.grad is not None and parameter.grad.abs().sum() > 0.0
        for parameter in predictor.parameters()
    )


def kernel_train_arguments():
    import ast as ast_module
    kernel_path = Path(__file__).parent.parent.parent / "data" / "kaggle_upload" / "kernel" / "run.py"
    kernel_tree = ast_module.parse(kernel_path.read_text())
    for node in ast_module.walk(kernel_tree):
        if not isinstance(node, ast_module.Call):
            continue
        if not (isinstance(node.func, ast_module.Attribute) and node.func.attr == "run"):
            continue
        argument_list = node.args[0]
        if not isinstance(argument_list, ast_module.List):
            continue
        literal_arguments = [
            element.value for element in argument_list.elts if isinstance(element, ast_module.Constant)
        ]
        if literal_arguments[:2] == ["python", "train.py"]:
            return literal_arguments, len(argument_list.elts)
    raise AssertionError(f"no train.py invocation found in {kernel_path}")


def test_the_kernel_command_line_parses_against_the_real_training_interface(tmp_path):
    literal_arguments, element_count = kernel_train_arguments()
    flag_arguments = literal_arguments[2:]
    while flag_arguments and not flag_arguments[0].startswith("--"):
        flag_arguments = flag_arguments[1:]
    anchors_path = tmp_path / "training_anchors.npz"
    import numpy as np
    np.savez(
        anchors_path,
        unit_anchors=np.zeros((contract.NUM_OBJECT_TYPES, model.QUERY_COUNT, 2), dtype=np.float32),
        provenance=contract.artifact_provenance("test", "kernel-interface-test"),
    )
    completed_arguments = []
    value_of_flag = {"--anchors": str(anchors_path), "--stop-after-seconds": "1.0"}
    skip_next = False
    for position, argument in enumerate(flag_arguments):
        if skip_next:
            skip_next = False
            continue
        if argument in value_of_flag:
            completed_arguments.extend([argument, value_of_flag[argument]])
            if position + 1 < len(flag_arguments) and not flag_arguments[position + 1].startswith("--"):
                skip_next = True
            continue
        completed_arguments.append(argument)
    non_literal_count = element_count - len(literal_arguments)
    if "--anchors" not in flag_arguments:
        completed_arguments.extend(["--anchors", str(anchors_path)])
    if "--stop-after-seconds" not in flag_arguments and non_literal_count:
        completed_arguments.extend(["--stop-after-seconds", "1.0"])
    parser_arguments = [str(tmp_path), str(tmp_path / "checkpoint.pt")] + completed_arguments
    import unittest.mock
    with unittest.mock.patch("sys.argv", ["train.py"] + parser_arguments), unittest.mock.patch.object(
        train, "optimiser_steps_per_epoch"
    ) as blocked:
        blocked.side_effect = AssertionError("parsing must fail before any work starts")
        import io, contextlib
        error_output = io.StringIO()
        try:
            with contextlib.redirect_stderr(error_output):
                train.main()
        except AssertionError as expected_stop:
            assert "parsing must fail before any work starts" in str(expected_stop)
        except SystemExit as parse_failure:
            raise AssertionError(
                f"the kernel's train.py command line does not parse: {error_output.getvalue()}"
            ) from parse_failure
