from pathlib import Path

import pytest
import torch

import train
from womd import contract, loss, model, pipeline
from womd.model import MotionPredictor, unit_anchor_offsets_per_type


def build_synthetic_map_rows(dot_count):
    map_rows = torch.randn(dot_count, contract.MAP_FEATURE_DIM)
    map_rows[:, contract.MAP_LEFT_BOUNDARY_CROSSING:] = torch.randint(
        0, contract.NUM_BOUNDARY_CROSSING_CODES, (dot_count, 2)
    ).float()
    return map_rows


def test_warmup_rises_to_the_learning_rate_holds_it_then_decays_linearly_to_zero():
    total_steps = 200
    warmup_steps = 20
    decay_steps = 50
    rates = [
        train.scheduled_learning_rate(step, warmup_steps, total_steps, decay_steps)
        for step in range(total_steps + 1)
    ]

    assert rates[0] < rates[warmup_steps - 1]
    assert rates[warmup_steps - 1] == train.LEARNING_RATE
    assert all(
        later >= earlier for earlier, later in zip(rates[:warmup_steps], rates[1:warmup_steps])
    )
    assert all(rate == train.LEARNING_RATE for rate in rates[warmup_steps:total_steps - decay_steps + 1])
    decaying = rates[total_steps - decay_steps:]
    assert all(later < earlier for earlier, later in zip(decaying, decaying[1:]))
    assert rates[total_steps - decay_steps + decay_steps // 2] == pytest.approx(
        0.5 * train.LEARNING_RATE
    )
    assert rates[total_steps - 1] == pytest.approx(train.LEARNING_RATE / decay_steps)
    assert rates[total_steps] == 0.0

    mid_run_step = warmup_steps + 10
    seconds_per_step = 2.0
    budget_seconds = 1000.0
    clock_window_seconds = decay_steps * seconds_per_step
    assert train.scheduled_learning_rate(
        mid_run_step, warmup_steps, total_steps, decay_steps,
        budget_seconds - clock_window_seconds - 1.0, budget_seconds, seconds_per_step,
    ) == train.LEARNING_RATE
    assert train.scheduled_learning_rate(
        mid_run_step, warmup_steps, total_steps, decay_steps,
        budget_seconds - 0.5 * clock_window_seconds, budget_seconds, seconds_per_step,
    ) == pytest.approx(0.5 * train.LEARNING_RATE)
    assert train.scheduled_learning_rate(
        mid_run_step, warmup_steps, total_steps, decay_steps,
        budget_seconds, budget_seconds, seconds_per_step,
    ) == 0.0
    assert train.scheduled_learning_rate(
        total_steps - 1, warmup_steps, total_steps, decay_steps,
        0.0, budget_seconds, seconds_per_step,
    ) == pytest.approx(train.LEARNING_RATE / decay_steps)


def test_resuming_retrains_the_interrupted_epoch_and_never_skips_a_completed_one():
    predictor = torch.nn.Linear(1, 1)
    optimizer = torch.optim.AdamW(predictor.parameters())
    scaler = train.GradScaler(enabled=False)
    interrupted = train.checkpoint_state(predictor, optimizer, scaler, 0, 2, 137)
    finished = train.checkpoint_state(predictor, optimizer, scaler, 0, 3, None)

    assert list(train.epochs_left_to_train(interrupted["completed_epochs"], 5)) == [2, 3, 4]
    assert list(train.epochs_left_to_train(finished["completed_epochs"], 5)) == [3, 4]
    assert list(train.epochs_left_to_train(0, 1)) == [0]
    assert list(
        train.epochs_left_to_train(
            train.checkpoint_state(predictor, optimizer, scaler, 0, 0, 137)["completed_epochs"], 1
        )
    ) == [0]
    assert list(
        train.epochs_left_to_train(
            train.checkpoint_state(predictor, optimizer, scaler, 0, 1, None)["completed_epochs"], 1
        )
    ) == []


def test_checkpoint_state_round_trips_through_the_shared_loader_and_rejects_a_tampered_version(
    tmp_path,
):
    predictor = torch.nn.Linear(1, 1)
    optimizer = torch.optim.AdamW(predictor.parameters())
    scaler = train.GradScaler(enabled=False)
    state = train.checkpoint_state(predictor, optimizer, scaler, 0, 1, None)
    checkpoint_path = tmp_path / "checkpoint.pt"
    torch.save(state, checkpoint_path)

    reloaded_state = model.load_checkpoint_state(checkpoint_path)
    assert reloaded_state["code_version"] == contract.STAGING_CODE_VERSION
    assert torch.equal(
        reloaded_state["model_state"]["weight"], predictor.state_dict()["weight"]
    )

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
    batch = {
        "agent_history": torch.randn(2, contract.HISTORY_STEPS, contract.AGENT_FEATURE_DIM),
        "agent_history_mask": torch.ones(2, contract.HISTORY_STEPS, dtype=torch.bool),
        "neighbour_history": torch.randn(2, 3, contract.HISTORY_STEPS, contract.AGENT_FEATURE_DIM),
        "neighbour_history_mask": torch.ones(2, 3, contract.HISTORY_STEPS, dtype=torch.bool),
        "neighbour_future_positions": torch.randn(2, 3, contract.FUTURE_STEPS, 2),
        "neighbour_future_mask": torch.ones(2, 3, contract.FUTURE_STEPS, dtype=torch.bool),
        "agent_signal_history": torch.zeros(
            2, contract.HISTORY_STEPS, contract.NUM_TRAFFIC_SIGNAL_STATES
        ),
        "neighbour_signal_history": torch.zeros(
            2, 3, contract.HISTORY_STEPS, contract.NUM_TRAFFIC_SIGNAL_STATES
        ),
        "map_rows": build_synthetic_map_rows(20),
        "map_dot_polyline_slot": torch.arange(20) // 5,
        "map_chunk_signal_history": torch.zeros(
            2, 4, contract.HISTORY_STEPS, contract.NUM_TRAFFIC_SIGNAL_STATES
        ),
        "map_chunk_lane_context": torch.randn(2, 4, contract.LANE_CONTEXT_DIM),
        "max_polylines_in_batch": torch.tensor(4),
        "future_positions": torch.randn(2, contract.FUTURE_STEPS, 2),
        "future_headings": torch.nn.functional.normalize(
            torch.randn(2, contract.FUTURE_STEPS, 2), dim=-1
        ),
        "future_mask": torch.ones(2, contract.FUTURE_STEPS, dtype=torch.bool),
    }

    (
        round_outputs, selected_unit_anchors, mode_valid,
        neighbour_future_positions, neighbour_log_standard_deviation,
    ) = predictor.predict_every_round(batch)
    total, _, _, _, _ = train.round_summed_prediction_loss(
        round_outputs, batch, selected_unit_anchors, mode_valid, 1.0, 1.0, 1.0
    )
    total = total + loss.neighbour_future_loss(
        neighbour_future_positions, neighbour_log_standard_deviation,
        batch["neighbour_future_positions"],
        batch["neighbour_future_mask"],
        batch["neighbour_history_mask"].any(dim=-1),
    )
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
    kernel_path = (
        Path(__file__).parent.parent.parent / "data" / "kaggle_upload" / "kernel" / "run.py"
    )
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
            element.value for element in argument_list.elts
            if isinstance(element, ast_module.Constant)
        ]
        if literal_arguments[:2] == ["python", "train.py"]:
            return literal_arguments, len(argument_list.elts)
    raise AssertionError(f"no train.py invocation found in {kernel_path}")


def test_the_kernel_command_line_parses_against_the_real_training_interface(tmp_path):
    literal_arguments, element_count = kernel_train_arguments()
    for weight_flag in (
        "--heading-loss-weight", "--classification-loss-weight",
        "--neighbour-future-loss-weight", "--speed-loss-weight",
    ):
        assert weight_flag in literal_arguments, (
            f"the kernel's train.py line omits {weight_flag}: a loss weight is a stated prior"
            f" and must be written on the command line, never inherited from a default"
        )
    flag_arguments = literal_arguments[2:]
    while flag_arguments and not flag_arguments[0].startswith("--"):
        flag_arguments = flag_arguments[1:]

    anchors_path = tmp_path / "training_anchors.npz"
    import numpy as np
    from womd import contract as contract_module
    np.savez(
        anchors_path,
        unit_anchors=np.zeros(
            (contract_module.NUM_OBJECT_TYPES, model.QUERY_COUNT, 2), dtype=np.float32
        ),
        anchor_counts=np.full(contract_module.NUM_OBJECT_TYPES, model.QUERY_COUNT, dtype=np.int64),
        provenance=contract_module.artifact_provenance("test", "kernel-interface-test"),
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
                position + 1
            ].startswith("--"):
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
    with unittest.mock.patch(
        "sys.argv", ["train.py"] + parser_arguments
    ), unittest.mock.patch.object(train, "optimiser_steps_per_epoch") as blocked:
        blocked.side_effect = AssertionError("parsing must fail before any work starts")
        import argparse
        parser = argparse.ArgumentParser()
        parser.add_argument("staged_directory", type=Path)
        parser.add_argument("checkpoint_path", type=Path)
        import io, contextlib
        error_output = io.StringIO()
        try:
            with contextlib.redirect_stderr(error_output):
                train.main()
        except AssertionError as expected_stop:
            assert "parsing must fail before any work starts" in str(expected_stop)
        except SystemExit as parse_failure:
            raise AssertionError(
                f"the kernel's train.py command line does not parse:"
                f" {error_output.getvalue()}"
            ) from parse_failure


def test_the_clock_stops_an_epoch_mid_way_and_leaves_a_resumable_checkpoint(tmp_path):
    import time
    torch.manual_seed(1)
    predictor = MotionPredictor(unit_anchor_offsets_per_type())
    optimizer = torch.optim.AdamW(train.parameter_groups(predictor), lr=train.LEARNING_RATE)
    scaler = train.GradScaler(enabled=False)
    batch = {
        "agent_history": torch.randn(2, contract.HISTORY_STEPS, contract.AGENT_FEATURE_DIM),
        "agent_history_mask": torch.ones(2, contract.HISTORY_STEPS, dtype=torch.bool),
        "neighbour_history": torch.randn(2, 3, contract.HISTORY_STEPS, contract.AGENT_FEATURE_DIM),
        "neighbour_history_mask": torch.ones(2, 3, contract.HISTORY_STEPS, dtype=torch.bool),
        "neighbour_future_positions": torch.randn(2, 3, contract.FUTURE_STEPS, 2),
        "neighbour_future_mask": torch.ones(2, 3, contract.FUTURE_STEPS, dtype=torch.bool),
        "agent_signal_history": torch.zeros(
            2, contract.HISTORY_STEPS, contract.NUM_TRAFFIC_SIGNAL_STATES
        ),
        "neighbour_signal_history": torch.zeros(
            2, 3, contract.HISTORY_STEPS, contract.NUM_TRAFFIC_SIGNAL_STATES
        ),
        "map_rows": build_synthetic_map_rows(20),
        "map_dot_polyline_slot": torch.arange(20) // 5,
        "map_chunk_signal_history": torch.zeros(
            2, 4, contract.HISTORY_STEPS, contract.NUM_TRAFFIC_SIGNAL_STATES
        ),
        "map_chunk_lane_context": torch.randn(2, 4, contract.LANE_CONTEXT_DIM),
        "max_polylines_in_batch": torch.tensor(4),
        "future_positions": torch.randn(2, contract.FUTURE_STEPS, 2),
        "future_headings": torch.nn.functional.normalize(
            torch.randn(2, contract.FUTURE_STEPS, 2), dim=-1
        ),
        "future_mask": torch.ones(2, contract.FUTURE_STEPS, dtype=torch.bool),
    }
    checkpoint_path = tmp_path / "checkpoint.pt"
    step_module, device_count = train.training_step_module(
        predictor, (1.0, 1.0, 1.0, 1.0), torch.device("cpu")
    )
    assert device_count == 1
    _, _, _, stopped_on_the_clock = train.train_epoch(
        predictor, step_module, optimizer, [batch, batch, batch], torch.device("cpu"), scaler,
        checkpoint_path, tmp_path / "checkpoint.pt.previous",
        3600, 2, 0,
        0, 1, 10, 1, float("inf"),
        time.perf_counter(), 0.0, 0,
    )
    assert stopped_on_the_clock
    checkpoint = model.load_checkpoint_state(checkpoint_path)
    assert checkpoint["completed_epochs"] == 2
    assert checkpoint["batch_index"] == 1


def test_splitting_a_batch_by_samples_reproduces_the_whole_batch_step():
    torch.manual_seed(3)
    predictor = MotionPredictor(unit_anchor_offsets_per_type()).eval()
    sample_count = 4
    chunks = 3
    dots_per_chunk = 5
    map_rows = build_synthetic_map_rows(sample_count * chunks * dots_per_chunk)
    batch = {
        "agent_history": torch.randn(sample_count, contract.HISTORY_STEPS, contract.AGENT_FEATURE_DIM),
        "agent_history_mask": torch.ones(sample_count, contract.HISTORY_STEPS, dtype=torch.bool),
        "neighbour_history": torch.randn(sample_count, 2, contract.HISTORY_STEPS, contract.AGENT_FEATURE_DIM),
        "neighbour_history_mask": torch.ones(sample_count, 2, contract.HISTORY_STEPS, dtype=torch.bool),
        "neighbour_future_positions": torch.randn(sample_count, 2, contract.FUTURE_STEPS, 2),
        "neighbour_future_mask": torch.ones(sample_count, 2, contract.FUTURE_STEPS, dtype=torch.bool),
        "agent_signal_history": torch.rand(sample_count, contract.HISTORY_STEPS, contract.NUM_TRAFFIC_SIGNAL_STATES),
        "neighbour_signal_history": torch.rand(sample_count, 2, contract.HISTORY_STEPS, contract.NUM_TRAFFIC_SIGNAL_STATES),
        "map_rows": map_rows,
        "map_dot_polyline_slot": torch.arange(sample_count * chunks * dots_per_chunk) // dots_per_chunk,
        "map_chunk_signal_history": torch.rand(sample_count, chunks, contract.HISTORY_STEPS, contract.NUM_TRAFFIC_SIGNAL_STATES),
        "map_chunk_lane_context": torch.randn(sample_count, chunks, contract.LANE_CONTEXT_DIM),
        "max_polylines_in_batch": torch.tensor(chunks),
        "future_positions": torch.randn(sample_count, contract.FUTURE_STEPS, 2),
        "future_headings": torch.nn.functional.normalize(torch.randn(sample_count, contract.FUTURE_STEPS, 2), dim=-1),
        "future_mask": torch.ones(sample_count, contract.FUTURE_STEPS, dtype=torch.bool),
    }
    step = train.TrainingStep(predictor, 1.0, 1.0, 1.0, 1.0)
    with torch.no_grad():
        whole = train.combine_step_outputs(step(batch))
        parts = [step(part) for part in pipeline.split_batch_by_samples(batch, 2)]
        gathered = {name: torch.cat([part[name] for part in parts], dim=0) for name in parts[0]}
        halves = train.combine_step_outputs(gathered)
    assert [part["agent_history"].shape[0] for part in pipeline.split_batch_by_samples(batch, 2)] == [2, 2]
    assert torch.allclose(halves["trajectories"], whole["trajectories"], atol=1e-4)
    assert torch.equal(halves["mode_valid"], whole["mode_valid"])
    for name in train.LOSS_COMPONENT_NAMES:
        assert float(halves[name]) == pytest.approx(float(whole[name]), rel=1e-4), name
