import argparse
import json
import sys
from pathlib import Path

MAP_POLYGON_KINDS = ("crosswalk", "speed_bump", "driveway")


def track_state_row(state):
    return [
        bool(state.valid),
        float(state.center_x),
        float(state.center_y),
        float(state.heading),
        float(state.velocity_x),
        float(state.velocity_y),
        float(state.length),
        float(state.width),
    ]


def map_feature_points(feature):
    kind = feature.WhichOneof("feature_data")
    if kind is None:
        return None
    if kind == "stop_sign":
        raw_points = [feature.stop_sign.position]
    elif kind in MAP_POLYGON_KINDS:
        raw_points = list(getattr(feature, kind).polygon)
    else:
        raw_points = list(getattr(feature, kind).polyline)
    return kind, [
        [float(point.x), float(point.y)] for point in raw_points
    ]


def extract_scenario_fields(scenario):
    tracks = [
        {
            "id": int(track.id),
            "object_type": int(track.object_type),
            "states": [
                track_state_row(state) for state in track.states
            ],
        }
        for track in scenario.tracks
    ]

    map_features = []
    for feature in scenario.map_features:
        extracted = map_feature_points(feature)
        if extracted is None:
            continue
        kind, points = extracted
        map_features.append(
            {"id": int(feature.id), "kind": kind, "points": points}
        )

    return {
        "scenario_id": scenario.scenario_id,
        "current_time_index": int(scenario.current_time_index),
        "sdc_track_index": int(scenario.sdc_track_index),
        "timestamps_seconds": [
            float(value) for value in scenario.timestamps_seconds
        ],
        "tracks": tracks,
        "map_features": map_features,
    }


def extract_ours(shard_path, sample_count):
    from womd import tfrecord
    from womd_protos import scenario_pb2

    extracted_scenarios = []
    with open(shard_path, "rb") as stream:
        for payload in tfrecord.read_records(
            stream, verify_checksums=True
        ):
            if len(extracted_scenarios) >= sample_count:
                break
            scenario = scenario_pb2.Scenario()
            scenario.ParseFromString(payload)
            extracted_scenarios.append(
                extract_scenario_fields(scenario)
            )
    return extracted_scenarios


def extract_theirs(shard_path, sample_count):
    import tensorflow as tf
    from waymo_open_dataset.protos import scenario_pb2

    extracted_scenarios = []
    for raw_record in tf.data.TFRecordDataset(str(shard_path)):
        if len(extracted_scenarios) >= sample_count:
            break
        scenario = scenario_pb2.Scenario()
        scenario.ParseFromString(raw_record.numpy())
        extracted_scenarios.append(extract_scenario_fields(scenario))
    return extracted_scenarios


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--role", choices=("ours", "theirs"), required=True
    )
    parser.add_argument("--shard-path", type=Path, required=True)
    parser.add_argument("--sample-count", type=int, required=True)
    arguments = parser.parse_args()

    if arguments.role == "ours":
        extracted_scenarios = extract_ours(
            arguments.shard_path, arguments.sample_count
        )
    else:
        extracted_scenarios = extract_theirs(
            arguments.shard_path, arguments.sample_count
        )

    json.dump(extracted_scenarios, sys.stdout)


if __name__ == "__main__":
    main()
