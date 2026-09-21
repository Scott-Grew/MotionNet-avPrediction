PYTHON ?= python3

SHARDS ?= data/validation_raw/validation.tfrecord-00000-of-00150
STAGED ?= data/staged
STAGED_TRAINING ?= data/staged_training
ANCHORS ?= data/training_anchors_54_sklearn.npz
CHECKPOINT ?= data/checkpoint.pt
PREDICTIONS ?= data/predictions.npz
BASELINE_PREDICTIONS ?= data/constant_velocity_predictions.npz
SCORE_DIRECTORY ?= data/score
ONNX ?= data/model.onnx

EPOCHS ?= 16
DECAY_FROM_EPOCH ?= 11
BATCH_SIZE ?= 64
WORKERS ?= 4
PREFETCH ?= 4
LEARNING_RATE ?= 5e-4
WARMUP_STEPS ?= 1000
GRADIENT_CLIP_NORM ?= 250.0
CHECKPOINT_EVERY_SECONDS ?= 300
STOP_AFTER_SECONDS ?= 42000

PROTO_CHECK_SCENARIOS ?= 100

EXPORT_DEVICE ?= cpu
EXPORT_SCENARIOS ?=
EXPORT_BUCKETS ?=
EXPORT_THREADS ?=

.DEFAULT_GOAL := help
.PHONY: help install protos test stage anchors train predict baseline \
	score check-protos export

help: ## list the targets
	@grep -E '^[a-z-]+:.*## ' $(MAKEFILE_LIST) | \
		awk -F ':.*## ' '{printf "  make %-14s %s\n", $$1, $$2}'
	@echo "  any path or number above can be set: make score STAGED=..."

install: ## install the pinned dependencies
	$(PYTHON) -m pip install -r requirements.txt

protos: ## generate womd_protos/ from proto/
	$(PYTHON) -m grpc_tools.protoc --proto_path=proto --python_out=. \
		proto/womd_protos/scenario.proto proto/womd_protos/map.proto
	touch womd_protos/__init__.py

test: export OMP_NUM_THREADS = 1
test: export MKL_NUM_THREADS = 1
test: export OPENBLAS_NUM_THREADS = 1
test: export VECLIB_MAXIMUM_THREADS = 1
test: export NUMEXPR_NUM_THREADS = 1
test: export MKL_THREADING_LAYER = SEQUENTIAL
test: export KMP_DUPLICATE_LIB_OK = TRUE
test: export KMP_INIT_AT_FORK = FALSE
test: womd_protos ## run the test suite, under 30 seconds
	$(PYTHON) -m pytest tests/ -q

womd_protos:
	$(MAKE) protos

stage: womd_protos ## turn raw WOMD shards into one .npz per scenario
	$(PYTHON) stage.py $(STAGED) $(SHARDS) --skip-existing

anchors: ## fit the 54 anchors per object type on the training set
	$(PYTHON) fit_anchors.py $(STAGED_TRAINING) $(ANCHORS)

train: ## train, resuming from the checkpoint if one exists
	$(PYTHON) train.py $(STAGED_TRAINING) $(CHECKPOINT) \
		--anchors $(ANCHORS) \
		--epochs $(EPOCHS) --decay-from-epoch $(DECAY_FROM_EPOCH) \
		--batch-size $(BATCH_SIZE) --workers $(WORKERS) \
		--prefetch $(PREFETCH) \
		--learning-rate $(LEARNING_RATE) \
		--warmup-steps $(WARMUP_STEPS) \
		--gradient-clip-norm $(GRADIENT_CLIP_NORM) \
		--checkpoint-every-seconds $(CHECKPOINT_EVERY_SECONDS) \
		--stop-after-seconds $(STOP_AFTER_SECONDS) \
		--mixed-precision --resume

predict: ## write the model's predictions for the staged scenarios
	$(PYTHON) submit.py $(CHECKPOINT) $(STAGED) $(ANCHORS) $(PREDICTIONS)

baseline: ## write the constant-velocity predictions
	$(PYTHON) submit.py --constant-velocity $(STAGED) \
		$(BASELINE_PREDICTIONS)

score: ## score the checkpoint with Waymo's metrics, in Docker
	$(PYTHON) scorer.py score $(CHECKPOINT) $(ANCHORS) $(STAGED) \
		$(SCORE_DIRECTORY)

check-protos: ## compare our protos with Waymo's, in Docker
	$(PYTHON) scorer.py check-reader $(firstword $(SHARDS)) \
		$(PROTO_CHECK_SCENARIOS)

export: ## export to ONNX, check it against torch, time both
	@test -n "$(EXPORT_SCENARIOS)" -a -n "$(EXPORT_BUCKETS)" \
		-a -n "$(EXPORT_THREADS)" || { echo "set EXPORT_SCENARIOS," \
		"EXPORT_BUCKETS and EXPORT_THREADS"; exit 1; }
	$(PYTHON) export_onnx.py $(STAGED) $(ONNX) --anchors $(ANCHORS) \
		--checkpoint $(CHECKPOINT) --scenarios $(EXPORT_SCENARIOS) \
		--buckets $(EXPORT_BUCKETS) --threads $(EXPORT_THREADS) \
		--device $(EXPORT_DEVICE)
