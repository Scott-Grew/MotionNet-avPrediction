FROM --platform=linux/amd64 tensorflow/tensorflow:2.13.0

RUN apt-get update && apt-get install -y --no-install-recommends protobuf-compiler \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir --no-deps waymo-open-dataset-tf-2-12-0==1.6.7 \
    && pip install --no-cache-dir "protobuf<4" openexr immutabledict

WORKDIR /work

CMD ["python3", "-c", "import importlib.metadata; import tensorflow; from waymo_open_dataset.metrics.ops import py_metrics_ops; print('waymo-open-dataset', importlib.metadata.version('waymo-open-dataset-tf-2-12-0')); print('tensorflow', tensorflow.__version__); print('metrics ops loaded:', hasattr(py_metrics_ops, 'motion_metrics'))"]
