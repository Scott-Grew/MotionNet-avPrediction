import os

# Environment pins for single-threaded math libraries and quiet
# TensorFlow logging. Must be set before torch or TensorFlow import.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("KMP_INIT_AT_FORK", "FALSE")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("MKL_THREADING_LAYER", "SEQUENTIAL")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
