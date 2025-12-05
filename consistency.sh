for i in {1..10}; do pytest tests -k tmp | grep " ACTUAL"; done

# $ sh consistency.sh
#  ACTUAL: array([  0,   0,   0, ..., 255, 255, 255], shape=(4194304,), dtype=int32)
#  ACTUAL: array([  0,   0,   0, ..., 255, 255, 255], shape=(4194304,), dtype=int32)
#  ACTUAL: array([0, 0, 0, ..., 0, 0, 0], shape=(4194304,), dtype=int32)
#  ACTUAL: array([0, 0, 0, ..., 0, 0, 0], shape=(4194304,), dtype=int32)
#  ACTUAL: array([0, 0, 0, ..., 0, 0, 0], shape=(4194304,), dtype=int32)
#  ACTUAL: array([0, 0, 0, ..., 0, 0, 0], shape=(4194304,), dtype=int32)
#  ACTUAL: array([  0,   0,   0, ..., 255, 255, 255], shape=(4194304,), dtype=int32)
#  ACTUAL: array([0, 0, 0, ..., 0, 0, 0], shape=(4194304,), dtype=int32)
#  ACTUAL: array([0, 0, 0, ..., 0, 0, 0], shape=(4194304,), dtype=int32)
#  ACTUAL: array([  0,   0,   0, ...,   0, 255,   0], shape=(4194304,), dtype=int32)

