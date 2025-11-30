import numpy as np
import coremltools as ct
from coremltools.converters.mil.mil import Builder as mb
from coremltools.converters.mil.mil import types

x_val = np.array([0, 4, 0, 4, 0, 0x1_0000, 0x1_0010], dtype=np.int32)
indices_val = np.array([5, 6], dtype=np.int32)

@mb.program(input_specs=[
    mb.TensorSpec(shape=x_val.shape, dtype=types.int32),
    mb.TensorSpec(shape=indices_val.shape, dtype=types.int32)
])
def prog(x, indices):
    out = mb.gather_along_axis(x=x, indices=indices, axis=0, name="output")
    return out

model = ct.convert(prog, source='milinternal', convert_to="mlprogram", compute_precision=ct.precision.FLOAT32)

prediction = model.predict({"x": x_val, "indices": indices_val})
result = prediction["output"]

print(f"{result[0]:#_x} {result[1]:#_x}")
