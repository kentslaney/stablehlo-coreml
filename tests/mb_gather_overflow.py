import numpy as np
import coremltools as ct
from coremltools.converters.mil.mil import Builder as mb
from coremltools.converters.mil.mil import types

def check_size(n, target):
    @mb.program(input_specs=[
        mb.TensorSpec(shape=(n,), dtype=types.int32),
        mb.TensorSpec(shape=(1,), dtype=types.int32)
    ])
    def prog(x, indices):
        out = mb.gather_along_axis(x=x, indices=indices, axis=0, name="output")
        return out

    model = ct.convert(prog, source='milinternal', convert_to="mlprogram", compute_precision=ct.precision.FLOAT32)

    x_val = np.arange(n, dtype=np.int32)
    indices_val = np.array([target], dtype=np.int32)

    prediction = model.predict({"x": x_val, "indices": indices_val})
    result = prediction["output"]

    print(f"{n:#_x} {target:#_x} {result[0]:#_x}")

check_size(0x10_0010, 0x10_0008)