import onnx

model = onnx.load("model.onnx")
print(onnx.helper.printable_graph(model.graph))

import onnxruntime as ort

sess = ort.InferenceSession("model.onnx")
print("Inputs:")
for inp in sess.get_inputs():
    print(f"Name: {inp.name}, Shape: {inp.shape}, Type: {inp.type}")

print("Outputs:")
for out in sess.get_outputs():
    print(f"Name: {out.name}, Shape: {out.shape}, Type: {out.type}")

