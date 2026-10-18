"""ONNX export and ONNX Runtime inference (Phase 5).

export: converts a worker-produced scikit-learn classification or regression artifact over numeric features to ONNX with
skl2onnx, after the same environment check the scikit-learn runner applies before unpickling, and returns the native
pipeline's outputs on the given rows for a parity comparison. Anything else is refused as unsupported_conversion; no
model is ever altered to make a conversion succeed.

predict: runs an ONNX model with ONNX Runtime on CPU, one thread. Every model passes `inspect` first: bounded size, no
external data or local functions, only the default and ai.onnx.ml domains within fixed opset ranges, and only operators
on the allowlist below (subgraphs included). The allowlist is exactly what the supported exports emit.
"""
import base64
import sys

import numpy as np

import ml_runtime_common as common
from ml_runtime_common import RunnerError

OPSET, ML_OPSET = 17, 3
OPSET_RANGES = {"": (9, OPSET), "ai.onnx.ml": (1, ML_OPSET)}
MAX_ONNX_BYTES = 64 * 1024 * 1024
ALLOWED_OPS = {
    ("", op) for op in ("Add", "ArgMax", "Cast", "Concat", "Div", "Equal", "Flatten", "Identity", "Mul", "ReduceMean", "ReduceSum",
                        "ReduceSumSquare", "Reshape", "Scan", "Sqrt", "Sub", "TopK", "Transpose")
} | {
    ("ai.onnx.ml", op) for op in ("ArrayFeatureExtractor", "Imputer", "LinearClassifier", "LinearRegressor", "SVMClassifier", "SVMRegressor",
                                  "Scaler", "TreeEnsembleClassifier", "TreeEnsembleRegressor")
}


def refuse(message):
    raise RunnerError("onnx_model_refused", message)


def unsupported(message):
    raise RunnerError("unsupported_conversion", message)


def graphs(graph):
    yield graph
    for node in graph.node:
        for attribute in node.attribute:
            if attribute.HasField("g"):
                yield from graphs(attribute.g)
            for sub in attribute.graphs:
                yield from graphs(sub)


def inspect(data):
    """The runtime's gate. Returns the model's signature, opsets and operators, or refuses it."""
    import onnx
    if len(data) > MAX_ONNX_BYTES:
        refuse("ONNX model exceeds the size limit")
    try:
        model = onnx.load_from_string(data)
    except Exception:
        refuse("Unreadable ONNX model")
    if model.functions:
        refuse("ONNX local functions are not allowed")
    opsets = {o.domain or "": o.version for o in model.opset_import}
    for domain, version in opsets.items():
        bounds = OPSET_RANGES.get(domain)
        if bounds is None or not bounds[0] <= version <= bounds[1]:
            refuse("ONNX opset is outside the supported range")
    operators = set()
    for g in graphs(model.graph):
        tensors = [*g.initializer, *(s.values for s in g.sparse_initializer), *(s.indices for s in g.sparse_initializer)]
        for node in g.node:
            operators.add((node.domain or "", node.op_type))
            for attribute in node.attribute:
                tensors += [attribute.t] if attribute.HasField("t") else []
                tensors += list(attribute.tensors) + [s.values for s in attribute.sparse_tensors] + ([attribute.sparse_tensor.values] if attribute.HasField("sparse_tensor") else [])
        if any(t.data_location == onnx.TensorProto.EXTERNAL or len(t.external_data) for t in tensors):
            refuse("ONNX external data is not allowed")
    if not operators <= ALLOWED_OPS:
        refuse("ONNX model uses an operator outside the allowlist")
    try:
        onnx.checker.check_model(model, full_check=True)
    except Exception:
        refuse("ONNX model failed validation")
    initializers = {t.name for t in model.graph.initializer}
    inputs = [i for i in model.graph.input if i.name not in initializers]
    if len(inputs) != 1:
        refuse("ONNX model must have exactly one input")
    tensor = inputs[0].type.tensor_type
    dims = [d.dim_value if d.HasField("dim_value") else None for d in tensor.shape.dim]
    if tensor.elem_type != onnx.TensorProto.FLOAT or len(dims) != 2 or dims[1] is None:
        refuse("ONNX input must be a float tensor of shape [rows, features]")
    names = {onnx.TensorProto.FLOAT: "float32", onnx.TensorProto.DOUBLE: "float64", onnx.TensorProto.INT64: "int64", onnx.TensorProto.STRING: "string"}
    outputs = [{"name": o.name, "elemType": names.get(o.type.tensor_type.elem_type, "other")} for o in model.graph.output]
    return {"signature": {"inputs": [{"name": inputs[0].name, "elemType": "float32", "shape": [None, dims[1]]}], "outputs": outputs},
            "opsets": opsets, "operators": sorted(f"{d}:{op}" if d else op for d, op in operators), "irVersion": model.ir_version}


def export(request):
    import onnx
    import skl2onnx
    import sklearn
    from skl2onnx.common.data_types import FloatTensorType
    import sklearn_runner as mb
    if request.get("backend") != "sklearn":
        unsupported("Only scikit-learn artifacts can be exported")
    spec = request["snapshot"]
    algorithm = spec["algorithm"]
    task = mb.MODELS[algorithm][1] if algorithm in mb.MODELS else None
    if task not in ("classification", "regression"):
        unsupported("Only classification and regression models can be exported")
    types = mb.feature_types(spec)
    if any(t != "number" for t in types):
        unsupported("Only numeric feature columns can be exported")
    env = mb.environment()
    mb.check_compatible(request["expectedEnvironment"], env)
    model = mb.load_artifact(request["artifact"], env)
    pipeline, features = model["pipeline"], spec["pipeline"]["definition"]["features"]
    estimator = pipeline[-1]
    classes = getattr(estimator, "classes_", None)
    if task == "classification" and np.asarray(classes).dtype.kind not in "iuOU":
        unsupported("Only integer or string class labels can be exported")
    options = {id(estimator): {"zipmap": False}} if task == "classification" else None
    try:
        onx = skl2onnx.to_onnx(pipeline, initial_types=[("input", FloatTensorType([None, len(features)]))],
                               target_opset={"": OPSET, "ai.onnx.ml": ML_OPSET}, options=options)
    except Exception:
        unsupported("The fitted pipeline cannot be converted to ONNX")
    for key, value in sorted((request.get("metadata") or {}).items()):
        onx.metadata_props.add(key=str(key)[:100], value=str(value)[:500])
    data = onx.SerializeToString()
    described = inspect(data)
    x = mb.to_matrix(request["rows"], features, types)
    native = {"predictions": pipeline.predict(x).tolist(),
              "probabilities": pipeline.predict_proba(x).tolist() if task == "classification" and hasattr(estimator, "predict_proba") else None}
    return {"onnx": base64.b64encode(data).decode(), **described, "native": native,
            "classes": None if classes is None else np.asarray(classes).tolist(),
            "producer": {"converter": "skl2onnx", "skl2onnx": skl2onnx.__version__, "onnx": onnx.__version__, "sklearn": sklearn.__version__}}


def predict(request):
    import onnxruntime as ort
    try:
        data = base64.b64decode(request["onnx"], validate=True)
    except Exception:
        refuse("Unreadable ONNX model")
    described = inspect(data)
    features = request["features"]
    if described["signature"]["inputs"][0]["shape"][1] != len(features):
        refuse("ONNX input width does not match the model's features")
    rows = request["rows"]
    x = np.array([[np.nan if row[f] is None else float(row[f]) for f in features] for row in rows], dtype=np.float32).reshape(len(rows), len(features))
    options = ort.SessionOptions()
    options.intra_op_num_threads, options.inter_op_num_threads = 1, 1
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    session = ort.InferenceSession(data, sess_options=options, providers=["CPUExecutionProvider"])
    outputs = session.run(None, {described["signature"]["inputs"][0]["name"]: x})
    if request["task"] == "classification":
        probabilities = outputs[1].astype(float).tolist() if len(outputs) > 1 else None
        return {"predictions": np.asarray(outputs[0]).ravel().tolist(), "probabilities": probabilities, "runtime": {"onnxruntime": ort.__version__}}
    if request["task"] == "regression":
        return {"predictions": np.asarray(outputs[0], dtype=float).reshape(len(rows), -1)[:, 0].tolist(), "probabilities": None, "runtime": {"onnxruntime": ort.__version__}}
    unsupported("Unsupported ONNX task")


def execute(request):
    command = request["command"]
    if command == "export":
        return export(request)
    if command == "predict":
        return predict(request)
    raise ValueError("Unknown ONNX command")


if __name__ == "__main__":
    sys.exit(common.serve(execute))
