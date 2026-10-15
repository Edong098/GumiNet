from pathlib import Path

import numpy as np
import torch

from guminet.engine.checkpoint import model_from_checkpoint
from guminet.utils import save_json


def onnx_predict(path, inputs):
    try:
        import onnxruntime as ort
    except ImportError as exc:
        raise ImportError("Install guminet[export] for ONNX Runtime") from exc
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    return session.run(None, {session.get_inputs()[0].name: np.asarray(inputs, dtype=np.float32)})[0]


def export_onnx(checkpoint, output, input_size=None, validate=True, atol=1e-4, rtol=1e-3):
    try:
        import onnx
    except ImportError as exc:
        raise ImportError("Install guminet[export] to export ONNX") from exc
    model, metadata = model_from_checkpoint(checkpoint)
    model.eval()
    size = input_size or metadata["input_size"]
    x = torch.randn(2, 3, size, size)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    # Legacy exporter intentionally avoids an additional onnxscript dependency.
    torch.onnx.export(
        model,
        x,
        str(output),
        input_names=["images"],
        output_names=["logits"],
        dynamic_axes={"images": {0: "batch", 2: "height", 3: "width"}, "logits": {0: "batch"}},
        opset_version=17,
        dynamo=False,
    )
    onnx.checker.check_model(onnx.load(str(output)))
    report = dict(path=str(output), opset=17, validated=False, input_size=size)
    if validate:
        errors = []
        for batch, resolution in ((2, size), (1, size + 16)):
            example = torch.randn(batch, 3, resolution, resolution)
            with torch.inference_mode():
                expected = model(example).numpy()
            actual = onnx_predict(output, example.numpy())
            np.testing.assert_allclose(actual, expected, atol=atol, rtol=rtol)
            errors.append(float(np.max(np.abs(actual - expected))))
        report.update(validated=True, max_absolute_error=max(errors), atol=atol, rtol=rtol)
    save_json(
        str(output) + ".json",
        dict(
            **report,
            class_to_idx=metadata["class_to_idx"],
            mean=metadata["mean"],
            std=metadata["std"],
            training_input_size=metadata["input_size"],
        ),
    )
    return report


def quantize_onnx(source, output):
    """Experimental dynamic MatMul/Gemm INT8; Conv stays FP32 for CPU compatibility."""
    from onnxruntime.quantization import QuantType, quantize_dynamic

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    quantize_dynamic(
        str(source), str(output), weight_type=QuantType.QInt8, op_types_to_quantize=["MatMul", "Gemm"]
    )
    return output


def validate_quantized(source, quantized, input_size=160, repeats=5, atol=0.1, rtol=0.1):
    errors = []
    for _ in range(repeats):
        x = np.random.default_rng(_).normal(size=(1, 3, input_size, input_size)).astype(np.float32)
        reference, actual = onnx_predict(source, x), onnx_predict(quantized, x)
        np.testing.assert_allclose(actual, reference, atol=atol, rtol=rtol)
        errors.append(float(np.max(np.abs(actual - reference))))
    report = dict(
        max_absolute_error=max(errors),
        samples=repeats,
        atol=atol,
        rtol=rtol,
        source_bytes=Path(source).stat().st_size,
        quantized_bytes=Path(quantized).stat().st_size,
    )
    save_json(str(quantized) + ".validation.json", report)
    return report
