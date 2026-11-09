import logging
from pathlib import Path
import onnxruntime as ort
from .preprocess import letterbox
from .yolo_decode import decode_yolo_output

log = logging.getLogger(__name__)

class OrtDetector:
    """Load a YOLO .onnx model and detect PPE on BGR frames."""
    def __init__(self, model_path, device="cpu", imgsz=640, confidence=0.35, names=None):
        path = Path(model_path)
        if not path.exists():
            raise FileNotFoundError(f"ONNX model not found: {path}. Add your trained PPE model.")
        providers = ["CPUExecutionProvider"]
        if device.lower() in ("cuda", "gpu"):
            available = ort.get_available_providers()
            if "CUDAExecutionProvider" in available:
                providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
            else:
                log.warning("CUDA EP unavailable; using CPU ONNX Runtime")
        self.session = ort.InferenceSession(str(path), providers=providers)
        inputs = self.session.get_inputs()
        if not inputs:
            raise RuntimeError("ONNX model has no inputs")
        self.input_name = inputs[0].name
        shape = inputs[0].shape
        if len(shape) == 4 and isinstance(shape[2], int) and isinstance(shape[3], int):
            self.imgsz = int(shape[2]) if shape[2] == shape[3] else imgsz
        else:
            self.imgsz = int(imgsz)
        self.confidence = confidence
        self.names = names
        outputs = self.session.get_outputs()
        self.output_name = outputs[0].name if outputs else None
        log.info("Loaded %s using %s", path, self.session.get_providers())

    def predict(self, frame):
        tensor, scale, pad_x, pad_y = letterbox(frame, self.imgsz)
        feeds = {self.input_name: tensor}
        outs = self.session.run([self.output_name], feeds) if self.output_name else self.session.run(None, feeds)
        return decode_yolo_output(outs[0], scale, pad_x, pad_y, frame.shape,
                                  confidence=self.confidence, names=self.names)
