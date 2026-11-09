import argparse
import logging
from pathlib import Path
import cv2
from detection.ort_detector import OrtDetector

def load_names(path):
    p = Path(path)
    return [line.strip() for line in p.read_text().splitlines() if line.strip()] if p.exists() else None

def main():
    parser = argparse.ArgumentParser(description="PPE detection using a custom YOLO ONNX model")
    parser.add_argument("--source", default="0", help="Video path, image path, or camera index")
    parser.add_argument("--model", default="models/ppe.onnx")
    parser.add_argument("--names", default="classes.txt")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.35)
    parser.add_argument("--output", default="runs/ppe_output.mp4")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    names = load_names(args.names)
    detector = OrtDetector(args.model, args.device, args.imgsz, args.conf, names)

    source = int(args.source) if args.source.isdigit() else args.source
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open source: {args.source}")
    is_camera = isinstance(source, int)
    writer = None
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    palette = [(255, 0, 255), (0, 255, 0), (255, 255, 0), (0, 165, 255)]
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        for det in detector.predict(frame):
            color = palette[det.class_id % len(palette)]
            cv2.rectangle(frame, (det.x1, det.y1), (det.x2, det.y2), color, 2)
            label = f"{det.class_name} {det.confidence:.2f}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
            y = max(0, det.y1 - th - 8)
            cv2.rectangle(frame, (det.x1, y), (det.x1 + tw + 6, y + th + 8), color, -1)
            cv2.putText(frame, label, (det.x1 + 3, y + th + 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2)
        if not is_camera and writer is None:
            fps = cap.get(cv2.CAP_PROP_FPS) or 25
            h, w = frame.shape[:2]
            writer = cv2.VideoWriter(args.output, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
        if writer is not None:
            writer.write(frame)
        cv2.imshow("PPE Detection - press q to quit", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cap.release()
    if writer is not None:
        writer.release()
        print(f"Saved output: {args.output}")
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
