import numpy as np
from .schema import Detection

def _nms(boxes, scores, iou_threshold=0.45):
    if len(boxes) == 0:
        return []
    x1, y1, x2, y2 = boxes.T
    areas = np.maximum(0, x2-x1) * np.maximum(0, y2-y1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size:
        i = int(order[0]); keep.append(i)
        if order.size == 1:
            break
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        inter = np.maximum(0, xx2-xx1) * np.maximum(0, yy2-yy1)
        union = areas[i] + areas[order[1:]] - inter + 1e-7
        iou = inter / union
        order = order[1:][iou <= iou_threshold]
    return keep

def decode_yolo_output(output, scale, pad_x, pad_y, frame_shape,
                       confidence=0.35, iou=0.45, names=None):
    a = np.asarray(output)
    if a.ndim == 3:
        a = a[0]
    if a.ndim != 2:
        raise ValueError(f"Unsupported model output shape: {np.asarray(output).shape}")

    # End-to-end exports often return x1,y1,x2,y2,score,class_id.
    if a.shape[-1] == 6:
        rows = a
        boxes = rows[:, :4].astype(float)
        scores = rows[:, 4].astype(float)
        class_ids = rows[:, 5].astype(int)
        # Some exports return normalized xyxy.
        if boxes.size and np.nanmax(np.abs(boxes)) <= 1.5:
            boxes[:, [0, 2]] *= 640
            boxes[:, [1, 3]] *= 640
    else:
        # Convert channel-first YOLOv8 output to rows.
        if a.shape[0] < a.shape[1] and a.shape[0] <= 256:
            a = a.T
        if a.shape[1] < 5:
            raise ValueError(f"Cannot interpret YOLO output shape: {a.shape}")
        # YOLOv5: xywh,obj,class scores; YOLOv8: xywh,class scores.
        if a.shape[1] >= 6 and (a[:, 4].min() >= 0 and a[:, 4].max() <= 1):
            # Heuristic: YOLOv5 objectness usually present; for PPE models this
            # branch is selected when a distinct objectness column exists.
            obj = a[:, 4]
            cls = a[:, 5:]
            scores_all = cls * obj[:, None]
        else:
            cls = a[:, 4:]
            scores_all = cls
        class_ids = scores_all.argmax(axis=1)
        scores = scores_all[np.arange(len(scores_all)), class_ids]
        xywh = a[:, :4].astype(float)
        boxes = np.empty_like(xywh)
        boxes[:, 0] = xywh[:, 0] - xywh[:, 2] / 2
        boxes[:, 1] = xywh[:, 1] - xywh[:, 3] / 2
        boxes[:, 2] = xywh[:, 0] + xywh[:, 2] / 2
        boxes[:, 3] = xywh[:, 1] + xywh[:, 3] / 2

    h, w = frame_shape[:2]
    results = []
    for cid in np.unique(class_ids):
        idx = np.where((class_ids == cid) & (scores >= confidence))[0]
        if not len(idx):
            continue
        kept = _nms(boxes[idx], scores[idx], iou)
        for k in kept:
            j = idx[k]
            b = boxes[j].copy()
            b[[0, 2]] = (b[[0, 2]] - pad_x) / scale
            b[[1, 3]] = (b[[1, 3]] - pad_y) / scale
            x1, y1, x2, y2 = b
            x1 = int(np.clip(x1, 0, w-1)); x2 = int(np.clip(x2, 0, w-1))
            y1 = int(np.clip(y1, 0, h-1)); y2 = int(np.clip(y2, 0, h-1))
            name = names[int(cid)] if names and int(cid) < len(names) else f"class_{int(cid)}"
            results.append(Detection(x1, y1, x2, y2, float(scores[j]), int(cid), name))
    return results
