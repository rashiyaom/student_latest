# =============================================================================
# trt_yolo_multi.py
# EvilEye — Protect your sessions from evil doing | Multi-Class TensorRT YOLOv8n Detector
#
# Extends the original trt_yolo.py to return ALL class detections
# (not just person), with class_id appended to each detection.
#
# Output format: list of [x1, y1, x2, y2, confidence, class_id]
#
# This allows the same engine to detect both:
#   class 0  → person
#   class 67 → cell phone
#   (and any other COCO class if needed)
#
# Preprocessing, inference, and NMS are identical to trt_yolo.py.
# The only difference is in postprocess: we return the argmax class
# per anchor instead of filtering to class 0 only.
#
# Compatible with: Python 3.6 / TensorRT 8.x / PyCUDA / Jetson Nano 2GB
# =============================================================================

import numpy as np
import cv2
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit  # noqa: F401

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

# Classes we care about — only these survive the per-class confidence filter
# Adding more here is zero-cost (same inference run)
TRACKED_CLASSES = {0, 67}   # person, cell phone

# ============================================================================ #

def _letterbox(image, new_shape=320, color=(114,114,114)):
    shape = image.shape[:2]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)
    ratio = min(new_shape[0]/shape[0], new_shape[1]/shape[1])
    new_unpad = (int(round(shape[1]*ratio)), int(round(shape[0]*ratio)))
    dw = new_shape[1] - new_unpad[0]
    dh = new_shape[0] - new_unpad[1]
    dw /= 2; dh /= 2
    if shape[::-1] != new_unpad:
        image = cv2.resize(image, new_unpad, interpolation=cv2.INTER_LINEAR)
    top,bottom = int(round(dh-.1)), int(round(dh+.1))
    left,right = int(round(dw-.1)), int(round(dw+.1))
    image = cv2.copyMakeBorder(image, top,bottom,left,right,
                               cv2.BORDER_CONSTANT, value=color)
    return image, ratio, (dw, dh)


def _nms(boxes, scores, iou_threshold):
    x1=boxes[:,0]; y1=boxes[:,1]; x2=boxes[:,2]; y2=boxes[:,3]
    areas=(x2-x1)*(y2-y1)
    order=scores.argsort()[::-1]
    keep=[]
    while order.size>0:
        i=order[0]; keep.append(i)
        xx1=np.maximum(x1[i],x1[order[1:]])
        yy1=np.maximum(y1[i],y1[order[1:]])
        xx2=np.minimum(x2[i],x2[order[1:]])
        yy2=np.minimum(y2[i],y2[order[1:]])
        w=np.maximum(0.,xx2-xx1); h=np.maximum(0.,yy2-yy1)
        iou=(w*h)/(areas[i]+areas[order[1:]]-w*h+1e-6)
        order=order[np.where(iou<=iou_threshold)[0]+1]
    return keep


class TRTYoloDetectorMulti(object):
    """
    TensorRT YOLOv8n detector returning all tracked classes.

    Usage:
        det = TRTYoloDetectorMulti('yolov8n.engine', input_size=320)
        dets = det.detect_all(frame, conf_thresh=0.35, iou_thresh=0.45)
        # dets: list of [x1,y1,x2,y2,conf,class_id]
    """

    def __init__(self, engine_path, input_size=320):
        self.input_size = input_size

        try:
            with open(engine_path, 'rb') as f, trt.Runtime(TRT_LOGGER) as rt:
                self.engine = rt.deserialize_cuda_engine(f.read())
            self.context  = self.engine.create_execution_context()

            self.inputs   = []
            self.outputs  = []
            self.bindings = []

            for binding in self.engine:
                shape = self.engine.get_binding_shape(binding)
                size  = trt.volume(shape)
                dtype = trt.nptype(self.engine.get_binding_dtype(binding))
                host_mem   = cuda.pagelocked_empty(size, dtype)
                device_mem = cuda.mem_alloc(host_mem.nbytes)
                self.bindings.append(int(device_mem))
                if self.engine.binding_is_input(binding):
                    self.inputs.append({'host':host_mem,'device':device_mem,'shape':shape})
                else:
                    self.outputs.append({'host':host_mem,'device':device_mem,'shape':shape})

            self.stream = cuda.Stream()
            print("[TRTYoloDetectorMulti] Engine: {} | Size: {}".format(
                engine_path, input_size))
        except FileNotFoundError:
            raise RuntimeError("[ERROR] Engine file not found: {}".format(engine_path))
        except Exception as e:
            raise RuntimeError("[ERROR] Failed to load TensorRT engine: {}".format(e))

    # ----------------------------------------------------------------------- #
    def _preprocess(self, frame):
        img, ratio, pad = _letterbox(frame, new_shape=self.input_size)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        img = img.transpose(2,0,1)[np.newaxis,:]
        return np.ascontiguousarray(img), ratio, pad

    # ----------------------------------------------------------------------- #
    def _infer(self, blob):
        np.copyto(self.inputs[0]['host'], blob.ravel())
        cuda.memcpy_htod_async(self.inputs[0]['device'],
                               self.inputs[0]['host'], self.stream)
        self.context.execute_async_v2(bindings=self.bindings,
                                      stream_handle=self.stream.handle)
        cuda.memcpy_dtoh_async(self.outputs[0]['host'],
                               self.outputs[0]['device'], self.stream)
        self.stream.synchronize()
        return self.outputs[0]['host'].reshape(self.outputs[0]['shape'])

    # ----------------------------------------------------------------------- #
    def detect_all(self, frame, conf_thresh=0.35, iou_thresh=0.45):
        """
        Run inference and return detections for all TRACKED_CLASSES.
        Returns list of [x1, y1, x2, y2, confidence, class_id].
        """
        try:
            blob, ratio, pad = self._preprocess(frame)
            raw = self._infer(blob)

            # Output shape: [1, 84, N]
            if raw.ndim == 3:
                raw = raw[0]        # (84, N)
            raw = raw.T             # (N, 84)

            boxes_raw   = raw[:, :4]        # cx cy w h
            class_scores= raw[:, 4:]        # (N, 80)

            orig_h, orig_w = frame.shape[:2]
            dw, dh = pad
            results = []

            for cls_id in TRACKED_CLASSES:
                if cls_id >= class_scores.shape[1]:
                    continue
                scores = class_scores[:, cls_id]
                mask   = scores >= conf_thresh
                if not np.any(mask):
                    continue

                fb = boxes_raw[mask]
                fs = scores[mask]

                cx=fb[:,0]; cy=fb[:,1]; w=fb[:,2]; h=fb[:,3]
                x1=cx-w/2; y1=cy-h/2; x2=cx+w/2; y2=cy+h/2
                xyxy = np.stack([x1,y1,x2,y2],axis=1)

                keep = _nms(xyxy, fs, iou_thresh)
                xyxy = xyxy[keep]; fs_k = fs[keep]

                # Rescale to original frame
                xyxy[:,0] = np.clip((xyxy[:,0]-dw)/ratio, 0, orig_w)
                xyxy[:,1] = np.clip((xyxy[:,1]-dh)/ratio, 0, orig_h)
                xyxy[:,2] = np.clip((xyxy[:,2]-dw)/ratio, 0, orig_w)
                xyxy[:,3] = np.clip((xyxy[:,3]-dh)/ratio, 0, orig_h)

                for i in range(len(xyxy)):
                    results.append([
                        int(xyxy[i,0]), int(xyxy[i,1]),
                        int(xyxy[i,2]), int(xyxy[i,3]),
                        float(fs_k[i]),
                        cls_id
                    ])

            return results
        except Exception as e:
            print("[ERROR] TensorRT inference failed: {}".format(e))
            print("[ERROR] Check engine compatibility with JetPack version")
            return []
