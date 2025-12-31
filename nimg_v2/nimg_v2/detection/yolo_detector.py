"""
YOLO 기반 객체 탐지 모듈
YOLOv5/v8/v11/v12 등 ultralytics 모델 지원

기존 nimg의 detect.py 방식 (DetectMultiBackend)과
ultralytics YOLO API 모두 지원
"""

import torch
import cv2
import numpy as np
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union
from pathlib import Path
import logging
import sys
import os

logger = logging.getLogger(__name__)


def letterbox(im, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
    """
    이미지를 YOLO 입력 크기에 맞게 리사이즈 (비율 유지)

    Args:
        im: 입력 이미지 (HWC)
        new_shape: 목표 크기
        color: 패딩 색상
        auto: 최소 직사각형 자동 조정
        scaleFill: 스케일 채우기
        scaleup: 업스케일 허용
        stride: 스트라이드

    Returns:
        리사이즈된 이미지, 비율, 패딩
    """
    shape = im.shape[:2]  # 현재 크기 [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # 스케일 비율 계산 (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # 업스케일 방지, 다운스케일만 허용
        r = min(r, 1.0)

    # 패딩 계산
    ratio = r, r  # width, height 비율
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh 패딩

    if auto:  # 최소 직사각형
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh 패딩
    elif scaleFill:  # 스케일 채우기
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height 비율

    dw /= 2  # 패딩 분할
    dh /= 2

    if shape[::-1] != new_unpad:  # 리사이즈
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # 패딩 추가

    return im, ratio, (dw, dh)


@dataclass
class Detection:
    """탐지 결과를 담는 데이터 클래스"""
    class_id: int
    class_name: str
    confidence: float
    x: int      # 좌상단 x
    y: int      # 좌상단 y
    width: int  # 너비
    height: int # 높이
    x2: int     # 우하단 x
    y2: int     # 우하단 y

    @property
    def bbox(self) -> Tuple[int, int, int, int]:
        """바운딩 박스 (x, y, w, h) 반환"""
        return (self.x, self.y, self.width, self.height)

    @property
    def bbox_xyxy(self) -> Tuple[int, int, int, int]:
        """바운딩 박스 (x1, y1, x2, y2) 반환"""
        return (self.x, self.y, self.x2, self.y2)

    @property
    def center(self) -> Tuple[int, int]:
        """중심점 반환"""
        return (self.x + self.width // 2, self.y + self.height // 2)

    @property
    def area(self) -> int:
        """면적 반환"""
        return self.width * self.height

    def iou(self, other: 'Detection') -> float:
        """다른 Detection과의 IoU 계산"""
        x1 = max(self.x, other.x)
        y1 = max(self.y, other.y)
        x2 = min(self.x2, other.x2)
        y2 = min(self.y2, other.y2)

        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        union = self.area + other.area - intersection

        return intersection / union if union > 0 else 0.0


def scale_boxes(img1_shape, boxes, img0_shape, ratio_pad=None):
    """
    좌표를 원본 이미지 크기에 맞게 조정

    Args:
        img1_shape: 모델 입력 크기 (height, width)
        boxes: 바운딩 박스 좌표
        img0_shape: 원본 이미지 크기 (height, width)
        ratio_pad: (ratio, (dw, dh)) 튜플

    Returns:
        조정된 좌표
    """
    if ratio_pad is None:  # letterbox에서 계산된 값 사용
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain = old / new
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    boxes[..., [0, 2]] -= pad[0]  # x padding
    boxes[..., [1, 3]] -= pad[1]  # y padding
    boxes[..., :4] /= gain
    clip_boxes(boxes, img0_shape)
    return boxes


def clip_boxes(boxes, shape):
    """좌표를 이미지 범위 내로 클리핑"""
    if isinstance(boxes, torch.Tensor):
        boxes[..., 0].clamp_(0, shape[1])  # x1
        boxes[..., 1].clamp_(0, shape[0])  # y1
        boxes[..., 2].clamp_(0, shape[1])  # x2
        boxes[..., 3].clamp_(0, shape[0])  # y2
    else:  # numpy
        boxes[..., [0, 2]] = boxes[..., [0, 2]].clip(0, shape[1])  # x1, x2
        boxes[..., [1, 3]] = boxes[..., [1, 3]].clip(0, shape[0])  # y1, y2


def non_max_suppression(
    prediction,
    conf_thres=0.25,
    iou_thres=0.45,
    classes=None,
    agnostic=False,
    multi_label=False,
    labels=(),
    max_det=300,
    nm=0,
):
    """
    비최대 억제 (NMS) 수행 - YOLOv5/v8+/v12 자동 감지

    Args:
        prediction: 모델 출력
        conf_thres: 신뢰도 임계값
        iou_thres: NMS IoU 임계값
        classes: 필터링할 클래스 목록
        agnostic: 클래스 무관 NMS
        multi_label: 다중 레이블 허용
        labels: 라벨
        max_det: 최대 탐지 수
        nm: 마스크 수

    Returns:
        탐지 결과 리스트
    """
    # tuple인 경우 첫 번째 요소 사용
    if isinstance(prediction, (list, tuple)):
        prediction = prediction[0]

    # 출력 형식 자동 감지
    # YOLOv5: (batch, num_boxes, 5 + nc) - 마지막 차원이 큼
    # YOLOv8+: (batch, 4 + nc, num_boxes) - 두 번째 차원이 작음
    if prediction.shape[1] < prediction.shape[2]:
        # YOLOv8+/v12 형식 -> ultralytics NMS 사용
        return non_max_suppression_ultralytics(
            prediction,
            conf_thres=conf_thres,
            iou_thres=iou_thres,
            classes=classes,
            agnostic=agnostic,
            max_det=max_det
        )
    else:
        # YOLOv5 형식 -> 기존 NMS 사용
        return non_max_suppression_yolov5(
            prediction,
            conf_thres=conf_thres,
            iou_thres=iou_thres,
            classes=classes,
            agnostic=agnostic,
            multi_label=multi_label,
            labels=labels,
            max_det=max_det,
            nm=nm
        )


def non_max_suppression_yolov5(
    prediction,
    conf_thres=0.25,
    iou_thres=0.45,
    classes=None,
    agnostic=False,
    multi_label=False,
    labels=(),
    max_det=300,
    nm=0,
):
    """
    YOLOv5 형식 비최대 억제 (NMS) 수행

    Args:
        prediction: 모델 출력 (batch_size, num_boxes, 5 + num_classes)
    """
    device = prediction.device
    mps = 'mps' in device.type
    if mps:
        prediction = prediction.cpu()
    bs = prediction.shape[0]  # batch size
    nc = prediction.shape[2] - nm - 5  # 클래스 수
    xc = prediction[..., 4] > conf_thres  # 신뢰도 필터 후보

    # 설정값
    max_wh = 7680
    max_nms = 30000
    redundant = True
    multi_label &= nc > 1
    merge = False

    output = [torch.zeros((0, 6 + nm), device=prediction.device)] * bs

    for xi, x in enumerate(prediction):
        x = x[xc[xi]]

        if labels and len(labels[xi]):
            lb = labels[xi]
            v = torch.zeros((len(lb), nc + nm + 5), device=x.device)
            v[:, :4] = lb[:, 1:5]
            v[:, 4] = 1.0
            v[range(len(lb)), lb[:, 0].long() + 5] = 1.0
            x = torch.cat((x, v), 0)

        if not x.shape[0]:
            continue

        x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf

        box = xywh2xyxy(x[:, :4])
        mask = x[:, 5 + nc:]

        if multi_label:
            i, j = (x[:, 5:5 + nc] > conf_thres).nonzero(as_tuple=False).T
            x = torch.cat((box[i], x[i, 5 + j, None], j[:, None].float(), mask[i]), 1)
        else:
            conf, j = x[:, 5:5 + nc].max(1, keepdim=True)
            x = torch.cat((box, conf, j.float(), mask), 1)[conf.view(-1) > conf_thres]

        if classes is not None:
            x = x[(x[:, 5:6] == torch.tensor(classes, device=x.device)).any(1)]

        n = x.shape[0]
        if not n:
            continue
        x = x[x[:, 4].argsort(descending=True)[:max_nms]]

        c = x[:, 5:6] * (0 if agnostic else max_wh)
        boxes, scores = x[:, :4] + c, x[:, 4]
        i = torch.ops.torchvision.nms(boxes, scores, iou_thres)
        i = i[:max_det]

        if merge and (1 < n < 3E3):
            iou = box_iou(boxes[i], boxes) > iou_thres
            weights = iou * scores[None]
            x[i, :4] = torch.mm(weights, x[:, :4]).float() / weights.sum(1, keepdim=True)
            if redundant:
                i = i[iou.sum(1) > 1]

        output[xi] = x[i]
        if mps:
            output[xi] = output[xi].to(device)

    return output


def non_max_suppression_ultralytics(
    prediction,
    conf_thres=0.25,
    iou_thres=0.45,
    classes=None,
    agnostic=False,
    max_det=300,
):
    """
    ultralytics YOLOv8+/v12 형식 비최대 억제 (NMS) 수행

    YOLOv8+ 출력 형식: (batch, 4 + num_classes, num_boxes)
    - objectness 점수 없음
    - 클래스 확률이 직접 사용됨

    Args:
        prediction: 모델 출력 (batch_size, 4 + num_classes, num_boxes) 또는 tuple

    Returns:
        탐지 결과 리스트 [(x1, y1, x2, y2, conf, cls), ...]
    """
    device = prediction.device
    bs = prediction.shape[0]

    # YOLOv8+ 형식: (batch, 4 + nc, num_boxes) -> (batch, num_boxes, 4 + nc)
    prediction = prediction.transpose(1, 2)

    nc = prediction.shape[2] - 4  # 클래스 수

    output = [torch.zeros((0, 6), device=device)] * bs

    max_wh = 7680
    max_nms = 30000

    for xi, x in enumerate(prediction):
        # x shape: (num_boxes, 4 + nc)
        # boxes: x[:, :4] (x_center, y_center, width, height)
        # class_scores: x[:, 4:]

        # 최대 클래스 확률과 클래스 인덱스
        class_conf, class_pred = x[:, 4:].max(1)

        # 신뢰도 필터
        conf_mask = class_conf > conf_thres
        x = x[conf_mask]
        class_conf = class_conf[conf_mask]
        class_pred = class_pred[conf_mask]

        if not x.shape[0]:
            continue

        # 클래스 필터
        if classes is not None:
            class_mask = (class_pred.unsqueeze(1) == torch.tensor(classes, device=device)).any(1)
            x = x[class_mask]
            class_conf = class_conf[class_mask]
            class_pred = class_pred[class_mask]

        if not x.shape[0]:
            continue

        # Box 변환: (cx, cy, w, h) -> (x1, y1, x2, y2)
        boxes = xywh2xyxy(x[:, :4])

        # 신뢰도 순 정렬
        conf_sort = class_conf.argsort(descending=True)[:max_nms]
        boxes = boxes[conf_sort]
        class_conf = class_conf[conf_sort]
        class_pred = class_pred[conf_sort]

        # Batched NMS
        c = class_pred.float() * (0 if agnostic else max_wh)
        boxes_offset = boxes + c.unsqueeze(1)

        i = torch.ops.torchvision.nms(boxes_offset, class_conf, iou_thres)
        i = i[:max_det]

        # 결과 조합: (x1, y1, x2, y2, confidence, class)
        output[xi] = torch.cat([
            boxes[i],
            class_conf[i].unsqueeze(1),
            class_pred[i].float().unsqueeze(1)
        ], dim=1)

    return output


def xywh2xyxy(x):
    """(x, y, w, h)를 (x1, y1, x2, y2)로 변환"""
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[..., 0] = x[..., 0] - x[..., 2] / 2  # x1
    y[..., 1] = x[..., 1] - x[..., 3] / 2  # y1
    y[..., 2] = x[..., 0] + x[..., 2] / 2  # x2
    y[..., 3] = x[..., 1] + x[..., 3] / 2  # y2
    return y


def box_iou(box1, box2, eps=1e-7):
    """
    두 박스 세트의 IoU 계산

    Args:
        box1: (N, 4) 텐서
        box2: (M, 4) 텐서

    Returns:
        (N, M) IoU 행렬
    """
    (a1, a2), (b1, b2) = box1.unsqueeze(1).chunk(2, 2), box2.unsqueeze(0).chunk(2, 2)
    inter = (torch.min(a2, b2) - torch.max(a1, b1)).clamp(0).prod(2)
    return inter / ((a2 - a1).prod(2) + (b2 - b1).prod(2) - inter + eps)


class YOLODetector:
    """
    YOLO 기반 객체 탐지기

    YOLOv5 DetectMultiBackend 방식과 ultralytics YOLO API 모두 지원.
    파인튜닝된 모델 호환성을 위해 DetectMultiBackend 방식 우선 사용.
    """

    def __init__(
        self,
        model_path: str,
        conf_threshold: float = 0.5,
        iou_threshold: float = 0.45,
        max_detections: int = 1000,
        device: Optional[str] = None,
        img_size: int = 640,
        half: bool = True,
        use_yolov5_backend: bool = True,
        data_yaml: Optional[str] = None
    ):
        """
        Args:
            model_path: 모델 파일 경로 (.pt)
            conf_threshold: 신뢰도 임계값
            iou_threshold: NMS IoU 임계값
            max_detections: 최대 탐지 개수
            device: 디바이스 ('cuda', 'cpu', 또는 None=자동)
            img_size: 모델 입력 이미지 크기
            half: FP16 반정밀도 사용 여부
            use_yolov5_backend: YOLOv5 DetectMultiBackend 사용 여부 (권장)
            data_yaml: 데이터셋 YAML 파일 경로 (클래스명 로드용)
        """
        self.model_path = Path(model_path)
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.max_detections = max_detections
        self.img_size = img_size
        self.half = half
        self.use_yolov5_backend = use_yolov5_backend
        self.data_yaml = data_yaml

        # 디바이스 설정
        if device is None or device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        # half precision은 CUDA에서만 지원
        if self.device.type != 'cuda':
            self.half = False

        # 모델 로드
        self._load_model()

        logger.info(f"YOLODetector initialized: model={model_path}, device={self.device}, "
                    f"backend={'YOLOv5' if self._using_yolov5_backend else 'ultralytics'}")

    def _load_model(self):
        """
        모델 로드 - 모델 유형 자동 감지

        로드 순서:
        1. ultralytics YOLO API (YOLOv8+/v11/v12 등 최신 모델)
        2. YOLOv5 DetectMultiBackend (기존 nimg 방식)
        3. 직접 torch.load (fallback)
        """
        self._using_yolov5_backend = False
        self._using_ultralytics = False
        self.names = {}
        self.stride = 32  # 기본값

        # 방법 1: ultralytics YOLO API (최신 모델용 - YOLOv8+/v11/v12)
        try:
            from ultralytics import YOLO

            self.model = YOLO(str(self.model_path))
            self.model.to(self.device)
            self.names = self.model.names

            # ultralytics 모델인지 확인 (YOLOv5 제외)
            # YOLOv5는 DetectMultiBackend로 처리하는 것이 더 적합
            model_info = str(type(self.model.model))
            if 'DetectionModel' in model_info or 'v8' in model_info or 'v11' in model_info or 'v12' in model_info:
                self._using_ultralytics = True
                logger.info(f"Loaded model using ultralytics YOLO: {len(self.names)} classes")
                return

        except Exception as e:
            logger.debug(f"ultralytics YOLO load skipped: {e}")

        # 방법 2: YOLOv5 DetectMultiBackend (기존 nimg 방식)
        if self.use_yolov5_backend:
            try:
                # nimg 패키지의 models.common에서 DetectMultiBackend 가져오기
                nimg_path = Path(__file__).parents[3] / 'nimg' / 'nimg'
                if nimg_path.exists():
                    sys.path.insert(0, str(nimg_path))

                from models.common import DetectMultiBackend
                from utils.general import check_img_size
                from utils.torch_utils import select_device

                # 디바이스 선택
                device = select_device(str(self.device).replace('cuda', '0'))

                # 모델 로드
                self.model = DetectMultiBackend(
                    str(self.model_path),
                    device=device,
                    dnn=False,
                    data=self.data_yaml,
                    fp16=self.half
                )
                self.stride = int(self.model.stride)
                self.names = self.model.names
                self.pt = self.model.pt

                # 이미지 크기 체크
                self.img_size = check_img_size(self.img_size, s=self.stride)

                # FP16 설정
                if self.half:
                    self.model.half()

                self._using_yolov5_backend = True
                logger.info(f"Loaded model using YOLOv5 DetectMultiBackend: "
                            f"{len(self.names)} classes, stride={self.stride}")
                return

            except Exception as e:
                logger.warning(f"Failed to load with DetectMultiBackend: {e}")
                logger.info("Falling back to alternative loading methods...")

        # 방법 2: 직접 torch.load로 모델 로드 (대안)
        try:
            # PyTorch 2.6+ 보안 변경 대응: weights_only=False 사용
            checkpoint = torch.load(str(self.model_path), map_location=self.device, weights_only=False)

            # 모델 구조 확인
            if 'model' in checkpoint:
                self.model = checkpoint['model'].float()
            elif 'ema' in checkpoint:
                self.model = checkpoint['ema'].float()
            else:
                self.model = checkpoint

            self.model = self.model.to(self.device)
            self.model.eval()

            # 클래스 이름 추출
            if hasattr(self.model, 'names'):
                self.names = self.model.names
            elif 'names' in checkpoint:
                self.names = checkpoint['names']

            # stride 추출
            if hasattr(self.model, 'stride'):
                self.stride = int(self.model.stride.max())

            # FP16 설정
            if self.half:
                self.model.half()

            self._using_yolov5_backend = True  # 같은 전처리 사용
            logger.info(f"Loaded model directly from checkpoint: {len(self.names)} classes")
            return

        except Exception as e:
            logger.warning(f"Failed to load model directly: {e}")

        # 방법 3: ultralytics YOLO API (최신 모델용)
        try:
            from ultralytics import YOLO
            self.model = YOLO(str(self.model_path))
            self.model.to(self.device)
            self.names = self.model.names
            self._using_ultralytics = True
            logger.info(f"Loaded model using ultralytics YOLO: {len(self.names)} classes")
            return

        except Exception as e:
            logger.warning(f"Failed to load with ultralytics: {e}")

        # 방법 4: YOLOv5 Hub (마지막 대안)
        try:
            self.model = torch.hub.load('ultralytics/yolov5', 'custom',
                                        path=str(self.model_path))
            self.model.to(self.device)
            self.names = self.model.names
            logger.info("Loaded model using YOLOv5 hub")
            return

        except Exception as e:
            raise RuntimeError(f"Failed to load YOLO model with all methods: {e}")

    @property
    def class_names(self) -> dict:
        """클래스 이름 딕셔너리 반환"""
        return self.names

    def detect(
        self,
        image: np.ndarray,
        classes: Optional[List[int]] = None
    ) -> List[Detection]:
        """
        이미지에서 객체 탐지

        Args:
            image: BGR 이미지 (numpy array)
            classes: 탐지할 클래스 ID 리스트 (None이면 모든 클래스)

        Returns:
            Detection 리스트
        """
        if image is None or image.size == 0:
            logger.warning("Empty image provided for detection")
            return []

        try:
            if self._using_yolov5_backend:
                detections = self._detect_yolov5_backend(image, classes)
            elif self._using_ultralytics:
                detections = self._detect_ultralytics(image, classes)
            else:
                detections = self._detect_yolov5_hub(image, classes)

            logger.debug(f"Detected {len(detections)} objects")
            return detections

        except Exception as e:
            logger.error(f"Detection failed: {e}")
            return []

    def _detect_yolov5_backend(
        self,
        image: np.ndarray,
        classes: Optional[List[int]] = None
    ) -> List[Detection]:
        """
        YOLOv5 DetectMultiBackend 방식으로 탐지 (기존 nimg/detect.py 방식)

        핵심 전처리:
        1. letterbox로 이미지 리사이즈 (비율 유지)
        2. BGR -> RGB 변환
        3. HWC -> CHW 변환
        4. 0-255 -> 0.0-1.0 정규화
        5. 배치 차원 추가
        6. half precision 적용 (필요시)
        """
        # 원본 이미지 크기 저장
        im0 = image.copy()
        im0_shape = im0.shape[:2]  # (height, width)

        # 1. Letterbox 전처리 (비율 유지하면서 리사이즈)
        im = letterbox(
            image,
            new_shape=self.img_size,
            stride=self.stride,
            auto=getattr(self, 'pt', True)  # PyTorch 모델이면 auto=True
        )[0]

        # 2. BGR -> RGB, HWC -> CHW
        im = im.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        im = np.ascontiguousarray(im)

        # 3. Tensor 변환
        im = torch.from_numpy(im).to(self.device)

        # 4. Half precision
        im = im.half() if self.half else im.float()

        # 5. 정규화 (0-255 -> 0.0-1.0)
        im /= 255.0

        # 6. 배치 차원 추가
        if len(im.shape) == 3:
            im = im[None]  # (1, C, H, W)

        # 7. 추론
        with torch.no_grad():
            pred = self.model(im, augment=False, visualize=False)

        # 8. NMS
        pred = non_max_suppression(
            pred,
            conf_thres=self.conf_threshold,
            iou_thres=self.iou_threshold,
            classes=classes,
            agnostic=False,
            max_det=self.max_detections
        )

        # 9. 결과 처리
        detections = []

        for i, det in enumerate(pred):
            if len(det):
                # 좌표를 원본 이미지 크기에 맞게 조정
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0_shape).round()

                # Detection 객체로 변환
                for *xyxy, conf, cls_id in reversed(det):
                    cls_id = int(cls_id)
                    x1, y1, x2, y2 = int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])

                    # 클래스 이름 가져오기
                    class_name = self.names.get(cls_id, f"class_{cls_id}")
                    if isinstance(class_name, list):
                        class_name = class_name[0] if class_name else f"class_{cls_id}"

                    detections.append(Detection(
                        class_id=cls_id,
                        class_name=str(class_name),
                        confidence=float(conf),
                        x=x1,
                        y=y1,
                        width=x2 - x1,
                        height=y2 - y1,
                        x2=x2,
                        y2=y2
                    ))

        return detections

    def _detect_ultralytics(
        self,
        image: np.ndarray,
        classes: Optional[List[int]] = None
    ) -> List[Detection]:
        """ultralytics YOLO API로 탐지"""
        results = self.model(
            image,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            max_det=self.max_detections,
            classes=classes,
            verbose=False
        )

        detections = []

        for result in results:
            boxes = result.boxes
            for box in boxes:
                xyxy = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0])
                cls_id = int(box.cls[0])

                x1, y1, x2, y2 = map(int, xyxy)

                class_name = self.names.get(cls_id, f"class_{cls_id}")

                detections.append(Detection(
                    class_id=cls_id,
                    class_name=str(class_name),
                    confidence=conf,
                    x=x1,
                    y=y1,
                    width=x2 - x1,
                    height=y2 - y1,
                    x2=x2,
                    y2=y2
                ))

        return detections

    def _detect_yolov5_hub(
        self,
        image: np.ndarray,
        classes: Optional[List[int]] = None
    ) -> List[Detection]:
        """YOLOv5 hub로 탐지"""
        self.model.conf = self.conf_threshold
        self.model.iou = self.iou_threshold
        self.model.max_det = self.max_detections

        if classes is not None:
            self.model.classes = classes

        results = self.model(image)
        predictions = results.xyxy[0].cpu().numpy()

        detections = []

        for pred in predictions:
            x1, y1, x2, y2, conf, cls_id = pred
            cls_id = int(cls_id)

            class_name = self.names.get(cls_id, f"class_{cls_id}")

            detections.append(Detection(
                class_id=cls_id,
                class_name=str(class_name),
                confidence=float(conf),
                x=int(x1),
                y=int(y1),
                width=int(x2 - x1),
                height=int(y2 - y1),
                x2=int(x2),
                y2=int(y2)
            ))

        return detections

    def detect_batch(
        self,
        images: List[np.ndarray],
        classes: Optional[List[int]] = None
    ) -> List[List[Detection]]:
        """
        배치 이미지 탐지

        Args:
            images: 이미지 리스트
            classes: 탐지할 클래스 ID 리스트

        Returns:
            Detection 리스트의 리스트
        """
        return [self.detect(img, classes) for img in images]

    def get_best_detection(
        self,
        detections: List[Detection],
        by: str = 'confidence'
    ) -> Optional[Detection]:
        """
        최상의 탐지 결과 반환

        Args:
            detections: Detection 리스트
            by: 정렬 기준 ('confidence', 'area')

        Returns:
            최상의 Detection 또는 None
        """
        if not detections:
            return None

        if by == 'confidence':
            return max(detections, key=lambda d: d.confidence)
        elif by == 'area':
            return max(detections, key=lambda d: d.area)
        else:
            raise ValueError(f"Unknown sorting criteria: {by}")

    def filter_detections(
        self,
        detections: List[Detection],
        min_confidence: Optional[float] = None,
        min_area: Optional[int] = None,
        max_area: Optional[int] = None,
        class_ids: Optional[List[int]] = None
    ) -> List[Detection]:
        """
        탐지 결과 필터링

        Args:
            detections: Detection 리스트
            min_confidence: 최소 신뢰도
            min_area: 최소 면적
            max_area: 최대 면적
            class_ids: 허용할 클래스 ID 리스트

        Returns:
            필터링된 Detection 리스트
        """
        result = detections

        if min_confidence is not None:
            result = [d for d in result if d.confidence >= min_confidence]

        if min_area is not None:
            result = [d for d in result if d.area >= min_area]

        if max_area is not None:
            result = [d for d in result if d.area <= max_area]

        if class_ids is not None:
            result = [d for d in result if d.class_id in class_ids]

        return result

    def draw_detections(
        self,
        image: np.ndarray,
        detections: List[Detection],
        color: Tuple[int, int, int] = (0, 255, 0),
        thickness: int = 2,
        font_scale: float = 0.5
    ) -> np.ndarray:
        """
        탐지 결과를 이미지에 그리기

        Args:
            image: 원본 이미지
            detections: Detection 리스트
            color: 바운딩 박스 색상 (BGR)
            thickness: 선 두께
            font_scale: 폰트 크기

        Returns:
            결과 이미지
        """
        result = image.copy()

        for det in detections:
            # 바운딩 박스
            cv2.rectangle(result, (det.x, det.y), (det.x2, det.y2), color, thickness)

            # 라벨
            label = f"{det.class_name}: {det.confidence:.2f}"
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)

            # 라벨 배경
            cv2.rectangle(result,
                         (det.x, det.y - label_size[1] - 5),
                         (det.x + label_size[0], det.y),
                         color, -1)

            # 라벨 텍스트
            cv2.putText(result, label, (det.x, det.y - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 1)

        return result

    def warmup(self, image_size: Tuple[int, int] = (640, 480)):
        """
        모델 워밍업 (첫 추론 지연 방지)

        Args:
            image_size: 워밍업 이미지 크기 (height, width)
        """
        dummy_image = np.zeros((image_size[0], image_size[1], 3), dtype=np.uint8)
        _ = self.detect(dummy_image)
        logger.info("Model warmup completed")
