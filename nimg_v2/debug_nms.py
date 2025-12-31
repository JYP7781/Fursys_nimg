#!/usr/bin/env python3
"""
NMS 디버깅 스크립트
"""

import sys
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / 'nimg_v2'))

import torch
import cv2
import numpy as np
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


def debug_nms():
    """NMS 과정 디버깅"""
    from nimg_v2.detection.yolo_detector import letterbox, xywh2xyxy

    MODEL_PATH = "/root/fursys_imgprosessing_ws/models/class187_image85286_v12x_250epochs.pt"
    TEST_IMAGE_PATH = "/root/fursys_imgprosessing_ws/20251208_155531_output/color_000000.png"

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 모델 로드
    logger.info("모델 로드 중...")
    checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=False)
    model = checkpoint['model'].float().to(device).eval()

    logger.info(f"Model class names count: {len(model.names)}")

    # 이미지 로드 및 전처리
    image = cv2.imread(TEST_IMAGE_PATH)
    logger.info(f"Original image shape: {image.shape}")

    # letterbox
    im, ratio, pad = letterbox(image, 640, stride=32)
    logger.info(f"Letterbox image shape: {im.shape}")

    # BGR -> RGB, HWC -> CHW
    im = im.transpose((2, 0, 1))[::-1]
    im = np.ascontiguousarray(im)
    im = torch.from_numpy(im).to(device).float() / 255.0
    if len(im.shape) == 3:
        im = im[None]

    logger.info(f"Tensor shape: {im.shape}")

    # 추론
    with torch.no_grad():
        pred = model(im)

    # 출력 분석
    if isinstance(pred, tuple):
        pred = pred[0]

    logger.info(f"Prediction shape: {pred.shape}")

    # YOLOv8+/v12 형식: (batch, 4 + nc, num_boxes)
    # -> transpose to (batch, num_boxes, 4 + nc)
    pred_t = pred.transpose(1, 2)
    logger.info(f"Transposed shape: {pred_t.shape}")

    x = pred_t[0]  # 첫 번째 배치
    nc = x.shape[1] - 4  # 클래스 수
    logger.info(f"Number of classes: {nc}")
    logger.info(f"Number of boxes: {x.shape[0]}")

    # 좌표 분석
    boxes_raw = x[:, :4]  # (cx, cy, w, h)
    class_scores = x[:, 4:]  # (nc,)

    logger.info(f"Raw boxes - min: {boxes_raw.min().item():.2f}, max: {boxes_raw.max().item():.2f}")

    # 클래스별 최대 점수
    class_conf, class_pred = class_scores.max(1)
    logger.info(f"Class confidence - min: {class_conf.min().item():.4f}, max: {class_conf.max().item():.4f}")
    logger.info(f"Class confidence - mean: {class_conf.mean().item():.4f}")

    # 신뢰도 필터링 테스트
    for thresh in [0.5, 0.25, 0.1, 0.05, 0.01]:
        mask = class_conf > thresh
        logger.info(f"conf > {thresh}: {mask.sum().item()} boxes")

    # 가장 높은 신뢰도의 박스들 확인
    top_k = 10
    top_conf, top_idx = class_conf.topk(top_k)
    logger.info(f"\nTop {top_k} confident boxes:")
    for i, (conf, idx) in enumerate(zip(top_conf, top_idx)):
        box = boxes_raw[idx].cpu().numpy()
        cls = class_pred[idx].item()
        cls_name = model.names.get(cls, f"class_{cls}")
        logger.info(f"  [{i}] conf={conf.item():.4f}, class={cls} ({cls_name}), box={box}")

    # 실제 NMS 수행 (낮은 임계값으로)
    conf_thres = 0.01  # 매우 낮은 임계값
    iou_thres = 0.45
    max_det = 300

    conf_mask = class_conf > conf_thres
    x_filtered = x[conf_mask]
    class_conf_filtered = class_conf[conf_mask]
    class_pred_filtered = class_pred[conf_mask]

    logger.info(f"\nAfter filtering (conf > {conf_thres}): {x_filtered.shape[0]} boxes")

    if x_filtered.shape[0] > 0:
        # Box 변환
        boxes = xywh2xyxy(x_filtered[:, :4])
        logger.info(f"Converted boxes (xyxy) - min: {boxes.min().item():.2f}, max: {boxes.max().item():.2f}")

        # 신뢰도 순 정렬
        conf_sort = class_conf_filtered.argsort(descending=True)[:30000]
        boxes = boxes[conf_sort]
        class_conf_sorted = class_conf_filtered[conf_sort]
        class_pred_sorted = class_pred_filtered[conf_sort]

        # NMS
        c = class_pred_sorted.float() * 7680
        boxes_offset = boxes + c.unsqueeze(1)

        try:
            i = torch.ops.torchvision.nms(boxes_offset, class_conf_sorted, iou_thres)
            i = i[:max_det]
            logger.info(f"After NMS: {len(i)} boxes")

            # 최종 결과
            for j, idx in enumerate(i[:5]):
                conf = class_conf_sorted[idx].item()
                cls = class_pred_sorted[idx].item()
                box = boxes[idx].cpu().numpy()
                cls_name = model.names.get(cls, f"class_{cls}")
                logger.info(f"  [{j}] conf={conf:.4f}, class={cls} ({cls_name}), box={box}")

        except Exception as e:
            logger.error(f"NMS failed: {e}")
    else:
        logger.warning("No boxes passed the confidence threshold!")


if __name__ == "__main__":
    debug_nms()
