#!/usr/bin/env python3
"""
기존 nimg와 비교 디버깅
"""

import sys
import os
from pathlib import Path

# nimg 경로 추가
NIMG_PATH = Path(__file__).parent.parent / 'nimg' / 'nimg'
sys.path.insert(0, str(NIMG_PATH))

import torch
import cv2
import numpy as np
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


def debug_with_original():
    """기존 nimg Detector 사용"""
    MODEL_PATH = "/root/fursys_imgprosessing_ws/models/class187_image85286_v12x_250epochs.pt"
    TEST_IMAGE_PATH = "/root/fursys_imgprosessing_ws/20251208_155531_output/color_000000.png"

    logger.info("=" * 60)
    logger.info("기존 nimg Detector 테스트")
    logger.info("=" * 60)

    try:
        from nimg.submodules.detect import Detector

        detector = Detector()
        detector.detect_set(MODEL_PATH)

        logger.info(f"Model stride: {detector.stride}")
        logger.info(f"Model names: {len(detector.names)} classes")
        logger.info(f"Half precision: {detector.model.fp16}")

        # 테스트 이미지 저장 후 탐지
        cv2.imwrite('source.png', cv2.imread(TEST_IMAGE_PATH))
        result_img, items, check_flag = detector.detect('source.png')

        logger.info(f"Detection result: check_flag={check_flag}")
        logger.info(f"Items count: {items.size()}")

        if items.size() > 0:
            for item in items.itemlist:
                logger.info(f"  - {item.getName()}: {item.getConfidence():.4f}")
                logger.info(f"    Box: ({item.x}, {item.y}) - ({item.x2}, {item.y2})")

        # 결과 이미지 저장
        cv2.imwrite('/root/fursys_imgprosessing_ws/src/nimg_v2/original_result.png', result_img)
        logger.info("Result saved to original_result.png")

    except Exception as e:
        logger.error(f"Error: {e}")
        import traceback
        traceback.print_exc()


def debug_model_format():
    """모델 출력 형식 확인"""
    MODEL_PATH = "/root/fursys_imgprosessing_ws/models/class187_image85286_v12x_250epochs.pt"
    TEST_IMAGE_PATH = "/root/fursys_imgprosessing_ws/20251208_155531_output/color_000000.png"

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 직접 모델 로드
    checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=False)
    model = checkpoint['model'].to(device)

    logger.info(f"Model type: {type(model)}")
    logger.info(f"Model class: {model.__class__.__name__}")
    logger.info(f"Model yaml: {model.yaml if hasattr(model, 'yaml') else 'N/A'}")

    # FP16 테스트
    logger.info("\n=== Testing with FP16 ===")
    model_half = model.half()
    model_half.eval()

    # 이미지 전처리 - 기존 nimg 방식 (LoadImages)
    image = cv2.imread(TEST_IMAGE_PATH)
    im0_shape = image.shape[:2]

    # letterbox
    def letterbox(im, new_shape=640, stride=32):
        shape = im.shape[:2]
        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)
        dw /= 2
        dh /= 2
        if shape[::-1] != new_unpad:
            im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))
        return im, (dw, dh)

    im, pad = letterbox(image, 640, 32)
    logger.info(f"Letterbox shape: {im.shape}")

    # 전처리
    im = im.transpose((2, 0, 1))[::-1]
    im = np.ascontiguousarray(im)
    im = torch.from_numpy(im).to(device)
    im = im.half()  # FP16
    im /= 255.0
    if len(im.shape) == 3:
        im = im[None]

    logger.info(f"Input tensor: shape={im.shape}, dtype={im.dtype}")

    # 추론
    with torch.no_grad():
        pred = model_half(im)

    if isinstance(pred, tuple):
        logger.info(f"Output is tuple with {len(pred)} elements")
        pred = pred[0]

    logger.info(f"Output shape: {pred.shape}")
    logger.info(f"Output dtype: {pred.dtype}")
    logger.info(f"Output range: [{pred.min().item():.4f}, {pred.max().item():.4f}]")

    # YOLOv8+/v12: (batch, 4+nc, num_boxes)
    if pred.shape[1] < pred.shape[2]:
        pred_t = pred.transpose(1, 2)
        logger.info(f"Transposed shape: {pred_t.shape}")

        x = pred_t[0]
        boxes = x[:, :4]
        scores = x[:, 4:]

        # Sigmoid 적용 여부 확인
        logger.info(f"\nScores range BEFORE sigmoid: [{scores.min().item():.4f}, {scores.max().item():.4f}]")

        # Sigmoid 적용
        scores_sigmoid = torch.sigmoid(scores)
        logger.info(f"Scores range AFTER sigmoid: [{scores_sigmoid.min().item():.4f}, {scores_sigmoid.max().item():.4f}]")

        # 최대 확률
        max_scores, max_idx = scores_sigmoid.max(dim=1)
        logger.info(f"Max confidence: {max_scores.max().item():.4f}")
        logger.info(f"Boxes > 0.5: {(max_scores > 0.5).sum().item()}")
        logger.info(f"Boxes > 0.25: {(max_scores > 0.25).sum().item()}")

        # Top 결과
        top_k = 5
        top_conf, top_idx = max_scores.topk(top_k)
        logger.info(f"\nTop {top_k} boxes:")
        for i, (conf, idx) in enumerate(zip(top_conf, top_idx)):
            cls = max_idx[idx].item()
            box = boxes[idx].cpu().numpy()
            logger.info(f"  [{i}] conf={conf.item():.4f}, class={cls}, box={box}")


if __name__ == "__main__":
    # 먼저 모델 형식 확인
    debug_model_format()

    print("\n" + "=" * 60 + "\n")

    # 기존 nimg 테스트
    debug_with_original()
