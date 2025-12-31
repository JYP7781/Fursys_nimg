"""
Reference Image Loader 단위 테스트
"""

import numpy as np
import pytest
from pathlib import Path
from nimg_v3.pose.reference_image_loader import (
    ReferenceImageLoader,
    ReferenceImage,
    ReferenceImageSet
)


class TestReferenceImageLoader:
    """참조 이미지 로더 테스트"""

    @pytest.fixture
    def test_data_path(self):
        """테스트 데이터 경로"""
        path = Path("/root/fursys_img_251229/extraction")
        if not path.exists():
            pytest.skip("Test data not available")
        return path

    def test_initialization(self, test_data_path):
        """로더 초기화"""
        loader = ReferenceImageLoader(str(test_data_path))
        assert loader.base_dir.exists()

    def test_view_folders_found(self, test_data_path):
        """뷰 디렉토리 검색"""
        loader = ReferenceImageLoader(str(test_data_path))

        # 최소 하나 이상의 뷰가 있어야 함
        assert len(loader.view_folders) > 0

        # 각 뷰 폴더에 rgb 디렉토리가 있어야 함
        for folder in loader.view_folders:
            assert (folder / 'rgb').exists()

    def test_load_all_views(self, test_data_path):
        """모든 뷰 로드"""
        loader = ReferenceImageLoader(str(test_data_path))
        image_set = loader.load_all_views(images_per_view=1)

        assert isinstance(image_set, ReferenceImageSet)
        assert len(image_set.images) > 0

        # 각도 분포 확인
        angles = image_set.view_angles
        assert len(angles) > 0

    def test_reference_image_properties(self, test_data_path):
        """참조 이미지 속성 확인"""
        loader = ReferenceImageLoader(str(test_data_path))
        image_set = loader.load_all_views(images_per_view=1)

        if len(image_set.images) == 0:
            pytest.skip("No images loaded")

        img = image_set.images[0]

        # RGB 이미지 확인
        assert img.rgb is not None
        assert img.rgb.dtype == np.uint8
        assert img.rgb.ndim == 3  # HxWxC

        # 깊이 이미지 확인
        if img.depth is not None:
            assert img.depth.ndim == 2
            assert img.depth.dtype in [np.float32, np.float64]

    def test_get_rgb_stack(self, test_data_path):
        """RGB 스택 생성"""
        loader = ReferenceImageLoader(str(test_data_path))
        image_set = loader.load_all_views(images_per_view=1)

        if len(image_set.images) < 2:
            pytest.skip("Need at least 2 images")

        rgb_stack = image_set.get_rgb_stack()
        assert rgb_stack.ndim == 4  # [N, H, W, 3]
        assert rgb_stack.shape[0] == len(image_set.images)

    def test_get_depth_stack(self, test_data_path):
        """Depth 스택 생성"""
        loader = ReferenceImageLoader(str(test_data_path))
        image_set = loader.load_all_views(images_per_view=1)

        if len(image_set.images) < 2:
            pytest.skip("Need at least 2 images")

        depth_stack = image_set.get_depth_stack()
        assert depth_stack.ndim == 3  # [N, H, W]
        assert depth_stack.shape[0] == len(image_set.images)


class TestReferenceImage:
    """ReferenceImage 클래스 테스트"""

    def test_has_mask_property(self):
        """has_mask 속성 테스트"""
        # 마스크 없음
        ref_img = ReferenceImage(
            rgb=np.zeros((100, 100, 3), dtype=np.uint8),
            depth=np.zeros((100, 100), dtype=np.float32),
            mask=None,
            view_angle="0",
            folder_name="test"
        )
        assert not ref_img.has_mask

        # 마스크 있음
        ref_img_with_mask = ReferenceImage(
            rgb=np.zeros((100, 100, 3), dtype=np.uint8),
            depth=np.zeros((100, 100), dtype=np.float32),
            mask=np.zeros((100, 100), dtype=np.uint8),
            view_angle="0",
            folder_name="test"
        )
        assert ref_img_with_mask.has_mask

    def test_image_size_property(self):
        """image_size 속성 테스트"""
        ref_img = ReferenceImage(
            rgb=np.zeros((480, 640, 3), dtype=np.uint8),
            depth=np.zeros((480, 640), dtype=np.float32)
        )
        assert ref_img.image_size == (480, 640)


class TestReferenceImageSet:
    """ReferenceImageSet 클래스 테스트"""

    def test_len(self):
        """길이 테스트"""
        images = [
            ReferenceImage(
                rgb=np.zeros((100, 100, 3), dtype=np.uint8),
                depth=np.zeros((100, 100), dtype=np.float32),
                view_angle=str(angle)
            )
            for angle in [0, 45, 90]
        ]
        image_set = ReferenceImageSet(images=images, object_name="test")
        assert len(image_set) == 3

    def test_iter(self):
        """반복 테스트"""
        images = [
            ReferenceImage(
                rgb=np.zeros((100, 100, 3), dtype=np.uint8),
                depth=np.zeros((100, 100), dtype=np.float32),
                view_angle=str(angle)
            )
            for angle in [0, 45]
        ]
        image_set = ReferenceImageSet(images=images)

        count = 0
        for img in image_set:
            assert isinstance(img, ReferenceImage)
            count += 1
        assert count == 2

    def test_getitem(self):
        """인덱싱 테스트"""
        images = [
            ReferenceImage(
                rgb=np.zeros((100, 100, 3), dtype=np.uint8),
                depth=np.zeros((100, 100), dtype=np.float32),
                view_angle=str(angle)
            )
            for angle in [0, 45, 90]
        ]
        image_set = ReferenceImageSet(images=images)

        assert image_set[0].view_angle == "0"
        assert image_set[1].view_angle == "45"
        assert image_set[2].view_angle == "90"

    def test_view_angles_property(self):
        """view_angles 속성 테스트"""
        images = [
            ReferenceImage(
                rgb=np.zeros((100, 100, 3), dtype=np.uint8),
                depth=np.zeros((100, 100), dtype=np.float32),
                view_angle=str(angle)
            )
            for angle in [0, 45, 90]
        ]
        image_set = ReferenceImageSet(images=images)

        angles = image_set.view_angles
        assert angles == ["0", "45", "90"]


class TestAngleExtraction:
    """각도 추출 테스트"""

    @pytest.fixture
    def test_data_path(self):
        """테스트 데이터 경로"""
        path = Path("/root/fursys_img_251229/extraction")
        if not path.exists():
            pytest.skip("Test data not available")
        return path

    def test_angle_extraction_from_folder_name(self, test_data_path):
        """폴더 이름에서 각도 추출"""
        loader = ReferenceImageLoader(str(test_data_path))

        test_cases = [
            ("20251229_093820_front", "0"),
            ("20251229_094115_45", "45"),
            ("20251229_094410_90", "90"),
            ("20251229_094705_135", "135"),
            ("20251229_095000_180", "180"),
            ("20251229_095255_225", "225"),
            ("20251229_095550_270", "270"),
            ("20251229_095845_315", "315"),
        ]

        for folder_name, expected_angle in test_cases:
            detected = loader._extract_angle_from_folder(folder_name)
            assert detected == expected_angle, f"Expected {expected_angle}, got {detected} for {folder_name}"


class TestExcludePatterns:
    """제외 패턴 테스트"""

    @pytest.fixture
    def test_data_path(self):
        """테스트 데이터 경로"""
        path = Path("/root/fursys_img_251229/extraction")
        if not path.exists():
            pytest.skip("Test data not available")
        return path

    def test_exclude_test_folders(self, test_data_path):
        """_test 폴더 제외 확인"""
        loader = ReferenceImageLoader(str(test_data_path), exclude_patterns=['_test'])

        for folder in loader.view_folders:
            assert '_test' not in folder.name


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
