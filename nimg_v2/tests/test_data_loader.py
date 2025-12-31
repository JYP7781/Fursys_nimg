"""
데이터 로더 테스트
"""

import pytest
import numpy as np
import tempfile
import os
import cv2
import pandas as pd

from nimg_v2.data.data_loader import DataLoader, FrameData
from nimg_v2.data.frame_synchronizer import FrameSynchronizer


class TestFrameData:
    """FrameData 테스트"""

    def test_frame_data_creation(self):
        """FrameData 생성 테스트"""
        rgb = np.zeros((480, 640, 3), dtype=np.uint8)
        depth = np.ones((480, 640), dtype=np.float32) * 1.5

        frame = FrameData(
            frame_idx=0,
            rgb=rgb,
            depth=depth,
            timestamp=0.0
        )

        assert frame.frame_idx == 0
        assert frame.rgb.shape == (480, 640, 3)
        assert frame.depth.shape == (480, 640)
        assert not frame.has_imu

    def test_frame_data_with_imu(self):
        """IMU 데이터가 있는 FrameData 테스트"""
        rgb = np.zeros((480, 640, 3), dtype=np.uint8)
        depth = np.ones((480, 640), dtype=np.float32)
        accel = np.array([0.0, -9.81, 0.0])
        gyro = np.array([0.01, 0.02, 0.03])

        frame = FrameData(
            frame_idx=0,
            rgb=rgb,
            depth=depth,
            timestamp=0.0,
            accel=accel,
            gyro=gyro
        )

        assert frame.has_imu
        np.testing.assert_array_equal(frame.accel, accel)
        np.testing.assert_array_equal(frame.gyro, gyro)

    def test_depth_valid_ratio(self):
        """유효 depth 비율 테스트"""
        rgb = np.zeros((100, 100, 3), dtype=np.uint8)
        depth = np.ones((100, 100), dtype=np.float32) * 2.0
        depth[:50, :] = 0.0  # 절반은 무효

        frame = FrameData(frame_idx=0, rgb=rgb, depth=depth, timestamp=0.0)

        assert 0.45 < frame.depth_valid_ratio < 0.55


class TestDataLoader:
    """DataLoader 테스트"""

    @pytest.fixture
    def temp_dataset(self):
        """임시 테스트 데이터셋 생성"""
        with tempfile.TemporaryDirectory() as tmpdir:
            # RGB 이미지 생성
            for i in range(10):
                rgb = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
                cv2.imwrite(os.path.join(tmpdir, f'color_{i:06d}.png'), rgb)

                depth = np.random.randint(500, 2000, (480, 640), dtype=np.uint16)
                cv2.imwrite(os.path.join(tmpdir, f'depth_{i:06d}.png'), depth)

            # IMU 데이터 생성
            imu_data = {
                'timestamp': [],
                'type': [],
                'x': [],
                'y': [],
                'z': []
            }
            for i in range(100):
                imu_data['timestamp'].append(i * 10.0)
                imu_data['type'].append('accel' if i % 2 == 0 else 'gyro')
                imu_data['x'].append(np.random.randn())
                imu_data['y'].append(np.random.randn())
                imu_data['z'].append(np.random.randn())

            df = pd.DataFrame(imu_data)
            df.to_csv(os.path.join(tmpdir, 'imu_data.csv'), index=False)

            yield tmpdir

    def test_data_loader_init(self, temp_dataset):
        """DataLoader 초기화 테스트"""
        loader = DataLoader(temp_dataset)

        assert len(loader) == 10
        assert loader.has_imu

    def test_load_frame(self, temp_dataset):
        """프레임 로드 테스트"""
        loader = DataLoader(temp_dataset)
        frame = loader.load_frame(0)

        assert isinstance(frame, FrameData)
        assert frame.frame_idx == 0
        assert frame.rgb is not None
        assert frame.depth is not None

    def test_iteration(self, temp_dataset):
        """이터레이션 테스트"""
        loader = DataLoader(temp_dataset)
        frames = list(loader)

        assert len(frames) == 10
        for i, frame in enumerate(frames):
            assert frame.frame_idx == i

    def test_invalid_index(self, temp_dataset):
        """잘못된 인덱스 테스트"""
        loader = DataLoader(temp_dataset)

        with pytest.raises(IndexError):
            loader.load_frame(100)

    def test_get_metadata(self, temp_dataset):
        """메타데이터 테스트"""
        loader = DataLoader(temp_dataset)
        metadata = loader.get_metadata()

        assert 'num_frames' in metadata
        assert metadata['num_frames'] == 10
        assert metadata['has_imu'] == True


class TestFrameSynchronizer:
    """FrameSynchronizer 테스트"""

    def test_time_window(self):
        """시간 윈도우 계산 테스트"""
        sync = FrameSynchronizer(fps=30.0, time_window_mode='center')

        t_start, t_end = sync.get_time_window(0)
        assert t_start == pytest.approx(0.0, abs=0.001)
        assert t_end == pytest.approx(1/30.0, abs=0.001)

    def test_forward_mode(self):
        """Forward 모드 테스트"""
        sync = FrameSynchronizer(fps=30.0, time_window_mode='forward')

        t_start, t_end = sync.get_time_window(1)
        assert t_start == pytest.approx(1/30.0, abs=0.001)
        assert t_end == pytest.approx(2/30.0, abs=0.001)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
