"""
nimg_v2 패키지 설치 스크립트
"""

from setuptools import setup, find_packages

setup(
    name='nimg_v2',
    version='2.0.0',
    description='RGB + Depth + IMU 기반 상대적 속도/각도 변화량 측정 시스템',
    author='Fursys Image Processing Team',
    packages=find_packages(),
    python_requires='>=3.8',
    install_requires=[
        'numpy>=1.21.0',
        'pandas>=1.3.0',
        'opencv-python>=4.5.0',
        'torch>=1.10.0',
        'ultralytics>=8.0.0',
        'filterpy>=1.4.5',
        'open3d>=0.15.0',
        'matplotlib>=3.4.0',
        'pyyaml>=5.4.0',
        'scipy>=1.7.0',
    ],
    extras_require={
        'dev': [
            'pytest>=7.0.0',
            'pytest-cov>=3.0.0',
        ],
    },
    entry_points={
        'console_scripts': [
            'nimg_v2=nimg_v2.main:main',
        ],
    },
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
    ],
)
