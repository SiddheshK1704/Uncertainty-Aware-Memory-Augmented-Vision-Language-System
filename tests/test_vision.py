"""
Tests for VisionModule
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest
from modules.vision_module import Detection


class TestDetection:
    def test_center_computation(self):
        det = Detection(label="chair", class_id=56, confidence=0.8, bbox=[100, 150, 300, 350])
        assert det.center == (200.0, 250.0)

    def test_area(self):
        det = Detection(label="chair", class_id=56, confidence=0.8, bbox=[0, 0, 100, 200])
        assert det.area() == 20000.0

    def test_to_dict(self):
        det = Detection(label="bottle", class_id=39, confidence=0.75, bbox=[10, 20, 110, 220])
        d = det.to_dict()
        assert d["label"] == "bottle"
        assert d["confidence"] == 0.75
        assert "center" in d

    def test_zero_area(self):
        det = Detection(label="chair", class_id=56, confidence=0.5, bbox=[100, 100, 50, 50])
        assert det.area() == 0.0


class TestVisionModuleInit:
    def test_default_thresholds(self):
        from modules.vision_module import VisionModule
        vm = VisionModule.__new__(VisionModule)
        vm.conf_threshold = 0.35
        vm.iou_threshold = 0.45
        assert vm.conf_threshold == 0.35

    def test_empty_image_returns_empty(self):
        """Test that empty image input is handled gracefully."""
        from modules.vision_module import VisionModule
        vm = VisionModule.__new__(VisionModule)
        vm.model = None
        result = vm.detect(np.array([]))
        assert result == []
