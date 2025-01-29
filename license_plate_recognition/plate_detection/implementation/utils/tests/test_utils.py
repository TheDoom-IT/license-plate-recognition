import unittest
import tensorflow as tf
import numpy as np
from ...yolo.utils import calculate_iou, calculate_area_iou, mean_average_precision, non_max_suppression, cell_to_bboxes

class TestCalculateIou(unittest.TestCase):
    def setUp(self):
        self.epsilon = 1e-3

    def test_complete_overlap(self):
        box1 = np.array([2, 2, 4, 4])
        box2 = np.array([2, 2, 4, 4])
        iou = calculate_iou(box1, box2, box_format="corners")
        self.assertTrue(np.abs(iou - 1) < self.epsilon)

    def test_no_overlap(self):
        box1 = np.array([1, 1, 3, 3])
        box2 = np.array([4, 4, 6, 6])
        iou = calculate_iou(box1, box2, box_format="corners")
        self.assertTrue(np.abs(iou - 0) < self.epsilon)

    def test_partial_overlap(self):
        box1 = np.array([1, 1, 3, 3])
        box2 = np.array([2, 2, 4, 4])
        iou = calculate_iou(box1, box2, box_format="corners")
        self.assertTrue(0 < iou < 1)

    def test_midpoint_format_overlap(self):
        # In midpoint format: [center_x, center_y, width, height]
        box1 = np.array([2, 2, 2, 2])  # Converts to corners: [1, 1, 3, 3]
        box2 = np.array([3, 3, 2, 2])  # Converts to corners: [2, 2, 4, 4]
        iou = calculate_iou(box1, box2, box_format="midpoint")
        self.assertTrue(0 < iou < 1)

    def test_midpoint_format_no_overlap(self):
        box1 = np.array([1, 1, 2, 2])  # Converts to corners: [0, 0, 2, 2]
        box2 = np.array([3, 3, 2, 2])  # Converts to corners: [2, 2, 4, 4]
        iou = calculate_iou(box1, box2, box_format="midpoint")
        self.assertTrue(np.abs(iou - 0) < self.epsilon)


class TestCalculateAreaIou(unittest.TestCase):
    def test_full_overlap(self):
        box1 = np.array([2, 3])  # Area = 6
        box2 = np.array([2, 3])  # Area = 6, same area, full overlap
        iou = calculate_area_iou(box1, box2)
        self.assertEqual(iou, 1)

    def test_partial_overlap(self):
        box1 = np.array([2, 2])  # Area = 4
        box2 = np.array([3, 3])  # Area = 9
        iou = calculate_area_iou(box1, box2)
        self.assertEqual(iou, 4 / 9)  # Intersection is min(4, 9), union is 9 (since 4 is completely within 9)


class TestNonMaxSuppression(unittest.TestCase):

    def test_filter_by_probability(self):
        prediction = [
            (1, 0.9, 1, 1, 3, 3),  # High probability
            (1, 0.1, 2, 2, 4, 4)   # Low probability
        ]
        iou_threshold = 0.5
        probability_threshold = 0.5
        results = non_max_suppression(prediction, iou_threshold, probability_threshold)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0][1], 0.9)

    def test_suppression_by_iou(self):
        prediction = [
            (1, 0.9, 1, 1, 3, 3),  # High probability, will be compared for IOU
            (1, 0.85, 1, 1, 3, 3)  # Slightly lower probability, identical location
        ]
        iou_threshold = 0.5
        probability_threshold = 0.5
        results = non_max_suppression(prediction, iou_threshold, probability_threshold)
        self.assertEqual(len(results), 1)  # Only one should remain after NMS
        self.assertEqual(abs(results[0][1]-0.9) < 1e-3, True)  # Higher probability should remain

    def test_no_suppression_needed(self):
        prediction = [
            (1, 0.9, 1, 1, 3, 3),  # High probability, non-overlapping
            (2, 0.85, 4, 4, 6, 6)  # Different class, also high probability
        ]
        iou_threshold = 0.5
        probability_threshold = 0.5
        results = non_max_suppression(prediction, iou_threshold, probability_threshold)
        self.assertEqual(len(results), 2)  # Both should remain as they do not overlap


class TestMeanAveragePrecision(unittest.TestCase):
    def setUp(self):
        self.epsilon = 1e-4

    def test_all_correct_one_class(self):
        preds = [
            [0, 0, 0.9, 0.55, 0.2, 0.3, 0.2],
            [0, 0, 0.8, 0.35, 0.6, 0.3, 0.2],
            [0, 0, 0.7, 0.8, 0.7, 0.2, 0.2],
        ]
        targets = [
            [0, 0, 0.9, 0.55, 0.2, 0.3, 0.2],
            [0, 0, 0.8, 0.35, 0.6, 0.3, 0.2],
            [0, 0, 0.7, 0.8, 0.7, 0.2, 0.2],
        ]
        mAP = 1
        mean_avg_prec = mean_average_precision(
            preds,
            targets,
            iou_threshold=0.5,
            box_format="midpoint",
            num_classes=1,
        )
        self.assertTrue(abs(mAP - mean_avg_prec) < self.epsilon)

    def test_all_correct_batch(self):
        preds = [
            [1, 0, 0.9, 0.55, 0.2, 0.3, 0.2],
            [0, 0, 0.8, 0.35, 0.6, 0.3, 0.2],
            [0, 0, 0.7, 0.8, 0.7, 0.2, 0.2],
        ]
        targets = [
            [1, 0, 0.9, 0.55, 0.2, 0.3, 0.2],
            [0, 0, 0.8, 0.35, 0.6, 0.3, 0.2],
            [0, 0, 0.7, 0.8, 0.7, 0.2, 0.2],
        ]
        correct_mAP = 1
        mean_avg_prec = mean_average_precision(
            preds,
            targets,
            iou_threshold=0.5,
            box_format="midpoint",
            num_classes=1,
        )
        self.assertTrue(abs(correct_mAP - mean_avg_prec) < self.epsilon)

    def test_all_wrong_class(self):
        preds = [
            [0, 1, 0.9, 0.55, 0.2, 0.3, 0.2],
            [0, 1, 0.8, 0.35, 0.6, 0.3, 0.2],
            [0, 1, 0.7, 0.8, 0.7, 0.2, 0.2],
        ]
        targets = [
            [0, 0, 0.9, 0.55, 0.2, 0.3, 0.2],
            [0, 0, 0.8, 0.35, 0.6, 0.3, 0.2],
            [0, 0, 0.7, 0.8, 0.7, 0.2, 0.2],
        ]
        mAP = 0
        mean_avg_prec = mean_average_precision(
            preds,
            targets,
            iou_threshold=0.5,
            box_format="midpoint",
            num_classes=2,
        )
        self.assertTrue(abs(mAP - mean_avg_prec) < self.epsilon)

    def test_one_inaccurate_box(self):
        preds = [
            [0, 0, 0.9, 0.15, 0.25, 0.1, 0.1],
            [0, 0, 0.8, 0.35, 0.6, 0.3, 0.2],
            [0, 0, 0.7, 0.8, 0.7, 0.2, 0.2],
        ]

        targets = [
            [0, 0, 0.9, 0.55, 0.2, 0.3, 0.2],
            [0, 0, 0.8, 0.35, 0.6, 0.3, 0.2],
            [0, 0, 0.7, 0.8, 0.7, 0.2, 0.2],
        ]
        mAP = 5 / 18
        mean_avg_prec = mean_average_precision(
            preds,
            targets,
            iou_threshold=0.5,
            box_format="midpoint",
            num_classes=1,
        )
        self.assertTrue(abs(mAP - mean_avg_prec) < self.epsilon)


class TestCellToBBoxes(unittest.TestCase):
    def test_training_mode(self):
        predictions = tf.random.normal([1, 3, 13, 13,  7])  # Example shape: [batch, num_of_anchor, grid, grid, features]
        anchors = tf.constant([[0.1, 0.2], [0.3, 0.4], [0.6, 0.8]])  # Example anchors
        output = cell_to_bboxes(predictions, anchors, 13, is_training=True)
        self.assertEqual(output.shape.as_list(), [1, 3*13*13, 6])  # Check output shape, adjust based on actual output

    def test_inference_mode(self):
        predictions = tf.random.normal([1, 3, 3, 3, 9])
        anchors = tf.constant([[0.1, 0.2], [0.3, 0.4], [0.6, 0.8]])
        output = cell_to_bboxes(predictions, anchors, 3, is_training=False)
        self.assertEqual(output.shape.as_list(), [1, 3 * 3 * 3, 6])  # Check output shape, adjust based on actual output
