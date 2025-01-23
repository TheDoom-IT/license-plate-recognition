import tensorflow as tf
from plate_detection.yolo.utils import calculate_iou_dt_box

class YoloLoss(tf.keras.losses.Loss):
    def __init__(self, anchors):
        self.anchors = anchors
        super().__init__()
        self.mse = tf.keras.losses.MeanSquaredError()
        self.bce = tf.keras.losses.BinaryCrossentropy()
        self.entropy =  tf.keras.losses.SparseCategoricalCrossentropy()

    def call(self, predictions, target):
        obj = target[..., 4] == 1
        noobj = target[..., 4] == 0
        has_obj = tf.reduce_any(obj)

        # For No Object Loss
        no_object_loss = self.bce(
            target[..., 4:5][noobj], tf.sigmoid(predictions[..., 4:5][noobj])
        )

        # For Object Loss
        anchors = tf.reshape(self.anchors, (1, 1, 1, 3, 2))
        box_preds = tf.concat([
            tf.sigmoid(predictions[..., :2]),
            tf.exp(predictions[..., 2:4]) * anchors
        ], axis=-1)
        ious = calculate_iou_dt_box(box_preds[obj], target[..., :4][obj])

        object_loss = self.mse(
            ious * target[..., 4:5][obj], tf.sigmoid(predictions[..., 4:5][obj])
        )
        object_loss = tf.cond(has_obj, lambda: object_loss, lambda: 0.0)

        # For Box Coordinates Loss
        box_predictions = tf.concat([
            tf.sigmoid(predictions[..., :2]),
            predictions[..., 2:],
        ], axis=-1)
        target_box = tf.concat([
            target[..., :2],
            (1e-16 + tf.math.log(target[..., 2:4]) / anchors),
            target[..., 4:]  # Class labels
        ], axis=-1)
        box_loss = self.mse(
            target_box[..., :4][obj], box_predictions[..., :4][obj]
        )
        box_loss = tf.cond(has_obj, lambda: box_loss, lambda: 0.0)

        # For Class Loss
        class_loss = self.entropy(
            target[..., 5][obj], predictions[..., 5:][obj]
        )
        class_loss = tf.cond(has_obj, lambda: class_loss, lambda: 0.0)

        return class_loss + box_loss + object_loss + no_object_loss
