import argparse

import tensorflow as tf
import numpy as np
from tqdm import tqdm
from ..blocks import YoloV3
from .utils import mean_average_precision
from .const import (
    LEARNING_RATE,
    WEIGHT_DECAY,
    ANCHORS,
    ANCHORS_MASKS,
    NUM_EPOCHS,
    NMS_IOU_THRESH,
    CONF_THRESHOLD,
    SAVE_MODEL,
)
from .utils import (
    get_evaluation_bboxes,
    save_checkpoint,
    check_class_accuracy,
)
from .loss import YoloLoss
from .dataset import YoloDataset


@tf.function
def train_step(x, y, model, optimizer, loss_fn, scaled_anchor):
    with tf.GradientTape() as tape:
        out = model(x, training=True)
        regularization_loss = tf.reduce_sum(model.losses)
        losses = []
        for idx in range(3):
            losses.append(loss_fn[idx](out[idx], y[idx]))

        total_loss = tf.reduce_sum(losses) + regularization_loss

    gradients = tape.gradient(total_loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    return total_loss


def train_fn(dataset, model, optimizer, loss_fn, scaled_anchor):
    loop = tqdm(dataset, leave=True)

    losses = []

    avg_loss = tf.keras.metrics.Mean('loss', dtype=tf.float32)
    for batch, (x, y) in enumerate(loop):
        y0, y1, y2 = y[0], y[1], y[2]
        loss = train_step(x, [y0, y1, y2], model, optimizer, loss_fn, scaled_anchor)

        losses.append(loss.numpy())

        mean_loss = np.mean(np.nan_to_num(losses))
        avg_loss.update_state(mean_loss)

        loop.set_postfix(loss=mean_loss)

    return np.nan_to_num(losses)


def main(obj_path):
    num_classes = 1
    model = YoloV3(num_classes=num_classes, is_training=True)()
    optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE, weight_decay=WEIGHT_DECAY)


    dataset_loader = YoloDataset(obj_path, ANCHORS)
    dataset = dataset_loader()

    dataset = dataset.shuffle(buffer_size=1000)
    dataset = dataset.batch(1)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    scaled_anchors = (
        tf.cast(tf.gather(ANCHORS, ANCHORS_MASKS) / 416, dtype=tf.float32)
        * tf.tile(tf.expand_dims(tf.expand_dims(tf.constant([13, 26, 52], dtype=tf.float32), axis=1), axis=2), [1, 3, 2])
    )
    loss_fn = []
    for idx in range(3):
      loss_fn.append(YoloLoss(anchors=scaled_anchors[idx]))

    model.compile(optimizer=optimizer, loss=loss_fn)
    total_loss = None
    for epoch in range(NUM_EPOCHS):
        losses = train_fn(dataset, model, optimizer, loss_fn, scaled_anchors)

        if total_loss is None:
          total_loss = losses
        else:
          total_loss = np.concatenate((total_loss, losses), axis=0)

        if epoch != 0 and epoch % 5 == 0:
          np.save("/content/drive/MyDrive/IASR/losses.npy", total_loss)

        if False:
            tqdm.write("Validation")
            check_class_accuracy(model, dataset_loader, dataset_loader.config['valid'], scaled_anchors)

            pred_boxes, true_boxes = get_evaluation_bboxes(dataset, model, iou_threshold=NMS_IOU_THRESH, anchors=ANCHORS, threshold=CONF_THRESHOLD)
            mpeval = mean_average_precision(pred_boxes, true_boxes, iou_threshold=NMS_IOU_THRESH, num_classes=num_classes)
            tqdm.write(f"Mean Average Precision: {mpeval:.4f}")

            if SAVE_MODEL:
                save_checkpoint(model, optimizer, dataset_loader.config['backup'])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Example of using argparse.')
    parser.add_argument('path', type=str, help='obj.data path.')

    args = parser.parse_args()
    main(args.path)
