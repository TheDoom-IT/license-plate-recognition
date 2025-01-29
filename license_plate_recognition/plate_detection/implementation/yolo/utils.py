import tensorflow as tf
from tqdm import tqdm
import pickle
from .const import (
    IMAGE_SIZE,
    ANCHORS,
    ANCHORS_MASKS,
    NMS_IOU_THRESH,
    CONF_THRESHOLD,
    DARKNET_LAYERS,
    SCALES,
)
import numpy as np
from PIL.Image import Image
from collections import Counter



def get_coordinates(box):
    w, h =  box[..., 2],  box[..., 3]
    x1 = box[..., 0] - w/2
    x2 = box[..., 0] + w/2
    y1 = box[..., 1] - h/2
    y2 = box[..., 1] + h/2

    return tf.stack([x1, y1, x2, y2], axis=-1)



def calculate_iou(box1, box2):
    box1 = get_coordinates(box1)
    box2 = get_coordinates(box2)

    top_left = tf.math.maximum(box1[..., None, :2], box2[..., :2])
    bottom_right = tf.math.minimum(box1[..., None, 2:], box2[..., 2:])

    diff = tf.clip_by_value(bottom_right - top_left, 0, 1)

    if diff.shape[0] < 2:
        return tf.ones(1)

    intersection = tf.math.multiply(diff[..., 0], diff[..., 1])

    a1 = (box1[..., 2] - box1[..., 0]) * (box1[..., 3] - box1[..., 1])
    a2 = (box2[..., 2] - box2[..., 0]) * (box2[..., 3] - box2[..., 1])

    return intersection / (a1[:, None] + a2 - intersection)


def non_max_suppression(predictions, iou_threshold, probability_threshold):
    def nms_single(prediction):
        boxes = tf.convert_to_tensor(prediction)

        mask = boxes[:, 1] > probability_threshold
        boxes = tf.boolean_mask(boxes, mask)

        indexes = tf.reverse(tf.argsort(boxes[:, 1]), axis=[0])
        boxes = tf.gather(boxes, indexes)

        classes = tf.cast(boxes[:, 0], dtype=tf.int32)
        bboxes = boxes[:, 2:]
        ious = calculate_iou(bboxes, bboxes)
        ious = ious - tf.eye(boxes.shape[0])

        keep = tf.cast(tf.ones(boxes.shape[0]), dtype=tf.bool)

        for index, (iou, _class) in enumerate(zip(ious, classes)):
            if not keep[index]:
                continue

            condition = (iou > iou_threshold) & (_class == classes)
            keep = keep & ~condition

        return tf.boolean_mask(boxes, keep, axis=0)

    return tf.map_fn(nms_single, predictions, dtype=tf.float32)


def cell_to_bboxes(predictions, anchors, output_scale, is_inference):
    x = predictions[..., 0:1]
    y = predictions[..., 1:2]
    scale = predictions[..., 2:4]

    num_anchors = len(anchors)
    if is_inference:
        anchors = tf.reshape(anchors, shape=(1, 1, 1, num_anchors, 2))
        x = tf.sigmoid(x)
        y = tf.sigmoid(y)
        scale = tf.exp(scale) * anchors * output_scale
        score = tf.sigmoid(predictions[..., 4:5])
        _class = tf.expand_dims(tf.cast(tf.argmax(predictions[..., 5:], axis=-1), dtype=x.dtype), -1)
    else:
        score = predictions[..., 4:5]
        _class = predictions[..., 5:6]


    grid = tf.expand_dims(tf.tile(
        tf.reshape(tf.range(output_scale, dtype=x.dtype), [1, 1, output_scale, 1]),
        [predictions.shape[0], output_scale, 1, 3]
    ), axis=-1)


    x = (1 / output_scale) * (x + grid)
    y = (1 / output_scale) * (y + tf.transpose(grid, [0, 2, 1, 3, 4]))
    scale = (1 / output_scale) * scale

    return tf.reshape(
        tf.concat([_class, score, x, y, scale], axis=-1),
        (predictions.shape[0], num_anchors * output_scale * output_scale, 6)
    )


def calculate_area_iou(box1, box2):
    intersection = tf.math.minimum(box1[..., 0], box2[..., 0]) * tf.math.minimum(
        box1[..., 1], box2[..., 1]
    )
    union = (
        box1[..., 0] * box1[..., 1] + box2[..., 0] * box2[..., 1] - intersection
    )
    return intersection / union


def get_evaluation_bboxes(
    loader,
    model,
    threshold,
):
    # Ensure model is in evaluation mode
    model.trainable = False

    predicted_boxes = []
    true_boxes = []
    train_idx = 0

    anchors = tf.convert_to_tensor(ANCHORS, dtype=tf.float32) / IMAGE_SIZE

    for batch_idx, (x, labels) in enumerate(tqdm(loader)):
        predictions = model(x, training=False)

        #predicted_boxes = get_bboxes(predictions, is_inference=False)
        true_bboxes = tf.convert_to_tensor(cell_to_bboxes(labels[0], tf.gather(anchors, ANCHORS_MASKS[0]), SCALES[0], False))

        batch_size = x.shape[0]


        for idx in range(batch_size):
            im_pred = (predictions[0][idx:idx+1], predictions[1][idx:idx+1], predictions[2][idx:idx+1])
            if isinstance(predicted_boxes, list):
                nms = tf.convert_to_tensor(get_bboxes(im_pred, is_inference=False))
                nms = tf.reshape(nms, [nms.shape[1], nms.shape[2]])
                if nms.shape[0] > 0:
                    predicted_boxes = tf.concat([tf.cast(tf.fill((nms.shape[0], 1), train_idx), dtype=tf.float32), nms], axis=-1)
            else:
                nms = tf.convert_to_tensor(get_bboxes(im_pred, is_inference=False))
                nms = tf.reshape(nms, [nms.shape[1], nms.shape[2]])
                if nms.shape[0] > 0:
                    nms = tf.concat([tf.cast(tf.fill((nms.shape[0], 1), train_idx),  dtype=tf.float32), nms], axis=-1)
                    predicted_boxes = tf.concat([predicted_boxes, nms], axis=0)

            if isinstance(true_boxes, list):
                true_box = true_bboxes[idx:idx+1]
                mask = true_box[..., 1] > threshold
                true_box = true_box[mask]
                box_sum = tf.reduce_sum(true_box, axis=1)
                _, indices = tf.unique(box_sum)
                indices, _ = tf.unique(indices)
                true_box = tf.gather(true_box, indices)
                if true_box.shape[0] > 0:
                    true_boxes = tf.concat([tf.cast(tf.fill((true_box.shape[0], 1), train_idx), dtype=tf.float32), true_box], axis=-1)
            else:
                true_box = true_bboxes[idx]
                mask = true_box[..., 1] > threshold
                true_box = true_box[mask]
                box_sum = tf.reduce_sum(true_box, axis=1)
                _, indices = tf.unique(box_sum)
                indices, _ = tf.unique(indices)
                true_box = tf.gather(true_box, indices)
                if true_box.shape[0] > 0:
                    true_box = tf.concat([tf.cast(tf.fill((true_box.shape[0], 1), train_idx),  dtype=tf.float32), true_box], axis=-1)
                    true_boxes = tf.concat([true_boxes, true_box], axis=0)

            train_idx += 1

    # Set model back to training mode
    model.trainable = True

    return predicted_boxes, true_boxes

def check_class_accuracy(model, loader, threshold):
    tot_class_preds, correct_class = 0, 0
    tot_noobj, correct_noobj = 0, 0
    tot_obj, correct_obj = 0, 0
    model.trainable = False

    for idx, (x, y) in enumerate(tqdm(loader)):
        # Forward pass
        predictions = model(x, training=False)

        for i in range(3):  # Assuming model outputs at 3 scales as in YOLO
            obj = y[i][..., 4] == 1
            noobj = y[i][..., 4] == 0

            correct_class += tf.reduce_sum(tf.cast(tf.argmax(predictions[i][obj][..., 5:], axis=-1) == tf.cast(y[i][obj][..., 5], tf.int64), tf.int32))
            tot_class_preds += tf.reduce_sum(tf.cast(obj, tf.int32))

            obj_preds = tf.sigmoid(predictions[i][..., 4]) > threshold
            correct_obj += tf.reduce_sum(tf.cast(obj_preds[obj] == tf.cast(y[i][..., 4], tf.bool)[obj], tf.int32))
            tot_obj += tf.reduce_sum(tf.cast(obj, tf.int32))
            correct_noobj += tf.reduce_sum(tf.cast(obj_preds[noobj] == tf.cast(y[i][..., 4], tf.bool)[noobj], tf.int32))
            tot_noobj += tf.reduce_sum(tf.cast(noobj, tf.int32))

    correct_class = tf.cast(correct_class, tf.float32)
    tot_class_preds = tf.cast(tot_class_preds, tf.float32)
    correct_obj = tf.cast(correct_obj, tf.float32)
    tot_obj = tf.cast(tot_obj, tf.float32)
    correct_noobj = tf.cast(correct_noobj, tf.float32)
    tot_noobj = tf.cast(tot_noobj, tf.float32)
    model.trainable = True

    print(f"Class accuracy is: {(correct_class / (tot_class_preds + 1e-16)) * 100:.2f}%")
    print(f"No obj accuracy is: {(correct_noobj / (tot_noobj + 1e-16)) * 100:.2f}%")
    print(f"Obj accuracy is: {(correct_obj / (tot_obj + 1e-16)) * 100:.2f}%")



def get_mean_std(loader):
    channels_sum, channels_sqrd_sum, num_batches = 0, 0, 0

    for data, _ in tqdm(loader):
        channels_sum += tf.reduce_mean(data, axis=[0, 1, 2])
        channels_sqrd_sum += tf.reduce_mean(tf.square(data), axis=[0, 1, 2])
        num_batches += 1

    mean = channels_sum / num_batches
    std = tf.sqrt(channels_sqrd_sum / num_batches - tf.square(mean))

    return mean, std


def save_checkpoint(model, optimizer, filename):
    print("=> Saving checkpoint")
    model.save_weights(filename)
    # Save optimizer state
    with open(filename + '_optimizer.pkl', 'wb') as f:
        pickle.dump(optimizer.get_config(), f)


def load_checkpoint(checkpoint_file, model, optimizer, lr):
    print("=> Loading checkpoint")
    model.load_weights(checkpoint_file)

    # Load optimizer state

    with open(checkpoint_file + '_optimizer.pkl', 'rb') as f:
        opt_config = pickle.load(f)
    optimizer.from_config(opt_config)

    # Update learning rate - this requires custom handling based on optimizer type
    if 'learning_rate' in opt_config:
        opt_config['learning_rate'] = lr
    optimizer.from_config(opt_config)



def read_darknet_weights(file_path, header_size=5):
    file = open(file_path, 'rb')

    header = np.fromfile(file, dtype=np.int32, count=header_size)

    return header, file

def get_conv_layer(layer):
    batch_norm = None
    cn_layer = layer.conv
    if layer.has_bn:
        batch_norm = layer.bn

    return {"cn": cn_layer, "bn": batch_norm}

def load_weights(layers, weights):
    for idx, layer in enumerate(layers):
        filters = layer['cn'].filters
        kernel_size = layer['cn'].kernel_size[0]
        input_dim = layer['cn'].input_spec.axes[-1]
        if layer['bn'] is None:
            conv_bias = np.fromfile(weights, dtype=np.float32, count=filters)
        else:
            bn_weights = np.fromfile(weights, dtype=np.float32, count=4 * filters)
            bn_weights = bn_weights.reshape((4, filters))[[1, 0, 2, 3]]

        conv_shape = (filters, input_dim, kernel_size, kernel_size)
        conv_weights = np.fromfile(weights, dtype=np.float32, count=np.prod(conv_shape))
        conv_weights = conv_weights.reshape(conv_shape).transpose([2, 3, 1, 0])

        if layer['bn'] is None:
            layer['cn'].set_weights([conv_weights, conv_bias])
        else:
            layer['cn'].set_weights([conv_weights])
            layer['bn'].set_weights(bn_weights)

    assert len(weights.read()) == 0, 'failed to read all data'
    weights.close()


def load_yolo_weights(file_path, model, yolo_layers):
    _, weights = read_darknet_weights(file_path)
    layers = []
    for layer_name in yolo_layers:
        layer = model.get_layer(layer_name)
        if layer.name.find("Conv2D") != -1:
            layers.append(get_conv_layer(layer))

        if layer.name.find("Residual") != -1:
            for l in layer.layers:
                cn_layer1 = l.layers[0]
                cn_layer2 = l.layers[1]

                layers.append(get_conv_layer(cn_layer1))
                layers.append(get_conv_layer(cn_layer2))

        if layer.name.find("Output") != -1:
            cn_layer1 = layer.prediction.layers[0]
            cn_layer2 = layer.prediction.layers[1]

            layers.append(get_conv_layer(cn_layer1))
            layers.append(get_conv_layer(cn_layer2))

    load_weights(layers, weights)
    print("Successfull data write")


def load_image_as_tf(image):
    if isinstance(image, np.ndarray) or isinstance(image, Image):
        image = tf.convert_to_tensor(image)

        return image.shape[:2].as_list(), tf.image.resize(tf.expand_dims(image, axis=0) / 255, (IMAGE_SIZE, IMAGE_SIZE))
    elif isinstance(image, str):
        image = tf.image.decode_image(open(image, 'rb').read(), channels=3)
        return image.shape[:2].as_list(), tf.image.resize(
            tf.expand_dims(image, axis=0) / 255, (IMAGE_SIZE, IMAGE_SIZE),
        )
    else:
        raise ValueError(f"Can't process this param: {type(image)}")


def get_original_bbox(scale, bbox):
    hscale, wscale= np.asarray(scale) / IMAGE_SIZE
    x1, y1, x2, y2 = bbox
    return np.array([x1 * wscale, y1 * hscale, x2 * wscale, y2 * hscale], dtype=np.int32)


def get_bboxes(outputs, is_inference=True, threshold=CONF_THRESHOLD):
    anchors = np.array(ANCHORS, dtype=np.float32) / IMAGE_SIZE
    _outputs = []
    for idx, output in enumerate(outputs):
        _outputs.append(
            cell_to_bboxes(output, anchors[ANCHORS_MASKS[idx]], output.shape[2], True)
        )
    _outputs = tf.concat(_outputs, axis=1)

    nms = non_max_suppression(_outputs, iou_threshold=NMS_IOU_THRESH, probability_threshold=threshold)

    if is_inference:
        nms = tf.reshape(nms, nms.shape[1:])
        box, _class, score = ([], [], [])

        for pred in nms:
            # [class, score, x_center, y_center, width, height]
            c, s, x, y, w, h = pred
            box.append([x - w/2, y - h/2, x + w/2, y + h/2])
            _class.append(int(c))
            score.append(s)

        box = np.asarray(np.array(box) * IMAGE_SIZE, dtype=np.int32)
        _class = np.array(_class, dtype=np.int32)
        score = np.array(score, dtype=np.float32)

        return box, _class, score
    else:
        return nms




def calculate_iou_dt_box(box1, box2, box_format="midpoint"):
    # intersection area between different boxes
    if box_format == "midpoint":
        b1_x1, b1_y1 = box1[..., 0:1] - box1[..., 2:3]/2, box1[..., 1:2] - box1[..., 3:4]/2
        b1_x2, b1_y2 =  box1[..., 0:1] + box1[..., 2:3]/2, box1[..., 1:2] + box1[..., 3:4]/2

        b2_x1, b2_y1 = box2[..., 0:1] - box2[..., 2:3]/2, box2[..., 1:2] - box2[..., 3:4]/2
        b2_x2, b2_y2 =  box2[..., 0:1] + box2[..., 2:3]/2, box2[..., 1:2] + box2[..., 3:4]/2

    else:
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[..., 0:1], box1[..., 1:2], box1[..., 2:3], box1[..., 3:4]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[..., 0:1], box2[..., 1:2], box2[..., 2:3], box2[..., 3:4]


    x1 = tf.math.maximum(b1_x1, b2_x1)
    y1 = tf.math.maximum(b1_y1, b2_y1)
    x2 = tf.math.minimum(b1_x2, b2_x2)
    y2 = tf.math.minimum(b1_y2, b2_y2)

    intersection = tf.math.maximum(x2 - x1, 0) * tf.math.maximum(y2 - y1, 0)
    union = (
        tf.math.abs((b1_x2 - b1_x1) * (b1_y2 - b1_y1))
        + tf.math.abs((b2_x2 - b2_x1) * (b2_y2 - b2_y1))
        - intersection + 1e-6
    )
    # union area of boxes and centroids
    return intersection / union


def mean_average_precision(predictions, true_values, iou_threshold, num_classes=20):
    average_precision = []
    epsilon = 1e-6

    predictions = predictions.numpy()
    true_values = true_values.numpy()

    for _class in range(num_classes):
        detection_mask =predictions[..., 1] == float(_class)
        detections = predictions[detection_mask]

        true_value_mask = true_values[..., 1] == float(_class)
        ground_truths = true_values[true_value_mask]

        amount_bboxes = Counter(ground_truths[:, 0].astype(int))
        for key, value in amount_bboxes.items():
            amount_bboxes[key] = np.zeros(amount_bboxes[key])

        detection_idxs = np.argsort(detections[:, 2])[::-1]
        detections = detections[detection_idxs]
        true_positives = np.zeros(len(detections))
        false_positives = np.zeros(len(detections))
        total_true_bboxes = len(ground_truths)

        for i, detection in enumerate(detections):
            gt_mask = ground_truths[:, 0] == detection[0]
            ground_truth_im = ground_truths[gt_mask]

            best_iou = 0
            best_gt_idx = -1

            for idx, gt in enumerate(ground_truth_im):
                iou = calculate_iou_dt_box(tf.convert_to_tensor(detection[3:]), tf.convert_to_tensor(gt[3:]))

                if iou > best_iou:
                    best_iou = iou.numpy()
                    best_gt_idx = idx

            if best_iou > iou_threshold:
                if amount_bboxes[int(detection[0])][best_gt_idx] == 0:
                    true_positives[i] = 1
                    amount_bboxes[int(detection[0])][best_gt_idx] = 1
                else:
                    false_positives[i] = 1
            else:
                false_positives[i] = 1

        true_positive_cumsum = np.cumsum(true_positives, axis=0)
        false_positives_cumsum = np.cumsum(false_positives, axis=0)

        recall = true_positive_cumsum / (total_true_bboxes + epsilon)
        precision = true_positive_cumsum / (true_positive_cumsum + false_positives_cumsum + epsilon)
        precision = np.concatenate((tf.constant([1]), precision))
        recall = np.concatenate((tf.constant([0]), recall))

        average_precision.append(np.trapezoid(precision, recall))

    return sum(average_precision) / len(average_precision)


def freeze_darknet_layers(model):
    for layers in DARKNET_LAYERS:
        model.get_layer(layers).trainable = False
    return model
