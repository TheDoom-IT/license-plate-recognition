BATCH_SIZE = 32
IMAGE_SIZE = 416
NUM_CLASSES = 1
LEARNING_RATE = 1e-5
WEIGHT_DECAY = 1e-4
NUM_EPOCHS = 2
CONF_THRESHOLD = 0.5
MAP_IOU_THRESH = 0.5
NMS_IOU_THRESH = 0.5
SCALES = [IMAGE_SIZE // 32, IMAGE_SIZE // 16, IMAGE_SIZE // 8]
LOAD_MODEL = False
SAVE_MODEL = True

ANCHORS = [
    (10, 13), (16,30), (37, 12), 
    (57, 16), (70, 24), (96, 29), 
    (102, 51), (156, 198), (373, 326)
]
ANCHORS_MASKS = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]

CLASSES = ["license-plate"]
YOLO_LAYERS = [
    "DarkNet_Conv2D_0",
    "DarkNet_Conv2D_1",
    "DarkNet_Residual_1",
    "DarkNet_Conv2D_2",
    "DarkNet_Residual_2",
    "DarkNet_Conv2D_3",
    "DarkNet_Residual_skip1",
    "DarkNet_Conv2D_4",
    "DarkNet_Residual_skip2",
    "DarkNet_Conv2D_5",
    "DarkNet_Residual_3",
    "Yolo_Conv2D_0",
    "Yolo_Conv2D_1",
    "Yolo_Residual_1",
    "Yolo_Conv2D_2",
    "Yolo_Output_1",
    "Yolo_Conv2D_3",
    "Yolo_Conv2D_4",
    "Yolo_Conv2D_5",
    "Yolo_Residual_2",
    "Yolo_Conv2D_6",
    "Yolo_Output_2",
    "Yolo_Conv2D_7",
    "Yolo_Conv2D_8",
    "Yolo_Conv2D_9",
    "Yolo_Residual_3",
    "Yolo_Conv2D_10",
    "Yolo_Output_3",
]