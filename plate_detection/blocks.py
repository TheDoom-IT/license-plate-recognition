import tensorflow as tf
from plate_detection.yolo.const import IMAGE_SIZE
import logging

# Setup logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class CNNBlock(tf.keras.layers.Layer):
    def __init__(self, out_channels, strides=1, has_bn=True, name="CNNBlock", **kwargs):
        super().__init__(name=name)
        self.padding = tf.keras.layers.ZeroPadding2D(((1, 0), (1, 0)))
        self.conv = tf.keras.layers.Conv2D(
            out_channels, 
            strides=strides, 
            padding="same" if strides == 1 else "valid",
            use_bias=not has_bn, 
            kernel_regularizer=tf.keras.regularizers.l2(0.0005), 
            **kwargs,
        )
        self.bn = tf.keras.layers.BatchNormalization()
        self.leaky_relu = tf.keras.layers.LeakyReLU(0.1)
        self.strides = strides
        self.has_bn = has_bn

    def call(self, inputs):
        if self.strides != 1:
            inputs = self.padding(inputs)
        x = self.conv(inputs)
        if self.has_bn:
            x = self.bn(x)
            x = self.leaky_relu(x)
        return x


class ResidualBlock(tf.keras.layers.Layer):
    def __init__(self, channels, use_residual=True, num_repeats=1, name="Residual", **kwargs):
        super().__init__(name=name)
        self.layers = []
        self.use_residual = use_residual
        self.num_repeats = num_repeats
        self.add = tf.keras.layers.Add()
        for _ in range(self.num_repeats):
            self.layers.append(
                tf.keras.Sequential([
                    CNNBlock(channels // 2, kernel_size=(1, 1), name=f"{name}_Conv2D1"),
                    CNNBlock(channels, kernel_size=(3, 3), name=f"{name}_Conv2D2")
                ])
            )

    def call(self, inputs):
        x = inputs
        for layer in self.layers:
            if self.use_residual:
                x = self.add([x, layer(x)])
            else:
                x = layer(x)        
        return x

class ScalePrediction(tf.keras.layers.Layer):
    def __init__(self, in_channels, num_classes, name="Output") -> None:
        super().__init__(name=name)
        self.prediction = tf.keras.Sequential([
            CNNBlock(2*in_channels, kernel_size=(3,3), name=f"{name}_Conv2D1"),
            CNNBlock((num_classes + 5)*3, has_bn=False, kernel_size=(1,1), name=f"{name}_Conv2D2"),
        ]
        )
        self.num_classes = num_classes

    def call(self, inputs):
        output = self.prediction(inputs)
        output = tf.keras.layers.Reshape((output.shape[1], output.shape[2], 3, self.num_classes + 5))(output)
        
        return output
  

class YoloV3:
    def __init__(self, in_channels=3, num_classes=20):
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.layers = self.create_conv_layers()

    def __call__(self):
        outputs = []
        skip_connections = []
        x = inputs = tf.keras.layers.Input([IMAGE_SIZE, IMAGE_SIZE, self.in_channels], name='input')

        for idx, layer in enumerate(self.layers):
            if idx in self.output_idx:
                outputs.append(layer(x))
                continue

            x = layer(x)

            if idx in self.skip_connection_idx:
                skip_connections.append(x)
            
            elif isinstance(layer, tf.keras.layers.UpSampling2D):
                x = tf.keras.layers.Concatenate()([x, skip_connections.pop()])
        
        return tf.keras.Model(inputs, outputs, name="YoloV3")

    def create_conv_layers(self):

        layers = [
            # Darknet layers
            
            # batch_normalize=1; filters=32; size=3; stride=1; pad=1; activation=leaky
            CNNBlock(32, kernel_size=(3, 3), name="DarkNet_Conv2D_0"),
            # batch_normalize=1; filters=64; size=3; stride=2; pad=1; activation=leaky
            CNNBlock(64, kernel_size=(3, 3), strides=2, name="DarkNet_Conv2D_1"),
            # batch_normalize=1; filters=32; size=1; stride=1; pad=1; activation=leaky
            # batch_normalize=1; filters=64; size=3; stride=1; pad=1; activation=leaky
            # shortcut=-3
            ResidualBlock(64, num_repeats=1, name="DarkNet_Residual_1"),
            # batch_normalize=1; filters=128; size=3; stride=2; pad=1; activation=leaky
            CNNBlock(128, kernel_size=(3,3), strides=2, name="DarkNet_Conv2D_2"),
            # batch_normalize=1; filters=64; size=1; stride=1; pad=1; activation=leaky
            # batch_normalize=1; filters=128; size=3; stride=1; pad=1; activation=leaky
            # shortcut=-3
            ResidualBlock(128, num_repeats=2, name="DarkNet_Residual_2"),
            # batch_normalize=1; filters=256; size=3; stride=2; pad=1; activation=leaky
            CNNBlock(256, kernel_size=(3,3), strides=2, name="DarkNet_Conv2D_3"),
            # batch_normalize=1; filters=128; size=1; stride=1; pad=1; activation=leaky
            # batch_normalize=1; filters=256; size=3; stride=1; pad=1; activation=leaky
            # shortcut=-3
            ResidualBlock(256, num_repeats=8, name="DarkNet_Residual_skip1"),
            # batch_normalize=1; filters=256; size=3; stride=2; pad=1; activation=leaky
            CNNBlock(512, kernel_size=(3,3), strides=2, name="DarkNet_Conv2D_4"),
            # batch_normalize=1; filters=256; size=1; stride=1; pad=1; activation=leaky
            # batch_normalize=1; filters=512; size=3; stride=1; pad=1; activation=leaky
            # shortcut=-3
            ResidualBlock(512, num_repeats=8, name="DarkNet_Residual_skip2"),
            # batch_normalize=1; filters=1024; size=3; stride=2; pad=1; activation=leaky
            CNNBlock(1024, kernel_size=(3,3), strides=2, name="DarkNet_Conv2D_5"),
            # batch_normalize=1; filters=512; size=1; stride=1; pad=1; activation=leaky
            # batch_normalize=1; filters=1024; size=3; stride=1; pad=1; activation=leaky
            # shortcut=-3
            ResidualBlock(1024, num_repeats=4, name="DarkNet_Residual_3"),

            # Darknet layers end here

            # Yolo layers

            # batch_normalize=1; filters=512; size=1; stride=1; pad=1; activation=leaky
            CNNBlock(512, kernel_size=(1,1), name="Yolo_Conv2D_0"),
            # batch_normalize=1; filters=1024; size=3; stride=1; pad=1; activation=leaky
            CNNBlock(1024, kernel_size=(3,3),  name="Yolo_Conv2D_1"),
            # batch_normalize=1; filters=512; size=1; stride=1; pad=1; activation=leaky
            # batch_normalize=1; filters=1024; size=3; stride=1; pad=1; activation=leaky
            ResidualBlock(1024, num_repeats=1, use_residual=False, name="Yolo_Residual_1"),
            # batch_normalize=1; filters=256; size=1; stride=1; pad=1; activation=leaky
            CNNBlock(512, kernel_size=(1,1),  name="Yolo_Conv2D_2"),
            # batch_normalize=1; filters=1024; size=3; stride=1; pad=1; activation=leaky
            # batch_normalize=0; filters=(num_classes + 5) * 3; size=1; stride=1; pad=1; activation=linear
            ScalePrediction(512, self.num_classes, name="Yolo_Output_1"), # Scale 1 output 15

            # batch_normalize=1; filters=256; size=1; stride=1; pad=1; activation=leaky
            CNNBlock(256, kernel_size=(1,1),  name="Yolo_Conv2D_3"),
            # upsample
            tf.keras.layers.UpSampling2D(size=2, interpolation="nearest", name="Yolo_UpSample_1"),

            # batch_normalize=1; filters=256; size=1; stride=1; pad=1; activation=leaky
            CNNBlock(256, kernel_size=(1,1),  name="Yolo_Conv2D_4"),
            # batch_normalize=1; filters=512; size=3; stride=1; pad=1; activation=leaky
            CNNBlock(512, kernel_size=(3,3),  name="Yolo_Conv2D_5"),
            # batch_normalize=1; filters=256; size=1; stride=1; pad=1; activation=leaky
            # batch_normalize=1; filters=512; size=3; stride=1; pad=1; activation=leaky
            ResidualBlock(512, num_repeats=1, use_residual=False, name="Yolo_Residual_2"),
            # batch_normalize=1; filters=256; size=1; stride=1; pad=1; activation=leaky
            CNNBlock(256, kernel_size=(1,1),  name="Yolo_Conv2D_6"),
            # batch_normalize=1; filters=1024; size=3; stride=1; pad=1; activation=leaky
            # batch_normalize=0; filters=(num_classes + 5) * 3; size=1; stride=1; pad=1; activation=linear
            ScalePrediction(256, self.num_classes, name="Yolo_Output_2"), # Scale 2 output 22

            # batch_normalize=1; filters=128; size=1; stride=1; pad=1; activation=leaky
            CNNBlock(128, kernel_size=(1,1),  name="Yolo_Conv2D_7"),

            # upsample
            tf.keras.layers.UpSampling2D(size=2, interpolation="nearest", name="Yolo_UpSample_2"),

            # batch_normalize=1; filters=128; size=1; stride=1; pad=1; activation=leaky
            CNNBlock(128, kernel_size=(1,1),  name="Yolo_Conv2D_8"),
            # batch_normalize=1; filters=256; size=3; stride=1; pad=1; activation=leaky
            CNNBlock(256, kernel_size=(3,3),  name="Yolo_Conv2D_9"),
            # batch_normalize=1; filters=128; size=1; stride=1; pad=1; activation=leaky
            # batch_normalize=1; filters=256; size=3; stride=1; pad=1; activation=leaky
            ResidualBlock(256, num_repeats=1, use_residual=False, name="Yolo_Residual_3"),
            # batch_normalize=1; filters=128; size=1; stride=1; pad=1; activation=leaky
            CNNBlock(128, kernel_size=(1,1),  name="Yolo_Conv2D_10"),
            # batch_normalize=1; filters=1024; size=3; stride=1; pad=1; activation=leaky
            # batch_normalize=0; filters=(num_classes + 5) * 3; size=1; stride=1; pad=1; activation=linear
            ScalePrediction(128, self.num_classes, name="Yolo_Output_3"), # Scale 3 output 29
        ]

        self.output_idx = [15, 22, 29]
        self.skip_connection_idx = [6, 8]

        return layers