import keras
from PIL import Image, ImageOps
import numpy as np
import os


class RecognitionService:
    EXPECTED_WIDTH = 27
    EXPECTED_HEIGHT = 40
    CLASSES = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    model: keras.api.models.Sequential

    def __init__(self):
        # keras.config.disable_interactive_logging()
        self.model = self.__load_model()

    @staticmethod
    def __load_model():
        script_dir = os.path.dirname(__file__)
        model_name = "recognition_model.keras"
        model_path = os.path.join(script_dir, "..", "..", ".weights", model_name)

        return keras.models.load_model(model_path)

    def recognize(self, characters: list[np.ndarray]) -> str:
        result = ""

        input = np.array([self.__adjust_image_size(character) for character in characters])
        predictions = self.model.predict(input)
        for prediction in predictions:
            predicted_class = np.argmax(prediction)
            result += self.CLASSES[predicted_class]

        return result

    def __adjust_image_size(self, image: np.ndarray) -> np.ndarray:
        """
        Adjust the image size to the expected size of the neural network.
        Add padding if necessary.
        """
        img = Image.fromarray(image)
        # resize keeping the aspect ratio
        img = ImageOps.contain(img, (self.EXPECTED_WIDTH, self.EXPECTED_HEIGHT), Image.Resampling.NEAREST)

        # add padding if image is smaller than the expected size
        result = np.array(img)
        if result.shape[0] < self.EXPECTED_HEIGHT:
            new_array = np.zeros((self.EXPECTED_HEIGHT, result.shape[1]), dtype=np.uint8)
            start = (self.EXPECTED_HEIGHT - result.shape[0]) // 2
            new_array[start:(result.shape[0] + start), :] = result
            result = new_array

        if result.shape[1] < self.EXPECTED_WIDTH:
            new_array = np.zeros((result.shape[0], self.EXPECTED_WIDTH), dtype=np.uint8)
            start = (self.EXPECTED_WIDTH - result.shape[1]) // 2
            new_array[:, start:(result.shape[1] + start)] = result
            result = new_array

        return result
