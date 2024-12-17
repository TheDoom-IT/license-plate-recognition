import subprocess
import os
import platform
from PIL import Image

import shutil
import json
from dataclasses import dataclass


BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

@dataclass
class Box:
    x1: int
    y1: int
    x2: int
    y2: int
    conf: float


class PlateDetectionService:
    def __init__(self, filename: str):
        self.filename = filename
        self.image = Image.open(filename)
        self.im = self.image.copy()
        # check if we have the correct cofiguration for detection

        self.yolo_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "yolov3-test.cfg")
        if not os.path.exists(self.yolo_path):
            raise FileNotFoundError("Could not find the yolov3-test.cfg.")

        self.weight_path =  os.path.join(BASE_DIR, ".weights", "yolov3-custom_final.weights")
        if not os.path.exists(self.weight_path):
            raise FileNotFoundError("Could not find the yolov3-custom_final.weights.")
        
        if platform.system() == "Windows":
            self.darknet_path = os.path.join(BASE_DIR, ".darknet", "darknet.exe")
            if not os.path.exists(self.darknet_path):
                raise FileNotFoundError("Could not find compiled darknet.exe.")
        elif platform.system() == "Linux":
            self.darknet_path = os.path.join(BASE_DIR, ".darknet", "darknet")
            if not os.path.exists(self.darknet_path):
                raise FileNotFoundError("Could not find complied darknet.")
        else:
            raise ValueError("Unsupported platform.")
        

        # copy some files if not exist
        with open(os.path.join(BASE_DIR, ".dataset", "obj.data"), "r") as file:
            conf = file.read()
        
        if platform.system() == "Windows":
            conf.replace(".dataset/obj.names", "dataset\\obj.names")
        else:
            conf.replace(".dataset/obj.names", "dataset/obj.names")
        
        with open(os.path.join(BASE_DIR, ".darknet", "obj.data"), "w") as file:
            file.write(conf)

        shutil.copy(os.path.join(BASE_DIR, ".dataset", "obj.names"), os.path.join(BASE_DIR, ".darknet", "obj.names"))

    def preprocess_image(self, output_dir):
        
        self.im.thumbnail((416, 416), Image.LANCZOS)

        padded_img = Image.new("RGB", (416, 416), (255, 255, 255))

        x = (416 - self.im.size[0]) // 2
        y = (416 - self.im.size[1]) // 2

        padded_img.paste(self.im, (x, y))
        padded_img.save(os.path.join(output_dir, "temp.jpg"))

    
    def get_true_bbox(self, bbox):
        tw, th = self.image.size[0], self.image.size[1]

        ratio = min(416 / tw, 416 / th)

        x_off = (416 - (tw * ratio)) // 2
        y_off = (416 - (th * ratio)) // 2

        x1, y1, x2, y2 = bbox

        x1 = (x1 - x_off) / ratio
        y1 = (y1 - y_off) / ratio
        x2 = (x2 - x_off) / ratio
        y2 = (y2 - y_off) / ratio

        
        x1 = max(0, min(x1, tw))
        x2 = max(0, min(x2, tw))
        y1 = max(0, min(y1, th))
        y2 = max(0, min(y2, th))

        return round(x1), round(y1), round(x2), round(y2)
    
        
    def detect(self) -> tuple[Image.Image, Box]:
        # TODO: implement plate detection
        command = [
            'darknet.exe', 
            "detector", 
            "test", 
            f'dataset\\obj.data', 
            f'"{self.yolo_path}"',
            f'"{self.weight_path}"',
            f'temp.jpg',
            "-ext_output"
            "-dont_show",
            "-out ",
            "result.json",
        ]
        work_dir = os.path.join(BASE_DIR, '.darknet')
        try:
            os.chdir(f"{work_dir}")
            self.preprocess_image(work_dir)
            subprocess.run(" ".join(command),  check=True, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            
            with open("result.json", "r") as f:
                result = json.load(f)
            
        except subprocess.CalledProcessError as e:
            print("Error:", e)
        os.chdir("..")

        box = result[0]['objects'][0]

        x1 = round((box["relative_coordinates"]["center_x"] - box["relative_coordinates"]["width"]/2) * 416)
        y1 = round((box["relative_coordinates"]["center_y"] - box["relative_coordinates"]["height"]/2) * 416)
        x2 = round((box["relative_coordinates"]["center_x"] + box["relative_coordinates"]["width"]/2) * 416)
        y2 = round((box["relative_coordinates"]["center_y"] + box["relative_coordinates"]["height"]/2) * 416)

        x1, y1, x2, y2 = self.get_true_bbox([x1, y1, x2, y2])
        
        plate = self.image.crop((x1, y1, x2, y2))

        return plate, Box(x1, y1, x2, y2, float(box["confidence"]))