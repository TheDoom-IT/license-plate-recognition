# Polish license plate detection and recognition

This project was created using Poetry. It can be
downloaded from [here](https://python-poetry.org/).
Poetry handles all of your dependencies and virtual environments.

## Project setup

The project requires Python 3.10.x and Poetry 1.8.x to work.
1. Install Python 3.10.x from [here](https://www.python.org/downloads/).
2. Install Poetry 1.8.x from [here](https://python-poetry.org/docs/1.8/).
3. Install dependencies running the following commands:
    ```bash
    poetry install
    ```
4. Download weights required to run plate detection and recognition models.
    You can download them from [here](https://drive.google.com/file/d/1KzqjkfQSQEKHL-vWejDEkhcVQibAKTYu/view?usp=sharing).
    Place them in the `.weights` directory.
5. (Optional) Download the dataset used to train the models.
    You can download it from [here](https://drive.google.com/file/d/1eHPUN2NzDRs4menl6IeKkKJboXPTckzR/view?usp=sharing).
    Place them in the `.dataset` directory.

After those steps your codebase should have the following structure:
```
license-plate-recognition/
|-- .gitignore
|-- .dataset/  <-- HERE
|-- .weights/  <-- HERE
|   |-- yolov3-custom_final.weights
|   |-- recognition_model.keras
|-- README.md
|-- license-plate-recognition/
|-- poetry.lock
|-- pyproject.toml
```

## Running the project

The application can be run using the following command:

```bash
# Run the project
poetry run python -m license_plate_recognition.main <path_to_image>
# Example
poetry run python -m license_plate_recognition.main example1.jpg
```

The application will read an image from the specified path and display the license plate number present in the image.

## Virtual environment

Poetry automatically create Python virtual environment for you.
Run `poetry env info` to see details about the virtual environment.

## Dependencies

New dependencies can be added using `poetry add <package>`.
Read more about it [here](https://python-poetry.org/docs/basic-usage/#specifying-dependencies).
