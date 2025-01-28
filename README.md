# Polish license plate detection and recognition

This project was created using Poetry. It can be
downloaded from [here](https://python-poetry.org/).

Poetry handles all of your dependencies and virtual environments.

## Running the project

The application can be run using the following command:

```bash
# Run the project
poetry run python -m license_plate_recognition.main <path_to_image>
```

The application will read an image from the specified path and display the license plate number present in the image.

## Project setup

The project requires Python 3.10 and Poetry to work.
To install all dependencies run the following commands:

```bash
# 1.  Install dependencies
poetry install
```

After installing the Python package, You need to extract [extras.zip](https://drive.google.com/file/d/1LHBDSbSFVhdKbvgZ_74m8OmUoxJ-Yow6/view?usp=sharing) file
to the root directory of the project.This contains the weights and dataset folder which is not
included in the code. you need to have directory structure like this:

```
license-plate-recognition/
|   poetry.lock
|   pyproject.toml
|   .gitignore
|   README.md
|___.dataset/   <-- HERE
|___.weights/  <-- HERE
|___character_segmentation/
|___license-plate-recognition/
|___plate_detection/
|___scripts/
```

All set? Execute the above command and watch our project come to life!

## Virtual environment

Poetry automatically create Python virtual environment for you.
Run `poetry env info` to see details about the virtual environment.

## Dependencies

New dependencies can be added using `poetry add <package>`.
Read more about it [here](https://python-poetry.org/docs/basic-usage/#specifying-dependencies).
