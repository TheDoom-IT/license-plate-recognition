# Polish license plate detection and recognition

This project was created using Poetry. It can be
downloaded from [here](https://python-poetry.org/).

Poetry handles all of your dependencies and virtual environments.

## Project setup
The project requires Python 3.10 and Poetry to work.
To install all dependencies run the following commands:
```bash
# Install dependencies
poetry install
```

To run the application use the following command:
```bash
# Run the project
poetry run python -m license_plate_recognition.main <path_to_image>
```
The application will read the image from the given path and display license plate number present on the image.

## Virtual environment
Poetry automatically create Python virtual environment for you.
Run `poetry env info` to see details about the virtual environment.

## Dependencies
New dependencies can be added using `poetry add <package>`.
Read more about it [here](https://python-poetry.org/docs/basic-usage/#specifying-dependencies).
