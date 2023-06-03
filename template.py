import os
from pathlib import Path
import logging

project_name = "CNNClassifier"
list_of_files = [
    ".github/workflows/.gitkeep",
    f"src/{project_name}/__init__.py",
    f"src/{project_name}/components/__init__.py",
    f"src/{project_name}/utils/__init__.py",
    f"src/{project_name}/utils/common.py",
    f"src/{project_name}/logging/__init__.py",
    f"src/{project_name}/config/__init__.py",
    f"src/{project_name}/config/configuration.py",
    f"src/{project_name}/pipeline/__init__.py",
    f"src/{project_name}/entity/__init__.py",
    f"src/{project_name}/constants/__init__.py",

    "config/config.yaml",
    "dvc.yaml",
    "params.yaml",
    "app.py",
    "main.py",
    "requirements.txt",
    "setup.py",
    "Dockerfile",
    "notebooks/trials.ipynb",
    "templates/index.html"
]

for file_path in list_of_files:
    filepath = Path(file_path)
    filedir, filename = os.path.split(Path(filepath))
    if not os.path.exists(Path(filepath)) and filedir != "":
        os.makedirs(filedir, exist_ok=True)
        logging.info(f"Creating directory: {filedir} for file: {filename}")

    if not os.path.exists(filepath):  ## may overwrte the prexisiting files
        with open(filepath, "w") as f:
            pass  # create an empty file
            logging.info(f"Creating empty file: {filepath}")
    else:
        logging.info(f"{filename} already exists")
