import os 
import logging 
from pathlib import Path

logging.basicConfig(level=logging.INFO,format='[%(asctime)s]: %(message)s: ' )

list_of_files=[
    "src/__init__.py",
    "src/helper.py",
    "src/prompt.py",
    ".env",
    "requirements.txt",
    "setup.py",
    "research/trails.ipynb",
    "app.py"
]

for files in list_of_files:
    file_path = Path(files)
    print("The files/filepaths are",file_path)
    filedir,filename = os.path.split(file_path) # to split folders and its files

    if filedir !="": #to create directory
        os.makedirs(filedir,exist_ok=True)
        logging.info(f"Creating file directory using template.py {filedir} for the files {filename}")

    if (not os.path.exists(file_path) or (os.path.getsize(file_path == 0))): #to create files under folders
        with open(file_path,"w") as f:
            pass
            logging.info(f"Creating files for: {file_path}")

    else:
        logging.info(f"{filename} already exists!!")

