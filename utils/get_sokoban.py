"""

"""

import os
import gdown
import zipfile


# def GetSokoban():
#     url = 'https://github.com/avivg7/sokoban-so.git'
#     output = 'sokoban.zip'
#     dir_path = 'sokoban'
#
#     # Only download and extract the file if it hasn't been done already
#     if not os.path.isdir(dir_path):
#         gdown.download(url, output, quiet=False)
#
#         with zipfile.ZipFile(output, 'r') as zip_ref:
#             zip_ref.extractall()

import os
import subprocess

import os
import subprocess
import zipfile

def GetSokoban():
    url = 'https://github.com/avivg7/sokoban-so.git'
    dir_path = 'sokoban'

    # Only clone the repository if it hasn't been done already
    if not os.path.isdir(dir_path):
        subprocess.check_call(['git', 'clone', url, dir_path])

    # Path to the zip file
    zip_file = os.path.join(dir_path, 'Compress.zip')

    # Check if the zip file exists
    if os.path.isfile(zip_file):
        # Extract the zip file
        with zipfile.ZipFile(zip_file, 'r') as zip_ref:
            zip_ref.extractall(dir_path)
    else:
        print(f"No zip file found at {zip_file}")

