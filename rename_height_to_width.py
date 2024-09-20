from glob import glob
import os
import sys
import platform
from he_database_generate import height_to_width
import pandas as pd

base_dir = '/root/shared-storage/shaoxingyu/workspace_backup/gsvqddb_train'
path_list = sorted(glob(f"{base_dir}/**/*.png", recursive=True))

for path in path_list:
    meta = path.split('@')
    height_str = meta[2]
    height = float(height_str)
    width = height_to_width(height)

    # new_path = path.replace('@' + height_str, '@' + str(width))
    new_path = f'{meta[0]}@{meta[1]}@{width}@{meta[3]}@{meta[4]}@{meta[5]}@{meta[6]}'

    os.rename(path, new_path)