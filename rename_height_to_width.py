from glob import glob
import os
import sys
import platform
from he_database_generate import height_to_width
import pandas as pd
from tqdm import trange

if platform.system() == "Windows":
    slash = '\\'
else:
    slash = '/'

base_dir = '/root/shared-storage/shaoxingyu/workspace_backup/gsvqddb_train'
path_list = sorted(glob(f"{base_dir}/**/*.png", recursive=True))
# csv_list = sorted(glob(f"{base_dir}/**/*.csv", recursive=True))
# for csv_path in csv_list:
#     df = pd.read_csv(csv_path, header=0)
#     df.insert(loc=2, column='photo_width_meters', value=None)
#     df.to_csv(csv_path, mode='w', index=False, header=True)

tbar = trange(len(path_list))
for path in path_list:
    csv_tmp = path.split(slash)
    tbar.set_description(csv_tmp[-1])
    tbar.update()
    meta = path.split('@')
    if len(meta) != 7:
        continue
    csv_path = path.replace(slash + csv_tmp[-1], '.csv').replace('Images', 'Dataframes')
    df = pd.read_csv(csv_path, header=0)
    idx = df.loc[(df["loc_x"] == int(meta[-3])) & (df["loc_y"] == int(meta[-2])) & (df["flight_height"] == round(float(meta[2]),2))].index.tolist()[0]
    height_str = meta[2]
    height = float(height_str)
    width = height_to_width(height)
    df.loc[idx, 'photo_width_meters'] = round(width,7)
    df.to_csv(csv_path, mode='w', index=False, header=True)

    # new_path = path.replace('@' + height_str, '@' + str(width))
    new_path = f'{meta[0]}@{meta[1]}@{width:.7f}@{meta[2]}@{meta[3]}@{meta[4]}@{meta[5]}@{meta[6]}'

    os.rename(path, new_path)