import os
import re
import pandas as pd

def create_lookup_table(path_to_fish_image='../data/fish_image'):

    # path needs to be relative to your cwd bcs idk htf python does absolute path lmao
    id_pattern = r'(\d+)(?=\.png)'
    trajectory_pattern =  r'\d+'
    df = pd.DataFrame()
    ids = list()
    images = list()
    fish_ids = list()
    trajectory_ids = list()
    absolute_paths = list()
    for fish_id in os.listdir(path_to_fish_image):
        for image in os.listdir(f'{path_to_fish_image}/{fish_id}'):
            uuid = re.search(id_pattern, image)
            trajectory = re.search(trajectory_pattern, image)
            fish_ids.append(fish_id)
            ids.append(uuid.group(1))
            images.append(image)
            trajectory_ids.append(trajectory.group(0))
            absolute_paths.append(f'{path_to_fish_image}/{fish_id}/{image}')
    df.index = ids
    df['image'] = images
    df['lable'] = fish_ids
    df['trajectory'] = trajectory_ids
    df['absolute_path'] = absolute_paths
    return df

print(create_lookup_table())
