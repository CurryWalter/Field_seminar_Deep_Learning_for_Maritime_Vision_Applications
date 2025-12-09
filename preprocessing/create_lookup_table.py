import os
import re
import pandas as pd

def create_lookup_table(path_to_fish_image='../data/fish_image'):

    # path needs to be relative to your cwd bcs idk htf python does relative path lmao
    id_pattern = r'(\d+)(?=\.png)'
    trajectory_pattern =  r'\d+'
    df = pd.DataFrame()
    ids = list()
    images = list()
    fish_ids = list()
    trajectory_ids = list()
    relative_path = list()
    for fish_id in os.listdir(path_to_fish_image):
        for image in os.listdir(f'{path_to_fish_image}/{fish_id}'):
            uuid = re.search(id_pattern, image)
            trajectory = re.search(trajectory_pattern, image)
            fish_ids.append(fish_id)
            ids.append(uuid.group(1))
            images.append(image)
            trajectory_ids.append(trajectory.group(0))
            relative_path.append(f'{path_to_fish_image}/{fish_id}/{image}')
    df.index = ids
    df.loc[:,'image_name'] = images
    df.loc[:,'label'] = fish_ids
    df.loc[:,'trajectory'] = trajectory_ids
    df.loc[:,'relative_path'] = relative_path
    return df

if __name__ == '__main__':
    create_lookup_table().to_csv('../data/fish_lookup_table.csv')