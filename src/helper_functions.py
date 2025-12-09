from preprocessing.create_lookup_table import create_lookup_table
def match_name_to_label(image_name):
    # input: image name output: label
    df = create_lookup_table()
    for _, row in df.iterrows():
        if row['image_name'] == image_name:
            return row['label']
        continue
    return None

def match_name_to_trajectory(image_name):
    # input: image name output: trajectory
    df = create_lookup_table()
    for _, row in df.iterrows():
        if row['image_name'] == image_name:
            return row['trajectory']
        continue
    return None
