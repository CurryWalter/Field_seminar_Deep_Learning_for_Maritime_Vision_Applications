import os

from data_augmenation import *

def fetch_train_annotations(splits):
    return pd.read_csv(f'../splits/{splits}/train.csv')

def augment_class(class_name, n_samples):
    annotations = fetch_train_annotations('baseline')
    filtered_annotations = annotations[annotations['label'] == class_name]

    if len(filtered_annotations) < n_samples:
        n_samples = len(filtered_annotations)

    selected_samples = filtered_annotations.sample(n=n_samples, random_state=0)

    augmented_images = []

    for img_path in selected_samples['relative_path']:
        image = Image.open(img_path)
        augmented_image = apply_random_augmentation(image, seed=0)
        augmented_images.append(augmented_image)

        base_name = os.path.basename(img_path)
        new_name = f"aug_{base_name}"
        new_path = os.path.join('../splits/baseline/train/', new_name)

        augmented_image.save(new_path)
        augmented_images.append(new_path)
    return augmented_images

augmented_images = augment_class('fish_01', 3)
