import os

from data_augmenation import *

def fetch_train_annotations(splits):
    return pd.read_csv(f'../splits/{splits}/train.csv')

def augment_class(class_name, n_samples, splits):
    annotations = fetch_train_annotations(splits)
    filtered_annotations = annotations[annotations['label'] == class_name]

    if len(filtered_annotations) < n_samples:
        n_samples = len(filtered_annotations)

    selected_samples = filtered_annotations.sample(n=n_samples, random_state=0)

    augmented_images = []

    for idx, img_path in enumerate(selected_samples['relative_path']):
        image = Image.open(img_path)
        augmented_image = apply_random_augmentation(image, seed=0)
        augmented_images.append(augmented_image)

        base_name = os.path.basename(img_path)
        new_name = f"aug_{idx}_{base_name}"
        new_path = os.path.join(f'../splits/{splits}/train/', new_name)

        augmented_image.save(new_path)
        augmented_annotations = pd.DataFrame({
            'relative_path': [new_path],
            'label': [class_name]
        })

        annotations = pd.concat([annotations, augmented_annotations])
        augmented_images.append(new_path)

    annotations.to_csv(f'../splits/{splits}/train.csv', index=False)

    return augmented_images

