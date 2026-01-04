import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from torch.nn import functional as F
from tqdm import tqdm
from sklearn import metrics
from matplotlib import pyplot as plt
from src.dataset import FishyDataset
from src.models import get_ResNet50, get_ViTb16
from training.configs import base_transforms

NUM_CLASSES = 23
def main(model, image_size=(100, 100)):
    device ='cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    model.eval()

    annotation_path = '../splits/baseline/test.csv'

    dataset = FishyDataset(annotation_path, base_transforms(image_size))
    dataloader = DataLoader(dataset, batch_size=512, shuffle=False)

    labels, preds = [], []
    for img, label in tqdm(dataloader):
        img = img.to(device)
        label = label.to(device)

        with torch.no_grad():
            output = model(img)

        # output = F.softmax(output, dim=1)
        output = torch.argmax(output, dim=1)
        preds += (output.detach().cpu().numpy().tolist())
        labels += (label.cpu().numpy().tolist())

    print(labels)
    print(preds)
    results = metrics.classification_report(labels, preds)
    with open('Vit_baseline_results.txt', 'w') as f:
        f.write(f"{results} \n")

    cf_matrix = metrics.confusion_matrix(labels, preds)
    cf_matrix = cf_matrix.astype(float) / cf_matrix.sum(axis=1)[:, np.newaxis]


    # visualize confusion matrix
    plt.imshow(cf_matrix, interpolation='nearest', cmap='viridis')

    plt.xticks(np.arange(23), np.arange(23), rotation=90)
    plt.yticks(np.arange(23), np.arange(23))

    plt.colorbar()
    plt.xlabel('Predicted Class')
    plt.ylabel('True Class')
    plt.title('Vit Baseline')

    plt.savefig('confusion_matrix_Vit_baseline.png')
    plt.show()


if __name__ == '__main__':
    # for resnet
    """model = get_ResNet50(NUM_CLASSES)
    weights = torch.load('../models/ResNet50/ResNet50_first_run_baseline_dataset')"""

    model = get_ViTb16(NUM_CLASSES, 224)
    weights = torch.load('../models/ViTb16/ViTb16_first_run_baseline_dataset')
    model.load_state_dict(weights)

    main(model, (224, 224))