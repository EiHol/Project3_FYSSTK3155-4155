import kagglehub, os, shutil, random, torch, cv2
from tqdm import tqdm
import numpy as np
import tensorflow as tf
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from PIL import Image 
import matplotlib.pyplot as plt

# fix result (adapted from Monshi et al 2021)
# https://github.com/MaramMonshi/CovidXrayNet/blob/main/DataAugmentation/1-resize.ipynb 
def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    tf.random.set_seed(seed)


def load_and_transform_data(SEED):

    seed_everything(SEED)

    # Copy to current directory
    target_dir = "chest_xray_data"

    if not os.path.exists(target_dir):

        # Download the dataset
        path = kagglehub.dataset_download("paultimothymooney/chest-xray-pneumonia")

        print("Downloaded to cache:", path)

        shutil.copytree(path, target_dir)
        print(f"Dataset copied to: {target_dir}")
        
    else:
        print(f"Dataset already exists at: {target_dir}")

    print("\nDataset ready at:", os.path.abspath(target_dir))

    dataset_path = 'chest_xray_data/chest_xray'
    new_dataset_path = 'chest_xray_data_split'

    if not os.path.exists(new_dataset_path):
        for split in ['train', 'val', 'test']:
            for cls in ['NORMAL', 'PNEUMONIA']:
                os.makedirs(f'{new_dataset_path}/{split}/{cls}', exist_ok=True)

        for cls in ['NORMAL', 'PNEUMONIA']:
            all_files = []
            for split in ['train', 'val', 'test']:
                source_folder = f'{dataset_path}/{split}/{cls}'
                files = os.listdir(source_folder)
                all_files.extend([(file, source_folder) for file in files])

            random.shuffle(all_files)

            train_files = all_files[:int(len(all_files)*0.8)]
            val_files = all_files[int(len(all_files)*0.8):int(len(all_files)*0.9)]
            test_files = all_files[int(len(all_files)*0.9):]

            for file, source_folder in train_files:
                dest = f'{new_dataset_path}/train/{cls}/{file}'
                shutil.copy(f'{source_folder}/{file}', dest)

            for file, source_folder in val_files:
                dest = f'{new_dataset_path}/val/{cls}/{file}'
                shutil.copy(f'{source_folder}/{file}', dest)

            for file, source_folder in test_files:
                dest = f'{new_dataset_path}/test/{cls}/{file}'
                shutil.copy(f'{source_folder}/{file}', dest)  

        print("\nDataset ready at:", os.path.abspath(new_dataset_path))
    
    else:
        print(f"Dataset already exists at: {new_dataset_path}\n")


    # Define the transforms to do for training data
    train_transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((480, 480)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(20),
        transforms.ColorJitter(brightness=0.3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    # Define the transforms to do for validation and testing data
    val_test_transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((480, 480)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    # Load datasets with progress bar
    def load_data(path, train_aug=True):
        transform = train_transform if train_aug else val_test_transform
        dataset = datasets.ImageFolder(path, transform=transform)
        
        images = []
        labels = []
        for i in tqdm(range(len(dataset)), desc=f"Loading {path.split('/')[-1]}"):
            img, label = dataset[i]
            images.append(img.numpy().squeeze())
            labels.append(label)
        
        return np.array(images), np.array(labels)

    X_train, y_train = load_data('chest_xray_data_split/train', train_aug=True)
    X_val, y_val = load_data('chest_xray_data_split/val', train_aug=False)
    X_test, y_test = load_data('chest_xray_data_split/test', train_aug=False)

    print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

    return X_train, y_train, X_val, y_val, X_test, y_test