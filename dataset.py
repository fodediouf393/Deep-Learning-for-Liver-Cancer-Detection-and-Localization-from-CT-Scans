import torch
import csv
import h5py
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import transforms
from PIL import Image
import os
import numpy as np
import cv2
from tqdm import tqdm
import torch.nn.functional as F
# import albumentations as A
# from albumentations.pytorch.transforms import ToTensorV2
import time
import random

class HDF5Dataset(Dataset):
    def __init__(self, data_path, patient_ids, transform=None, mode='train', classification_type=True):
        self.data_path = data_path
        self.patient_ids = [bytes.decode(patient) + '_dic_msk' for patient in patient_ids]
        self.transform = transform
        self.mode = mode
        self.classification_type = classification_type
        self.patient_slices = self._prepare_patient_slices()
        self.hf = h5py.File(self.data_path, "r")

    def __getitem__(self, index):
        patient_id, slice_idx = self.patient_slices[index]
        image = self.hf[patient_id][0][slice_idx]
        mask = self.hf[patient_id][1][slice_idx]
        image = cv2.resize(image, (128, 128))
        mask = cv2.resize(mask, (128, 128), interpolation=cv2.INTER_NEAREST)
        image = self._preprocess_image(image)
        mask = self._preprocess_mask(mask)
        return image, mask

    def __len__(self):
        return len(self.patient_slices)

    def _prepare_patient_slices(self):
        slices = []
        with h5py.File(self.data_path, "r") as hf:
            for patient_id in self.patient_ids:
                try:
                    n_slices = len(hf[patient_id][0])
                    slices.extend([(patient_id, i) for i in range(n_slices)])
                except KeyError:
                    print(f"Could not read patient {patient_id}")
        return slices

    def _preprocess_mask(self, mask):
        if self.classification_type:
            if 255 in mask:
                return torch.tensor([2])
            elif 127 in mask:
                return torch.tensor([1])
            else:
                return torch.tensor([0])
        else:
            mask = np.where(mask == 127, 0, mask)
            mask = np.where(mask == 191, 1, mask)
            mask = np.where(mask == 255, 2, mask)
            mask = torch.as_tensor(mask, dtype=torch.long)
            return mask.unsqueeze(0)

    def _apply_window(self, image, window_level, window_width):
        min_value = window_level - window_width // 2
        max_value = window_level + window_width // 2
        windowed_image = np.clip(image, min_value, max_value)
        return windowed_image

    def _preprocess_image(self, image):
        image = self._apply_window(image, 100, 200)
        image = torch.as_tensor(image / 255.0, dtype=torch.float32)
        return image.unsqueeze(0)

def SelectPatientsTrainVal(input_path, val_split):
    hf = h5py.File(input_path, "r")
    PatientsId = hf['patient_id'][0]
    print("patientId shape ", PatientsId.shape)
    np.random.seed(42)
    np.random.shuffle(PatientsId)
    NPatients = PatientsId.shape[0]
    PatientsIdTrain = PatientsId[:int((1 - val_split) * NPatients + 0.5)]
    PatientsIdVal = PatientsId[int((1 - val_split) * NPatients + 0.5):]
    hf.close()
    return np.array(PatientsIdTrain), np.array(PatientsIdVal)

def main():
    path = '/home/IMA_project/Liver/database_27.hdf5'
    PatientsIdTrain, PatientsIdVal = SelectPatientsTrainVal(path, 0.2)
    train_ds = HDF5Dataset(path, PatientsIdTrain[:1], transform=None, classification_type=False)
    val_ds = HDF5Dataset(path, PatientsIdVal[:1], transform=None, mode='valid', classification_type=False)
    n_train = len(train_ds)
    n_val = len(val_ds)
    loader_params = dict(batch_size=16, num_workers=4, pin_memory=True, shuffle=False)
    train_dl = DataLoader(train_ds, **loader_params)
    val_dl = DataLoader(val_ds, **loader_params)
    print(len(train_dl))
    for i, (image, mask) in enumerate(train_dl):
        print(torch.unique(mask))

if __name__ == '__main__':
    main()
