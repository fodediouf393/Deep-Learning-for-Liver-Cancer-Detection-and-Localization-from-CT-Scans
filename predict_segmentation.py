import torch
import cv2
import numpy as np
import h5py
from lit_model import Model

class LiverSegmentation:
    def __init__(self, model_path, data_path, device="cuda" if torch.cuda.is_available() else "cpu"):
        print("Initialisation de LiverSegmentation...")
        self.device = device
        self.model = self._load_model(model_path)
        self.model.to(self.device)
        self.model.eval()  # Mode évaluation
        self.data_path = data_path
        print(f"Modèle chargé avec succès sur {self.device}")

    def _load_model(self, model_path):
        print(f"Chargement du modèle depuis {model_path}...")
        model = Model.load_from_checkpoint(model_path)
        print("Modèle chargé avec succès !")
        return model

    def preprocess_image(self, image):
        print("Prétraitement de l'image...")
        image = cv2.resize(image, (128, 128))
        image = self._apply_window(image, 100, 200)
        image = image.astype(np.float32) / 255.0
        image = torch.tensor(image).unsqueeze(0).unsqueeze(0)  # [B, C, H, W]
        print("Image prétraitée avec succès !")
        return image.to(self.device)

    def _apply_window(self, image, window_level, window_width):
        min_value = window_level - window_width // 2
        max_value = window_level + window_width // 2
        return np.clip(image, min_value, max_value) / window_width

    def get_available_patients(self):
        print("Chargement de la liste des patients...")
        with h5py.File(self.data_path, "r") as hf:
            patients = list(hf.keys())
        print(f"Patients disponibles : {patients}")
        return patients

    def load_data(self, patient_id):
        print(f"Chargement des données du patient {patient_id}...")
        with h5py.File(self.data_path, "r") as hf:
            if patient_id not in hf:
                raise ValueError(f"Patient ID {patient_id} not found in the dataset.")
            data = hf[patient_id][0][:]  # Images
            masks = hf[patient_id][1][:]  # Masques
        print(f"Données du patient {patient_id} chargées avec succès ! ({len(data)} images)")
        return data, masks

    def predict(self, image):
        print("Prédiction de la segmentation...")
        with torch.no_grad():
            preprocessed_image = self.preprocess_image(image)
            output = self.model.net(preprocessed_image)
            output = torch.argmax(output, dim=1).squeeze(0)
            output = output.cpu().numpy()
            output = cv2.resize(output, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
        print("Prédiction terminée !")
        return output

    def segment_patient(self, patient_id):
        print(f"Segmentation des images du patient {patient_id}...")
        images, _ = self.load_data(patient_id)
        masks = [self.predict(image) for image in images]
        print(f"Segmentation complète pour {len(masks)} images !")
        return masks

# Exemple d'utilisation
if __name__ == "__main__":
    model_path = "/home/hepatic_tumor_detection/Liver_project_segmentation/checkpoint_folder/my_model-epoch=012-val_loss=0.199.ckpt.ckpt"
    data_path = "./liver_light.hdf5"

    print("Lancement de la segmentation du foie...")
    segmenter = LiverSegmentation(model_path, data_path)

    patients = segmenter.get_available_patients()
    if not patients:
        print("Aucun patient trouvé dans le dataset. Vérifiez le fichier HDF5.")
        exit()

    patient_id = patients[15]
    print(f"Segmentation pour le patient : {patient_id}")

    segmented_masks = segmenter.segment_patient(patient_id)

    for i, mask in enumerate(segmented_masks):
        file_name = f"segmented_mask_{i}.png"
        cv2.imwrite(file_name, mask * 127)
        print(f"Masque sauvegardé : {file_name}")

    print("Segmentation terminée avec succès !")