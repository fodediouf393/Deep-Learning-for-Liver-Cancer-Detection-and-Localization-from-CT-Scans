import torch
import cv2
import numpy as np
import h5py
import os
from lit_model import Model

class LiverClassification:
    def __init__(self, model_path, data_path, device="cuda" if torch.cuda.is_available() else "cpu"):
        print("Initialisation de LiverClassification...")
        self.device = device
        self.model = self._load_model(model_path)
        self.model.to(self.device)
        self.model.eval()  # Mode évaluation
        self.data_path = data_path
        print(f"Modèle chargé avec succès sur {self.device}")

    def _load_model(self, model_path):
        print(f"Chargement du modèle depuis {model_path}...")
        model = Model.load_from_checkpoint(model_path, task="classification")
        print("Modèle chargé avec succès !")
        return model

    def preprocess_image(self, image):
        print("Prétraitement de l'image...")
        image = cv2.resize(image, (128, 128))  # Redimensionner à la taille attendue par le modèle
        image = self._apply_window(image, 100, 200)  # Appliquer une fenêtre de niveau
        image = image.astype(np.float32) / 255.0  # Normaliser entre 0 et 1
        image = torch.tensor(image).unsqueeze(0).unsqueeze(0)  # [B, C, H, W]
        print("Image prétraitée avec succès !")
        return image.to(self.device)

    def _apply_window(self, image, window_level, window_width):
        min_value = window_level - window_width // 2
        max_value = window_level + window_width // 2
        return np.clip(image, min_value, max_value) / window_width

    def get_available_patients(self):
        print("Chargement de la liste des patients...")
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Le fichier {self.data_path} n'existe pas.")
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
            labels = hf[patient_id][1][:]  # Labels
        print(f"Données du patient {patient_id} chargées avec succès ! ({len(data)} images)")
        return data, labels

    def predict(self, image):
        print("Prédiction de la classification...")
        with torch.no_grad():
            preprocessed_image = self.preprocess_image(image)
            output = self.model.net(preprocessed_image)
            probabilities = torch.softmax(output, dim=1)  # Convertir les logits en probabilités
            predicted_class = torch.argmax(probabilities, dim=1).item()  # Obtenir la classe prédite
        print("Prédiction terminée !")
        return predicted_class, probabilities.squeeze().cpu().numpy()

    def classify_patient(self, patient_id):
        print(f"Classification des images du patient {patient_id}...")
        images, _ = self.load_data(patient_id)
        predictions = [self.predict(image) for image in images]
        print(f"Classification complète pour {len(predictions)} images !")
        return predictions

# Exemple d'utilisation
if __name__ == "__main__":
    model_path = "/home/hepatic_tumor_detection/Liver_project/checkpoint_folder/my_model-epoch=010-val_loss=0.324.ckpt"
    data_path = "/home/hepatic_tumor_detection/Liver_project/liver_light.hdf5"  # Utilisez un chemin absolu

    print("Lancement de la classification du foie...")
    classifier = LiverClassification(model_path, data_path)

    patients = classifier.get_available_patients()
    if not patients:
        print("Aucun patient trouvé dans le dataset. Vérifiez le fichier HDF5.")
        exit()

    patient_id = patients[0]  # Utiliser le premier patient disponible
    print(f"Classification pour le patient : {patient_id}")

    predictions = classifier.classify_patient(patient_id)

    for i, (predicted_class, probabilities) in enumerate(predictions):
        print(f"Image {i}: Classe prédite = {predicted_class}, Probabilités = {probabilities}")

    print("Classification terminée avec succès !")