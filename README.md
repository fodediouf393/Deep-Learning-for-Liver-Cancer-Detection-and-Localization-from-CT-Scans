# Liver Tumor Detection and Segmentation

=======


Fodé DIOUF


---

## 1. Objectif du projet

Développer un système automatisé pour :
- **Classifier** les images de foie (sain vs pathologique) à l’aide du modèle **VGG16**.
- **Segmenter** les tumeurs à l’aide du modèle **U-Net**, afin de localiser précisément les zones pathologiques.

---

## 2. Problématique

> Comment automatiser la classification et la segmentation des tumeurs hépatiques à partir d’images CT tout en assurant précision et fiabilité clinique ?

### Défis :
- Variabilité des images CT
- Complexité des formes tumorales
- Nécessité d'une grande précision clinique

---

## 3. Approche et méthodologie

### Classification
- **Architecture :** VGG16
- **Objectif :** Distinguer les images saines des images contenant des tumeurs

### Segmentation
- **Architecture :** U-Net
- **Objectif :** Localiser précisément les zones tumorales

### Impact attendu
- Réduction du temps d’analyse pour les radiologues
- Amélioration de la précision du diagnostic

---

## 4. Implémentation

### Fichiers principaux :
- `dataset.py` : Chargement, prétraitement et préparation des données
- `lit_model.py` : Encapsulation des modèles avec PyTorch Lightning
- `train.py` : Entraînement des modèles
- `vgg16.py` : Implémentation du modèle VGG16
- `unet.py` : Implémentation du modèle U-Net

### Prédictions :
- `predict_classification.py` : Chargement d’un modèle VGG16 pour classification
- `predict_segmentation.py` : Chargement d’un modèle U-Net pour segmentation

---

## 5. Résultats expérimentaux

### 📊 **Classification (VGG16)**  
Chemin des graphiques : `./Graphs_Classification/`

- **Train Loss par Epoch**  
  ![train_loss_epoch](./Graphs_Classification/IMG_3283.PNG)

- **Train Accuracy par Epoch**  
  ![train_accuracy_epoch](./Graphs_Classification/IMG_3284.PNG)

- **Train Loss par Step**  
  ![train_loss_step](./Graphs_Classification/IMG_3285.PNG)

- **Train Accuracy par Step**  
  ![train_accuracy_step](./Graphs_Classification/IMG_3286.PNG)

- **Validation Loss**  
  ![val_loss](./Graphs_Classification/IMG_3287.PNG)

- **Validation Accuracy**  
  ![val_accuracy](./Graphs_Classification/IMG_3288.PNG)

---

### 🟥 **Segmentation (U-Net)** 

- **Exemples de segmentation sur le test**  
  ![segmentation1](./Graphs_Segmentation/Image1.png)  
  ![segmentation2](./Graphs_Segmentation/Image2.png)  
  ![segmentation3](./Graphs_Segmentation/Image3.png)  
  ![segmentation4](./Graphs_Segmentation/Image4.png)
 
Chemin des graphiques : `./Graphs_Segmentation/`

- **Validation Dice Score**  
  ![val_dice_score](./Graphs_Segmentation/Image8.png)

- **Train Dice Score par Epoch**  
  ![train_dice_score_epoch](./Graphs_Segmentation/Image7.png)

- **Validation Loss**  
  ![val_loss](./Graphs_Segmentation/Image6.png)

- **Train Loss par Epoch**  
  ![train_loss_epoch](./Graphs_Segmentation/Image5.png)


---

## 6. Conclusion

### Ce que nous avons appris :
- Rôle crucial de l’IA en imagerie médicale
- Application concrète de VGG16 et U-Net dans un contexte réel
- Complexité des pipelines médicaux (prétraitement, annotation, évaluation)

### Difficultés rencontrées :
- Qualité et hétérogénéité des images CT
- Réglage fin des hyperparamètres pour éviter overfitting/sous-entraînement

---

## 7. Perspectives
- Amélioration du modèle de segmentation avec attention mechanism
- Entraînement sur des datasets plus riches et cliniquement annotés
- Intégration dans un outil complet d’aide au diagnostic médical



