# Détecteur de Kart avec YOLOv8

Ce projet utilise le modèle YOLOv8 pour détecter des karts dans des images. Il est structuré pour entraîner le modèle sur un jeu de données personnalisé et, à terme, pour utiliser le modèle entraîné dans une application Streamlit.

## Fichiers Volumineux Manquants

Pour maintenir le dépôt léger et respecter les limites de stockage de GitHub, deux fichiers volumineux ont été volontairement exclus du dépôt. Voici comment les obtenir ou les remplacer.

### 1. Modèle de base : `yolov8n.pt`

Ce fichier est le modèle de base YOLOv8 Nano, pré-entraîné sur le jeu de données COCO. Le script d'entraînement (`train.py`) l'utilise comme point de départ.

**Action requise :** Aucune action manuelle n'est nécessaire. La bibliothèque `ultralytics` le téléchargera automatiquement la première fois que vous lancerez un entraînement.

### 2. Vidéo de test : `video_prompt_test_model.mp4`

Une vidéo de test était initialement présente, mais sa taille dépassait les limites de stockage de GitHub (plus de 2 Go).

**Action requise :** Vous devrez fournir votre propre fichier vidéo au format `.mp4` pour tester les fonctionnalités de détection sur vidéo. Lorsque vous implémenterez un script de prédiction ou l'application Streamlit, il vous suffira de spécifier le chemin vers votre propre fichier vidéo.
