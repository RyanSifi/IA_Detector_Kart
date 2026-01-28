import torch
import mlflow
import mlflow.pytorch
from ultralytics import YOLO

# 1. Connexion au serveur
mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("Karting_YOLO_Experiment")

# Chemin vers ton fichier
model_path = "runs/detect/train3/weights/best.pt"

print(f"Chargement du checkpoint depuis {model_path}...")

try:
    # On charge TOUT le fichier (la valise)
    checkpoint = torch.load(model_path, map_location=torch.device('cpu'), weights_only=False)
    
    # CORRECTIF : On vérifie si c'est un dictionnaire et on sort le modèle
    if isinstance(checkpoint, dict) and 'model' in checkpoint:
        print("Dictionnaire détecté. Extraction du modèle interne...")
        model = checkpoint['model']
    else:
        model = checkpoint

    # On s'assure que le modèle est en mode évaluation (pas entraînement)
    if hasattr(model, 'eval'):
        model.eval()
        
    print(f"Modèle extrait avec succès. Type: {type(model)}")

except Exception as e:
    print(f"ECHEC : {e}")
    exit()

# 3. Envoi au serveur MLflow
with mlflow.start_run() as run:
    print("Envoi vers MLflow en cours...")
    
    # Maintenant 'model' est bien un torch.nn.Module
    mlflow.pytorch.log_model(model, artifact_path="model")
    
    mlflow.log_param("source_file", model_path)
    mlflow.log_param("type", "YOLOv8_Extracted")
    
    model_uri = f"runs:/{run.info.run_id}/model"
    
    registered_model = mlflow.register_model(model_uri, "YoloV8_Karting")
    
    print(f"SUCCÈS : YOLO enregistré. Version: {registered_model.version}")