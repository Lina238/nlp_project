import os
import json
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
import torch

def inspect_saved_model():
    """Examine le contenu du dossier saved_model et affiche les informations importantes"""
    saved_model_path = "saved_model"
    
    print("\n=== Inspection du modèle sauvegardé ===")
    
    # Vérifier si le dossier existe
    if not os.path.exists(saved_model_path):
        print(f"Erreur: Le dossier {saved_model_path} n'existe pas!")
        return
        
    print("\n1. Structure du dossier:")
    for root, dirs, files in os.walk(saved_model_path):
        level = root.replace(saved_model_path, '').count(os.sep)
        indent = ' ' * 4 * level
        print(f"{indent}{os.path.basename(root)}/")
        subindent = ' ' * 4 * (level + 1)
        for f in files:
            print(f"{subindent}{f}")
            
    # Lire la configuration si elle existe
    config_path = os.path.join(saved_model_path, "config.json")
    if os.path.exists(config_path):
        print("\n2. Configuration du modèle:")
        with open(config_path, 'r') as f:
            config = json.load(f)
            for key, value in config.items():
                print(f"- {key}: {value}")
    
    # Vérifier la taille du modèle
    total_size = 0
    for root, dirs, files in os.walk(saved_model_path):
        for f in files:
            fp = os.path.join(root, f)
            total_size += os.path.getsize(fp)
    
    print(f"\n3. Taille totale du modèle: {total_size / (1024*1024):.2f} MB")
    
    # Essayer de charger le modèle pour vérifier sa structure
    try:
        print("\n4. Test de chargement du modèle:")
        model = AutoModelForQuestionAnswering.from_pretrained(saved_model_path)
        tokenizer = AutoTokenizer.from_pretrained(saved_model_path)
        
        print("- Modèle chargé avec succès")
        print(f"- Nombre de paramètres: {sum(p.numel() for p in model.parameters()):,}")
        print(f"- Type de tokenizer: {type(tokenizer).__name__}")
        
    except Exception as e:
        print(f"Erreur lors du chargement du modèle: {str(e)}")
        
    print("\nInspection terminée.")

if __name__ == "__main__":
    inspect_saved_model()