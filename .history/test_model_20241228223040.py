import os
import json
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
import torch
from peft import PeftModel

def inspect_saved_model():
    """Examine le contenu du dossier saved_model et affiche les informations importantes"""
    # Code existant maintenu tel quel
    saved_model_path = "saved_model"
    
    print("\n=== Inspection du modèle sauvegardé ===")
    
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
            
    config_path = os.path.join(saved_model_path, "config.json")
    if os.path.exists(config_path):
        print("\n2. Configuration du modèle:")
        with open(config_path, 'r') as f:
            config = json.load(f)
            for key, value in config.items():
                print(f"- {key}: {value}")
    
    total_size = 0
    for root, dirs, files in os.walk(saved_model_path):
        for f in files:
            fp = os.path.join(root, f)
            total_size += os.path.getsize(fp)
    
    print(f"\n3. Taille totale du modèle: {total_size / (1024*1024):.2f} MB")
    
    try:
        print("\n4. Test de chargement du modèle:")
        model = AutoModelForQuestionAnswering.from_pretrained(saved_model_path)
        tokenizer = AutoTokenizer.from_pretrained(saved_model_path)
        
        print("- Modèle chargé avec succès")
        print(f"- Nombre de paramètres: {sum(p.numel() for p in model.parameters()):,}")
        print(f"- Type de tokenizer: {type(tokenizer).__name__}")
        
        return model, tokenizer
        
    except Exception as e:
        print(f"Erreur lors du chargement du modèle: {str(e)}")
        return None, None

def test_example(model, tokenizer, question, context):
    """
    Teste le modèle avec un exemple spécifique
    Args:
        model: Le modèle chargé
        tokenizer: Le tokenizer
        question: La question à poser
        context: Le contexte médical
    """
    print("\n=== Test d'un exemple spécifique ===")
    print(f"\nQuestion: {question}")
    print(f"Contexte: {context}")

    # Préparation des inputs
    inputs = tokenizer(
        question,
        context,
        add_special_tokens=True,
        return_tensors="pt",
        max_length=512,
        truncation=True,
        padding="max_length"
    )

    # Passage en mode évaluation
    model.eval()

    # Prédiction
    with torch.no_grad():
        outputs = model(**inputs)
        
        # Récupération des positions de début et fin
        start_scores = outputs.start_logits
        end_scores = outputs.end_logits
        
        # Récupération des meilleurs indices
        start_index = torch.argmax(start_scores)
        end_index = torch.argmax(end_scores)
        
        # Décodage de la réponse
        answer = tokenizer.decode(inputs["input_ids"][0][start_index:end_index+1])
        
        # Calcul du score de confiance
        confidence = float(
            torch.softmax(start_scores, dim=1).max() * 
            torch.softmax(end_scores, dim=1).max()
        )

    print("\nRésultats:")
    print(f"Réponse: {answer}")
    print(f"Indice de confiance: {confidence:.2%}")

if __name__ == "__main__":
    # Inspection et chargement du modèle
    model, tokenizer = inspect_saved_model()
    
    if model is not None and tokenizer is not None:
        # Exemple de test
        question = "Quels sont les symptômes principaux d'une crise cardiaque?"
        context = """Une crise cardiaque, ou infarctus du myocarde, se manifeste généralement 
                    par plusieurs symptômes caractéristiques. Les signes les plus courants 
                    incluent une douleur thoracique intense et prolongée, souvent décrite 
                    comme une sensation d'oppression ou de serrement. Cette douleur peut 
                    irradier vers le bras gauche, la mâchoire, le cou ou le dos. D'autres 
                    symptômes comprennent des sueurs froides, un essoufflement, des nausées 
                    et des vomissements. Certains patients peuvent également ressentir une 
                    anxiété intense ou une sensation de mort imminente."""
        
        # Test de l'exemple
        test_example(model, tokenizer, question, context)