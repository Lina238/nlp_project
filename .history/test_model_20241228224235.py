import os
import json
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
import torch
from peft import PeftModel
from tp import QAMetrics
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

def test_example(model, tokenizer, question, context, reference_answer):
    """
    Teste le modèle avec un exemple spécifique en utilisant les métriques QA
    Args:
        model: Le modèle chargé
        tokenizer: Le tokenizer
        question: La question à poser
        context: Le contexte médical
        reference_answer: La réponse de référence correcte
    """
    print("\n=== Test d'un exemple spécifique ===")
    print(f"\nQuestion: {question}")
    print(f"Contexte: {context[:200]}...")  # Affiche les 200 premiers caractères pour la lisibilité
    print(f"Réponse de référence: {reference_answer}")

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
        
        # Trouver les n meilleurs indices de début et de fin
        n_best = 5
        start_indexes = torch.topk(start_scores, n_best).indices[0]
        end_indexes = torch.topk(end_scores, n_best).indices[0]
        
        # Essayer différentes combinaisons de début/fin pour trouver la meilleure réponse
        best_score = float('-inf')
        best_answer = ""
        
        for start_idx in start_indexes:
            for end_idx in end_indexes:
                if end_idx < start_idx or end_idx - start_idx > 50:  # Éviter les réponses trop longues
                    continue
                    
                score = start_scores[0][start_idx] + end_scores[0][end_idx]
                if score > best_score:
                    best_score = score
                    answer_span = inputs["input_ids"][0][start_idx:end_idx+1]
                    best_answer = tokenizer.decode(answer_span)
        
        # Nettoyage de la réponse prédite
        predicted_answer = best_answer.replace("[CLS]", "").replace("[SEP]", "").strip()
        predicted_answer = ' '.join(predicted_answer.split())  # Normaliser les espaces

        # Calcul des différentes métriques
        exact_match = QAMetrics.calculate_exact_match(predicted_answer, reference_answer)
        f1_score = QAMetrics.calculate_f1(predicted_answer, reference_answer)
        cosine_sim = QAMetrics.calculate_cosine_similarity(predicted_answer, reference_answer)

    print("\nRésultats:")
    print(f"Réponse prédite: {predicted_answer}")
    print("\nMétriques:")
    print(f"Exact Match Score: {exact_match:.2f}")
    print(f"F1 Score: {f1_score:.2%}")
    print(f"Similarité Cosinus: {cosine_sim:.2%}")

    # Analyse détaillée
    print("\nAnalyse détaillée:")
    pred_tokens = set(predicted_answer.lower().split())
    ref_tokens = set(reference_answer.lower().split())
    
    common_tokens = pred_tokens & ref_tokens
    only_in_pred = pred_tokens - ref_tokens
    only_in_ref = ref_tokens - pred_tokens
    
    print(f"\nMots correctement identifiés ({len(common_tokens)}):")
    print(", ".join(common_tokens) if common_tokens else "Aucun")
    
    print(f"\nMots en trop dans la prédiction ({len(only_in_pred)}):")
    print(", ".join(only_in_pred) if only_in_pred else "Aucun")
    
    print(f"\nMots manquants de la référence ({len(only_in_ref)}):")
    print(", ".join(only_in_ref) if only_in_ref else "Aucun")
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
        reference_answer = "Les signes les plus courants incluent une douleur thoracique intense et prolongée, souvent décrite comme une sensation d'oppression ou de serrement."
        # Test de l'exemple
        # Test avec l'exemple de la crise cardiaque
        test_example(model, tokenizer, question, context, reference_answer)