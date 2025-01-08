# Import des bibliothèques nécessaires
import json
import os
import torch  # Bibliothèque principale de deep learning
from torch.utils.data import Dataset, DataLoader  # Classes pour la gestion des données
from transformers import (
    AutoModelForQuestionAnswering,  # Modèle de base pour le question-answering
    AutoTokenizer,  # Pour tokeniser le texte
    TrainingArguments,  # Arguments d'entraînement
    Trainer  # Classe d'entraînement
)
from peft import (
    get_peft_config,  # Configuration pour le fine-tuning efficace
    PeftModel,  # Modèle avec PEFT
    TaskType,  # Types de tâches supportées
    PromptTuningConfig,  # Configuration pour le prompt tuning
    get_peft_model,  # Fonction pour appliquer PEFT
    LoraConfig  # Configuration pour LoRA
)
from torch.nn.utils import prune  # Outils de pruning
from datasets import load_dataset  # Chargement de datasets
from sklearn.metrics import f1_score  # Métrique d'évaluation
import numpy as np  # Calculs numériques
from tqdm import tqdm  # Barre de progression
import wandb  # Suivi des expériences

"""
Configuration des paramètres principaux:
- model_name = "bert-base-uncased" : Modèle pré-entraîné équilibré
- batch_size = 8 : Taille adaptée à la mémoire GPU
- learning_rate = 2e-5 : Taux d'apprentissage recommandé
- num_epochs = 3 : Nombre d'époques optimal
- max_length = 512 : Longueur maximale des séquences
- lora_r = 16 : Rang de compression LoRA
- lora_alpha = 32 : Facteur d'échelle LoRA
- lora_dropout = 0.1 : Dropout pour régularisation
- pruning_amount = 0.3 : Taux de pruning
"""

class MedicalQADataset(Dataset):
    """Gestion du dataset de questions-réponses médicales"""
    
    def __init__(self, data, tokenizer, max_length=512):
        """
        Initialisation du dataset
        Args:
            data: Données brutes
            tokenizer: Tokenizer pour le texte
            max_length: Longueur max des séquences
        """
        print(f"\nInitializing MedicalQADataset with max_length={max_length}")
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        print(f"Dataset initialized with {len(self.data)} examples")

    def __len__(self):
        """Retourne la taille du dataset"""
        return len(self.data)

    def __getitem__(self, idx):
        """
        Récupère et prépare un exemple
        Args:
            idx: Index de l'exemple
        Returns:
            dict: Exemple tokenisé avec positions
        """
        # Récupération d'un exemple
        item = self.data[idx]
        
        # Tokenisation du texte
        full_input = f"{item['instruction']} {item['input']}"
        answer = item['output']
        
        # Tokenisation du texte
        encoding = self.tokenizer(
            full_input,
            text_pair=answer,  # La réponse comme deuxième texte
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        
        # On trouve les positions de début et fin de la réponse dans le texte tokenisé
        answer_tokens = self.tokenizer(answer)
        answer_start = len(self.tokenizer(full_input)['input_ids']) - 1  # Position après le texte d'entrée
        answer_end = answer_start + len(answer_tokens['input_ids']) - 1
        
        # Retour du dictionnaire formaté
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'start_positions': torch.tensor([answer_start]),
            'end_positions': torch.tensor([answer_end])
        }

class MedicalQAModel:
    """Classe principale du modèle QA médical"""
    
    def __init__(self, model_name="bert-base-uncased", peft_technique="adapter"):
        """
        Initialisation du modèle
        Args:
            model_name: Nom du modèle pré-entraîné
            peft_technique: Technique de fine-tuning (adapter/prompt)
        """
        print(f"\nInitializing MedicalQAModel with:")
        print(f"- Model name: {model_name}")
        print(f"- PEFT technique: {peft_technique}")
        
        self.model_name = model_name
        print("Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        print("Loading base model...")
        self.model = AutoModelForQuestionAnswering.from_pretrained(model_name)
        self.peft_technique = peft_technique
        self.setup_peft()

    def setup_peft(self):
        """Configure la technique PEFT choisie"""
        print(f"\nSetting up PEFT with technique: {self.peft_technique}")
        if self.peft_technique == "adapter":
            # Configuration LoRA pour adapter
            print("Configuring LoRA adapter...")
            peft_config = LoraConfig(
                task_type=TaskType.QUESTION_ANS, 
                inference_mode=False,
                r=16,  # Rang de la matrice
                lora_alpha=32,  # Facteur d'échelle
                lora_dropout=0.1,  # Taux de dropout
                target_modules=['query', 'value']  # Modules ciblés
            )
        elif self.peft_technique == "prompt":
            # Configuration du prompt tuning
            print("Configuring prompt tuning...")
            peft_config = PromptTuningConfig(
                task_type=TaskType.QUESTION_ANS, 
                num_virtual_tokens=20,  # Nombre de tokens virtuels
                prompt_tuning_init="TEXT",  # Initialisation
                tokenizer_name_or_path=self.model_name
            )
        
        # Application de la configuration PEFT
        print("Applying PEFT configuration to model...")
        self.model = get_peft_model(self.model, peft_config)
        print("PEFT setup completed")

    def apply_pruning(self, amount=0.3):
        """
        Applique le pruning au modèle
        Args:
            amount: Pourcentage de connexions à supprimer
        """
        print(f"\nApplying pruning with amount={amount}")
        for name, module in self.model.named_modules():
            if isinstance(module, torch.nn.Linear):
                # Pruning L1 non structuré
                prune.l1_unstructured(module, name='weight', amount=amount)
                # Rendre le pruning permanent
                prune.remove(module, 'weight')
        print("Pruning completed")

class QAMetrics:
    """Classe pour le calcul des métriques d'évaluation"""
    
    @staticmethod
    def calculate_exact_match(pred_text, true_text):
        """
        Calcule le score de correspondance exacte
        Args:
            pred_text: Texte prédit
            true_text: Texte vrai
        Returns:
            int: 1 si exact match, 0 sinon
        """
        return int(pred_text.strip() == true_text.strip())

    @staticmethod
    def calculate_f1(pred_text, true_text):
        """
        Calcule le score F1
        Args:
            pred_text: Texte prédit
            true_text: Texte vrai
        Returns:
            float: Score F1
        """
        # Tokenisation
        pred_tokens = pred_text.lower().split()
        true_tokens = true_text.lower().split()
        
        # Gestion des cas vides
        if len(pred_tokens) == 0 or len(true_tokens) == 0:
            return int(pred_tokens == true_tokens)
        
        # Calcul des tokens communs
        common_tokens = set(pred_tokens) & set(true_tokens)
        if len(common_tokens) == 0:
            return 0
        
        # Calcul précision et rappel
        precision = len(common_tokens) / len(pred_tokens)
        recall = len(common_tokens) / len(true_tokens)
        
        # Retour du score F1
        return 2 * precision * recall / (precision + recall)

    @staticmethod
    def calculate_cosine_similarity(pred_text, true_text):
        """
        Calcule la similarité cosinus
        Args:
            pred_text: Texte prédit
            true_text: Texte vrai
        Returns:
            float: Score de similarité
        """
        from sklearn.feature_extraction.text import TfidfVectorizer
        vectorizer = TfidfVectorizer()
        vectors = vectorizer.fit_transform([pred_text, true_text])
        return (vectors * vectors.T).A[0, 1]

def train_model(model, train_dataloader, eval_dataloader, num_epochs=3):
    """
    Entraîne le modèle
    Args:
        model: Modèle à entraîner
        train_dataloader: Loader données d'entraînement
        eval_dataloader: Loader données d'évaluation
        num_epochs: Nombre d'époques
    """
    print("\nInitializing training...")
    # Initialisation du suivi
    wandb.init(project="medical-qa-peft")
    
    # Configuration optimiseur et scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    scheduler = torch.optim.lr_scheduler.LinearLR(optimizer)
    
    # Boucle d'entraînement
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        
        # Boucle sur les batches
        for batch in tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask'],
                start_positions=batch['start_positions'],
                end_positions=batch['end_positions']
            )
            
            # Calcul et accumulation de la perte
            loss = outputs.loss
            total_loss += loss.item()
            
            # Backward pass et optimisation
            loss.backward()
            optimizer.step()
            scheduler.step()
        
        # Évaluation
        metrics = evaluate_model(model, eval_dataloader)
        
        # Logging des métriques
        wandb.log({
            "epoch": epoch,
            "loss": total_loss / len(train_dataloader),
            "exact_match": metrics['exact_match'],
            "f1": metrics['f1'],
            "cosine_similarity": metrics['cosine_similarity']
        })
        
        # Affichage des résultats
        print(f"Epoch {epoch+1} - Loss: {total_loss/len(train_dataloader):.4f}")
        print(f"Metrics: {metrics}")

def evaluate_model(model, eval_dataloader):
    """
    Évalue le modèle
    Args:
        model: Modèle à évaluer
        eval_dataloader: Loader de données d'évaluation
    Returns:
        dict: Métriques d'évaluation
    """
    print("\nStarting model evaluation...")
    model.eval()
    metrics = {
        'exact_match': [],
        'f1': [],
        'cosine_similarity': []
    }
    
    with torch.no_grad():
        for batch in eval_dataloader:
            # Prédictions
            outputs = model(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask']
            )
            
            # Extraction des positions prédites
            pred_start = torch.argmax(outputs.start_logits, dim=1)
            pred_end = torch.argmax(outputs.end_logits, dim=1)
            
            # Calcul des métriques
            for i in range(len(pred_start)):
                pred_text = tokenizer.decode(
                    batch['input_ids'][i][pred_start[i]:pred_end[i]+1]
                )
                true_text = tokenizer.decode(
                    batch['input_ids'][i][
                        batch['start_positions'][i]:batch['end_positions'][i]+1
                    ]
                )
                
                # Accumulation des métriques
                metrics['exact_match'].append(
                    QAMetrics.calculate_exact_match(pred_text, true_text)
                )
                metrics['f1'].append(
                    QAMetrics.calculate_f1(pred_text, true_text)
                )
                metrics['cosine_similarity'].append(
                    QAMetrics.calculate_cosine_similarity(pred_text, true_text)
                )
    
    # Calcul des moyennes
    final_metrics = {
        k: sum(v)/len(v) for k, v in metrics.items()
    }
    print("Evaluation completed")
    return final_metrics

def test_model(model, tokenizer, test_questions):
    """
    Teste le modèle sur des questions spécifiques
    Args:
        model: Modèle à tester
        tokenizer: Tokenizer
        test_questions: Liste de tuples (question, contexte)
    Returns:
        list: Résultats des prédictions
    """
    print("\nStarting model testing...")
    model.eval()
    results = []
    
    for question, context in test_questions:
        # Préparation des inputs
        inputs = tokenizer(
            question,
            context,
            max_length=512,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Prédiction
        with torch.no_grad():
            outputs = model(**inputs)
            
            # Extraction des logits
            start_logits = outputs.start_logits
            end_logits = outputs.end_logits
            
            # Recherche des meilleures positions
            start_idx = torch.argmax(start_logits)
            end_idx = torch.argmax(end_logits)
            
            # Décodage de la réponse
            answer = tokenizer.decode(
                inputs['input_ids'][0][start_idx:end_idx+1]
            )
            
            # Calcul du score de confiance
            confidence = float(
                torch.softmax(start_logits, dim=1).max() * \
                torch.softmax(end_logits, dim=1).max()
            )
            
            # Stockage des résultats
            results.append({
                'question': question,
                'context': context,
                'predicted_answer': answer,
                'confidence': confidence
            })
    
    print("Testing completed")
    return results

def main():
    """Fonction principale d'exécution"""
    print("\n=== Starting Medical QA System ===")
    
    # Chargement du dataset
    print("\nLoading dataset...")
    dataset = load_dataset("Malikeh1375/medical-question-answering-datasets", "all-processed")
    
    # Affichage d'un exemple pour vérification
    print("\nExample data structure:")
    sample = dataset["train"][0]
    print(f"\nInstruction: {sample['instruction'][:100]}...")
    print(f"Input (question): {sample['input'][:100]}...")
    print(f"Output (answer): {sample['output'][:100]}...")
    
    # Initialisation du modèle
    print("\nInitializing model...")
    qa_model = MedicalQAModel(
        model_name="bert-base-uncased",
        peft_technique="adapter"
    )
    
    # Application du pruning
    print("\nApplying pruning...")
    qa_model.apply_pruning(amount=0.3)
    
    # Création des datasets
    print("\nCreating datasets...")
    train_dataset = MedicalQADataset(
        dataset['train'],
        qa_model.tokenizer
    )
    eval_dataset = MedicalQADataset(
        dataset['validation'],
        qa_model.tokenizer
    )
    
    print(f"Training dataset size: {len(train_dataset)}")
    print(f"Evaluation dataset size: {len(eval_dataset)}")
    
    # Création des dataloaders
    print("\nCreating dataloaders...")
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=8,
        shuffle=True
    )
    eval_dataloader = DataLoader(
        eval_dataset,
        batch_size=8
    )
    
    # Entraînement
    print("\nStarting training process...")
    train_model(
        qa_model.model,
        train_dataloader,
        eval_dataloader,
        num_epochs=3
    )

    # Sauvegarde du modèle après l'entraînement
    print("\nSaving model...")
    save_model(qa_model.model, "saved_model")

    # Questions de test
    print("\nPreparing test questions...")
    test_questions = [
        (
            "Quels sont les symptômes de la COVID-19?",
            "La COVID-19 peut provoquer divers symptômes. Les plus courants sont la fièvre, la toux sèche et la fatigue. D'autres symptômes moins fréquents comprennent la perte du goût ou de l'odorat, la congestion nasale, la conjonctivite, les maux de gorge, les maux de tête, les douleurs musculaires ou articulaires, différents types d'éruptions cutanées, des nausées ou vomissements, la diarrhée, des frissons ou des vertiges."
        ),
        (
            "Comment traiter l'hypertension?",
            "Le traitement de l'hypertension combine souvent des changements de mode de vie et des médicaments. Les modifications du mode de vie incluent une alimentation saine, la réduction du sel, l'exercice régulier, la limitation de l'alcool et l'arrêt du tabac. Les médicaments couramment prescrits comprennent les diurétiques, les inhibiteurs de l'ECA, les bêtabloquants et les antagonistes calciques."
        ),
        (
            "Quelles sont les causes du diabète de type 2?",
            "Le diabète de type 2 est principalement causé par une combinaison de facteurs génétiques et environnementaux. Les principaux facteurs de risque incluent l'obésité, le manque d'activité physique, une mauvaise alimentation, l'âge avancé, les antécédents familiaux, et certains groupes ethniques. La maladie se développe lorsque le corps devient résistant à l'insuline ou ne produit pas assez d'insuline pour maintenir des niveaux normaux de glucose."
        )
    ]
    
    # Exécution des tests et affichage des résultats
    print("\nRunning tests...")
    results = test_model(qa_model.model, qa_model.tokenizer, test_questions)
    
    # Affichage des résultats détaillés
    print("\nRésultats des tests sur les questions médicales:")
    print("=" * 80)
    for result in results:
        print(f"\nQuestion: {result['question']}")
        print(f"Réponse prédite: {result['predicted_answer']}")
        print(f"Niveau de confiance: {result['confidence']:.2%}")
        print("-" * 40)

if __name__ == "__main__":
    try:
        print("\n=== Starting Medical QA Application ===")
        main()
    except Exception as e:
        print(f"\nError occurred: {str(e)}")
        print(f"Error type: {type(e).__name__}")
        import traceback
        print("\nFull traceback:")
        traceback.print_exc()
        # Log de l'erreur si wandb est initialisé
        if wandb.run is not None:
            wandb.log({"error": str(e)})
        raise
    finally:
        # Fermeture propre de wandb
        if wandb.run is not None:
            wandb.finish()

def save_model(model, output_dir):
    """
    Sauvegarde le modèle et sa configuration
    Args:
        model: Modèle à sauvegarder
        output_dir: Répertoire de destination
    """
    print(f"\nSaving model to {output_dir}...")
    os.makedirs(output_dir, exist_ok=True)
    
    # Sauvegarde du modèle
    model.save_pretrained(output_dir)
    print("Model weights saved")
    
    # Sauvegarde du tokenizer
    model.tokenizer.save_pretrained(output_dir)
    print("Tokenizer saved")
    
    # Sauvegarde de la configuration
    config = {
        'model_name': model.config.model_type,
        'peft_technique': model.peft_config.peft_type if hasattr(model, 'peft_config') else None,
        'pruning_amount': 0.3,  # Valeur par défaut utilisée
        'max_length': 512
    }
    
    with open(os.path.join(output_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=2)
    print("Configuration saved")

def load_saved_model(model_dir):
    """
    Charge un modèle sauvegardé
    Args:
        model_dir: Répertoire contenant le modèle
    Returns:
        MedicalQAModel: Modèle chargé
    """
    print(f"\nLoading model from {model_dir}...")
    
    # Chargement de la configuration
    with open(os.path.join(model_dir, 'config.json'), 'r') as f:
        config = json.load(f)
    print("Configuration loaded")
    
    # Initialisation du modèle
    model = MedicalQAModel(
        model_name=config['model_name'],
        peft_technique=config['peft_technique']
    )
    
    # Chargement des poids
    print("Loading model weights...")
    model.model = PeftModel.from_pretrained(
        model.model,
        model_dir
    )
    print("Model loaded successfully")
    
    return model