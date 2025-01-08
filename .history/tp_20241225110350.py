# Import des bibliothèques nécessaires
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
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

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
        encoding = self.tokenizer(
            item['question'],
            item['context'],
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Conversion des positions en tenseurs
        start_positions = torch.tensor([item['answer_start']])
        end_positions = torch.tensor([item['answer_end']])

        # Retour du dictionnaire formaté
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'start_positions': start_positions.flatten(),
            'end_positions': end_positions.flatten()
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
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForQuestionAnswering.from_pretrained(model_name)
        self.peft_technique = peft_technique
        self.setup_peft()

    def setup_peft(self):
        """Configure la technique PEFT choisie"""
        if self.peft_technique == "adapter":
            # Configuration LoRA pour adapter
            peft_config = LoraConfig(
                task_type=TaskType.QUESTION_ANSWERING,
                inference_mode=False,
                r=16,  # Rang de la matrice
                lora_alpha=32,  # Facteur d'échelle
                lora_dropout=0.1,  # Taux de dropout
                target_modules=['query', 'value']  # Modules ciblés
            )
        elif self.peft_technique == "prompt":
            # Configuration du prompt tuning
            peft_config = PromptTuningConfig(
                task_type=TaskType.QUESTION_ANSWERING,
                num_virtual_tokens=20,  # Nombre de tokens virtuels
                prompt_tuning_init="TEXT",  # Initialisation
                tokenizer_name_or_path=self.model_name
            )
        
        # Application de la configuration PEFT
        self.model = get_peft_model(self.model, peft_config)

    def apply_pruning(self, amount=0.3):
        """
        Applique le pruning au modèle
        Args:
            amount: Pourcentage de connexions à supprimer
        """
        for name, module in self.model.named_modules():
            if isinstance(module, torch.nn.Linear):
                # Pruning L1 non structuré
                prune.l1_unstructured(module, name='weight', amount=amount)
                # Rendre le pruning permanent
                prune.remove(module, 'weight')

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
    return {
        k: sum(v)/len(v) for k, v in metrics.items()
    }

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
    
    return results

def main():
    """Fonction principale d'exécution"""
    
    # Chargement du dataset
    dataset = load_dataset("Malikeh1375/medical-question-answering-datasets")
    dataset = dataset.select(range(1000))  # Limitation à 1000 exemples
    
    # Initialisation du modèle
    qa_model = MedicalQAModel(
        model_name="bert-base-uncased",
        peft_technique="adapter"
    )
    
    # Application du pruning
    qa_model.apply_pruning(amount=0.3)
    
    # Création des datasets
    train_dataset = MedicalQADataset(
        dataset['train'],
        qa_model.tokenizer
    )
    eval_dataset = MedicalQADataset(
        dataset['validation'],
        qa_model.tokenizer
    )
    
    # Création des dataloaders
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
    train_model(
        qa_model.model,
        train_dataloader,
        eval_dataloader,
        num_epochs=3
    )
    
    # Questions de test
    test_questions = [
        (
            "Quels sont les symptômes de la COVID-19?",
            "La COVID-19 peut provoquer divers symptômes. Les