from tp import MedicalQADataset
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AutoModelForQuestionAnswering, AutoTokenizer, AdamW, get_linear_schedule_with_warmup
import torch
from tqdm import tqdm
import wandb

def train_model(model, train_dataloader, eval_dataloader, device, num_epochs=1):
    """
    Fonction d'entraînement personnalisée
    """
    optimizer = AdamW(model.parameters(), lr=2e-5)
    total_steps = len(train_dataloader) * num_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=total_steps
    )

    model.train()
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{num_epochs}")
        
        for batch in progress_bar:
            # Déplacer les tenseurs vers le device
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            start_positions = batch['start_positions'].to(device)
            end_positions = batch['end_positions'].to(device)
            
            # Réinitialiser les gradients
            optimizer.zero_grad()
            
            # Forward pass avec calcul de la perte
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                start_positions=start_positions,
                end_positions=end_positions
            )
            
            loss = outputs.loss
            # S'assurer que la perte est un tenseur qui requiert des gradients
            if not loss.requires_grad:
                loss.requires_grad = True
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            # Mise à jour des poids
            optimizer.step()
            scheduler.step()
            
            # Logging avec wandb
            wandb.log({
                "loss": loss.item(),
                "learning_rate": scheduler.get_last_lr()[0]
            })
            
            # Mise à jour de la barre de progression
            progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})

def retrain_model(saved_model_path, num_epochs=5, sample_size=1000):
    """
    Charge et ré-entraîne un modèle sauvegardé avec plus d'époques
    """
    print("\n=== Starting Model Retraining ===")
    
    # Configuration de wandb
    wandb.init(project="medical-qa-peft", name=f"retraining_{num_epochs}epochs")
    
    # Configuration du device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUtilisation du device: {device}")
    
    # Chargement du modèle
    try:
        print(f"\nTentative de chargement du modèle depuis {saved_model_path}...")
        model = AutoModelForQuestionAnswering.from_pretrained(saved_model_path)
        tokenizer = AutoTokenizer.from_pretrained(saved_model_path)
        print("Modèle chargé avec succès depuis le fichier sauvegardé")
    except:
        print("\nImpossible de charger le modèle sauvegardé. Chargement du modèle BERT pré-entraîné...")
        model = AutoModelForQuestionAnswering.from_pretrained("bert-base-multilingual-cased")
        tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")
        print("Modèle BERT pré-entraîné chargé avec succès")
    
    # Déplacement du modèle vers le device
    model = model.to(device)
    
    # Activation explicite des gradients pour tous les paramètres
    for param in model.parameters():
        param.requires_grad_(True)
    
    # Chargement du dataset
    print("\nLoading dataset...")
    dataset = load_dataset("Malikeh1375/medical-question-answering-datasets", "all-processed")
    
    # Réduction de la taille du dataset
    dataset_reduced = dataset["train"].shuffle(seed=42).select(range(sample_size))
    
    # Split train/val
    train_val_split = dataset_reduced.train_test_split(test_size=0.1, seed=42)
    train_data = train_val_split["train"]
    val_data = train_val_split["test"]
    
    print(f"\nTaille du dataset d'entraînement: {len(train_data)}")
    print(f"Taille du dataset de validation: {len(val_data)}")
    
    # Création des datasets
    train_dataset = MedicalQADataset(train_data, tokenizer)
    eval_dataset = MedicalQADataset(val_data, tokenizer)
    
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
    
    # Entraînement et sauvegarde à chaque époque
    print(f"\nStarting retraining process with {num_epochs} epochs...")
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        
        # Entraînement
        train_model(
            model,
            train_dataloader,
            eval_dataloader,
            device,
            num_epochs=1
        )
        
        # Sauvegarde du modèle
        save_path = f"retrain_epoch_{epoch + 1}"
        print(f"\nSaving model for epoch {epoch + 1} to {save_path}...")
        model.save_pretrained(save_path)
        tokenizer.save_pretrained(save_path)
    
    wandb.finish()
    print("\n=== Retraining completed ===")

if __name__ == "__main__":
    retrain_model("saved_model", num_epochs=3, sample_size=6000)