from tp import load_saved_model, MedicalQADataset, train_model, save_model
from datasets import load_dataset
from torch.utils.data import DataLoader

def retrain_model(saved_model_path, num_epochs=5):
    """
    Charge et ré-entraîne un modèle sauvegardé avec plus d'époques
    Args:
        saved_model_path: Chemin vers le modèle sauvegardé
        num_epochs: Nombre d'époques pour le ré-entraînement
    """
    print("\n=== Starting Model Retraining ===")
    
    # Chargement du modèle sauvegardé
    print(f"\nLoading saved model from {saved_model_path}...")
    qa_model = load_saved_model(saved_model_path)
    
    # Chargement du dataset
    print("\nLoading dataset...")
    dataset = load_dataset("Malikeh1375/medical-question-answering-datasets", "all-processed")
    
    # Utilisation du dataset complet cette fois
    train_val_split = dataset["train"].train_test_split(test_size=0.1, seed=42)
    train_data = train_val_split["train"]
    val_data = train_val_split["test"]
    
    print(f"\nTaille du dataset d'entraînement: {len(train_data)}")
    print(f"Taille du dataset de validation: {len(val_data)}")
    
    # Création des datasets
    print("\nCreating datasets...")
    train_dataset = MedicalQADataset(
        train_data,
        qa_model.tokenizer
    )
    eval_dataset = MedicalQADataset(
        val_data,
        qa_model.tokenizer
    )
    
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
    
    # Ré-entraînement avec plus d'époques
    print(f"\nStarting retraining process with {num_epochs} epochs...")
    train_model(
        qa_model.model,
        train_dataloader,
        eval_dataloader,
        num_epochs=num_epochs
    )
    
    # Sauvegarde du modèle ré-entraîné
    print("\nSaving retrained model...")
    save_model(qa_model.model, qa_model.tokenizer, "retrained_model")
    
    print("\n=== Retraining completed ===")

if __name__ == "__main__":
    retrain_model("saved_model", num_epochs=5)