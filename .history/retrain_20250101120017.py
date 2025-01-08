from tp import load_saved_model, MedicalQADataset, train_model, save_model
from datasets import load_dataset
from torch.utils.data import DataLoader

def retrain_model(saved_model_path, num_epochs=5, sample_size=1000):
    """
    Charge et ré-entraîne un modèle sauvegardé avec plus d'époques
    Args:
        saved_model_path: Chemin vers le modèle sauvegardé
        num_epochs: Nombre d'époques pour le ré-entraînement
        sample_size: Nombre d'exemples à utiliser pour l'entraînement
    """
    print("\n=== Starting Model Retraining ===")
    
    # Chargement du modèle sauvegardé
    print(f"\nLoading saved model from {saved_model_path}...")
    qa_model = load_saved_model(saved_model_path)
    
    # Chargement du dataset
    print("\nLoading dataset...")
    dataset = load_dataset("Malikeh1375/medical-question-answering-datasets", "all-processed")
    
    # Réduction de la taille du dataset
    dataset_reduced = dataset["train"].shuffle(seed=42).select(range(sample_size))
    
    # Split train/val sur le dataset réduit
    train_val_split = dataset_reduced.train_test_split(test_size=0.1, seed=42)
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
    
    # Ré-entraînement avec sauvegarde à chaque époque
    print(f"\nStarting retraining process with {num_epochs} epochs...")
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        
        # Entraînement pour une époque
        train_model(
            qa_model.model,
            train_dataloader,
            eval_dataloader,
            num_epochs=1
        )
        
        # Sauvegarde du modèle après chaque époque
        save_path = f"retrain_epoch_{epoch + 1}"
        print(f"\nSaving model for epoch {epoch + 1} to {save_path}...")
        save_model(qa_model.model, qa_model.tokenizer, save_path)
    
    print("\n=== Retraining completed ===")

if __name__ == "__main__":
    # Utilisation d'un échantillon plus petit (1000 exemples) et 3 époques
    retrain_model("saved_model", num_epochs=3, sample_size=1000)