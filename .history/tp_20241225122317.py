import json
import os
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForQuestionAnswering, AutoTokenizer
from peft import get_peft_config, PeftModel, TaskType, PromptTuningConfig, get_peft_model, LoraConfig
from torch.nn.utils import prune
from datasets import load_dataset
from sklearn.metrics import f1_score
import numpy as np
from tqdm import tqdm
import wandb

class MedicalQAModel:
    def __init__(self, model_name="bert-base-uncased", peft_technique="adapter"):
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
        print(f"\nSetting up PEFT with technique: {self.peft_technique}")
        if self.peft_technique == "adapter":
            print("Configuring LoRA adapter...")
            peft_config = LoraConfig(
                task_type=TaskType.QUESTION_ANSWERING_WITH_HEAD,  # Fixed: Using correct task type
                inference_mode=False,
                r=16,
                lora_alpha=32,
                lora_dropout=0.1,
                target_modules=['query', 'value']
            )
        elif self.peft_technique == "prompt":
            print("Configuring prompt tuning...")
            peft_config = PromptTuningConfig(
                task_type=TaskType.QUESTION_ANSWERING_WITH_HEAD,  # Fixed: Using correct task type
                num_virtual_tokens=20,
                prompt_tuning_init="TEXT",
                tokenizer_name_or_path=self.model_name
            )
        
        print("Applying PEFT configuration to model...")
        self.model = get_peft_model(self.model, peft_config)
        print("PEFT setup completed")

def main():
    print("\n=== Starting Medical QA System ===")
    
    print("\nLoading dataset...")
    dataset = load_dataset("Malikeh1375/medical-question-answering-datasets", "all-processed")
    dataset = dataset["train"].select(range(1000))
    
    # Print sample data
    print("\nSample data structure:")
    sample_item = dataset[0]
    print(f"Keys in dataset: {sample_item.keys()}")
    print(f"Example question: {sample_item['question'][:100]}...")
    print(f"Example context length: {len(sample_item['context'])} characters")
    
    print("\nInitializing model...")
    qa_model = MedicalQAModel(
        model_name="bert-base-uncased",
        peft_technique="adapter"
    )
    
    print("\nApplying pruning...")
    qa_model.apply_pruning(amount=0.3)
    
    print("\nCreating datasets and dataloaders...")
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
    
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=8,
        shuffle=True
    )
    eval_dataloader = DataLoader(
        eval_dataset,
        batch_size=8
    )
    
    print("\nStarting training...")
    train_model(
        qa_model.model,
        train_dataloader,
        eval_dataloader,
        num_epochs=3
    )

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\nError occurred: {str(e)}")
        print(f"Error type: {type(e).__name__}")
        import traceback
        print("\nFull traceback:")
        traceback.print_exc()
        if wandb.run is not None:
            wandb.log({"error": str(e)})
        raise
    finally:
        if wandb.run is not None:
            wandb.finish()