import unittest
from tp import (
    MedicalQAModel,
    MedicalQADataset,
    QAMetrics,
    test_model,
    evaluate_model
)
import torch
from transformers import AutoTokenizer
import os
from torch.utils.data import DataLoader

class TestMedicalQA(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Initialisation des ressources communes à tous les tests"""
        print("\n=== Initializing test resources ===")
        cls.model_name = "bert-base-uncased"
        cls.qa_model = MedicalQAModel(model_name=cls.model_name, peft_technique="adapter")
        cls.tokenizer = AutoTokenizer.from_pretrained(cls.model_name)
        cls.test_questions = [
            (
                "Quels sont les symptômes de la COVID-19?",
                "La COVID-19 peut provoquer divers symptômes. Les plus courants sont la fièvre, la toux sèche et la fatigue."
            ),
            (
                "Comment traiter l'hypertension?",
                "Le traitement de l'hypertension combine souvent des changements de mode de vie et des médicaments."
            )
        ]

    def setUp(self):
        """Initialisation avant chaque test"""
        print("\nSetting up test...")
        self.sample_data = [
            {
                'instruction': "Répondre à la question médicale",
                'input': "Quels sont les symptômes du diabète?",
                'output': "Les symptômes principaux sont la soif excessive et la fatigue."
            },
            {
                'instruction': "Répondre à la question médicale",
                'input': "Comment traiter la migraine?",
                'output': "Les traitements incluent des analgésiques et le repos."
            }
        ]

    def test_detailed_evaluation(self):
        """Test détaillé avec évaluation des métriques"""
        print("\n=== Running Detailed Evaluation Test ===")
        
        # Création des datasets
        train_dataset = MedicalQADataset(self.sample_data, self.tokenizer, max_length=512)
        eval_dataset = MedicalQADataset(self.sample_data, self.tokenizer, max_length=512)
        
        # Création des dataloaders
        train_dataloader = DataLoader(train_dataset, batch_size=2, shuffle=True)
        eval_dataloader = DataLoader(eval_dataset, batch_size=2)
        
        # Évaluation du modèle
        print("\nCalculating evaluation metrics...")
        
        # Test avec des questions spécifiques
        test_results = test_model(self.qa_model.model, self.tokenizer, self.test_questions)
        
        print("\n=== Detailed Evaluation Results ===")
        print("\n1. Test Questions Results:")
        for i, result in enumerate(test_results, 1):
            print(f"\nQuestion {i}:")
            print(f"Q: {result['question']}")
            print(f"A: {result['predicted_answer']}")
            print(f"Confidence: {result['confidence']:.2%}")
        
        # Calcul des métriques
        print("\n2. Metrics Calculation Examples:")
        pred_text = "Les symptômes incluent la fièvre"
        true_text = "Les symptômes incluent la fièvre et la toux"
        
        em_score = QAMetrics.calculate_exact_match(pred_text, true_text)
        f1 = QAMetrics.calculate_f1(pred_text, true_text)
        cosine_sim = QAMetrics.calculate_cosine_similarity(pred_text, true_text)
        
        print("\nExample Metrics:")
        print(f"Exact Match Score: {em_score}")
        print(f"F1 Score: {f1:.4f}")
        print(f"Cosine Similarity: {cosine_sim:.4f}")
        
        # Évaluation sur le dataset
        print("\n3. Dataset Evaluation:")
        metrics = evaluate_model(self.qa_model.model, eval_dataloader)
        
        print("\nFinal Evaluation Metrics:")
        for metric_name, value in metrics.items():
            print(f"{metric_name}: {value:.4f}")

    def tearDown(self):
        """Nettoyage après chaque test"""
        print("\nCleaning up after test...")
        torch.cuda.empty_cache()

    @classmethod
    def tearDownClass(cls):
        """Nettoyage final après tous les tests"""
        print("\n=== Cleaning up all test resources ===")
        del cls.qa_model
        torch.cuda.empty_cache()

def run_tests():
    """Fonction pour exécuter tous les tests avec focus sur l'évaluation"""
    print("\n=== Starting Medical QA System Tests with Detailed Metrics ===")
    suite = unittest.TestSuite()
    suite.addTest(TestMedicalQA('test_detailed_evaluation'))
    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(suite)

if __name__ == '__main__':
    run_tests()