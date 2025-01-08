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

class TestMedicalQA(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Initialisation des ressources communes à tous les tests"""
        print("\nInitializing test resources...")
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

    def test_model_initialization(self):
        """Test de l'initialisation du modèle"""
        print("\nTesting model initialization...")
        self.assertIsNotNone(self.qa_model)
        self.assertIsNotNone(self.qa_model.model)
        self.assertIsNotNone(self.qa_model.tokenizer)

    def test_dataset_creation(self):
        """Test de la création du dataset"""
        print("\nTesting dataset creation...")
        dataset = MedicalQADataset(self.sample_data, self.tokenizer, max_length=512)
        self.assertEqual(len(dataset), len(self.sample_data))
        
        # Test de l'extraction d'un item
        item = dataset[0]
        self.assertIn('input_ids', item)
        self.assertIn('attention_mask', item)
        self.assertIn('start_positions', item)
        self.assertIn('end_positions', item)

    def test_metrics_calculation(self):
        """Test du calcul des métriques"""
        print("\nTesting metrics calculation...")
        pred_text = "Les symptômes incluent la fièvre"
        true_text = "Les symptômes incluent la fièvre et la toux"
        
        # Test Exact Match
        em_score = QAMetrics.calculate_exact_match(pred_text, true_text)
        self.assertIsInstance(em_score, int)
        self.assertIn(em_score, [0, 1])
        
        # Test F1 Score
        f1_score = QAMetrics.calculate_f1(pred_text, true_text)
        self.assertIsInstance(f1_score, float)
        self.assertGreaterEqual(f1_score, 0)
        self.assertLessEqual(f1_score, 1)

    def test_model_prediction(self):
        """Test des prédictions du modèle"""
        print("\nTesting model prediction...")
        results = test_model(self.qa_model.model, self.tokenizer, self.test_questions)
        
        self.assertEqual(len(results), len(self.test_questions))
        for result in results:
            self.assertIn('question', result)
            self.assertIn('predicted_answer', result)
            self.assertIn('confidence', result)
            self.assertIsInstance(result['confidence'], float)
            self.assertGreaterEqual(result['confidence'], 0)
            self.assertLessEqual(result['confidence'], 1)

    def test_model_saving_loading(self):
        """Test de la sauvegarde et du chargement du modèle"""
        print("\nTesting model saving and loading...")
        save_path = "test_saved_model"
        
        try:
            # Sauvegarde du modèle
            self.qa_model.model.save_pretrained(save_path)
            self.qa_model.tokenizer.save_pretrained(save_path)
            
            # Vérification que les fichiers existent
            self.assertTrue(os.path.exists(save_path))
            
            # Nettoyage
            import shutil
            shutil.rmtree(save_path)
            
        except Exception as e:
            self.fail(f"Model saving/loading failed with error: {str(e)}")

    def tearDown(self):
        """Nettoyage après chaque test"""
        print("\nCleaning up after test...")
        torch.cuda.empty_cache()  # Libération de la mémoire GPU si utilisée

    @classmethod
    def tearDownClass(cls):
        """Nettoyage final après tous les tests"""
        print("\nCleaning up all test resources...")
        del cls.qa_model
        torch.cuda.empty_cache()

def run_tests():
    """Fonction pour exécuter tous les tests"""
    print("\n=== Starting Medical QA System Tests ===")
    unittest.main(verbosity=2)

if __name__ == '__main__':
    run_tests()