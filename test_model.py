import os
import unittest
from city_gpt_utils import load_model, build_prompt, generate_names
from city_gpt import MODEL_PATH

class TestCityGPTModel(unittest.TestCase):
    def setUp(self):
        """Skip all tests in this suite if the model doesn't exist."""
        if not os.path.exists(MODEL_PATH):
            self.skipTest(f"Model checkpoint not found at {MODEL_PATH}. Skipping tests.")

    def test_model_loads_successfully(self):
        """Test that the model and vocabulary can be loaded."""
        model, stoi, itos = load_model()
        self.assertIsNotNone(model)
        self.assertIsInstance(stoi, dict)
        self.assertIsInstance(itos, dict)
        # Ensure our special tokens are in the vocabulary
        self.assertIn("<SOS>", stoi)
        self.assertIn("\n", stoi)

    def test_prompt_builder(self):
        """Test that the prompt builder formats strings correctly."""
        prompt_us_ppl = build_prompt("US", "P", "PPL")
        self.assertEqual(prompt_us_ppl, "🇺🇸🏙️PPL|")
        
        prompt_random = build_prompt("", "", "")
        self.assertEqual(prompt_random, "")
        
        prompt_with_prefix = build_prompt("US", "P", "PPL", "New ")
        self.assertEqual(prompt_with_prefix, "🇺🇸🏙️PPL|new ")

    def test_model_generation(self):
        """Test that the model can successfully generate names."""
        prompt = build_prompt("US", "P", "PPL")
        
        # Generate 2 names
        names = generate_names(prompt_str=prompt, num_names=2, max_new_tokens=30)
        
        # Verify the output
        self.assertEqual(len(names), 2)
        for name in names:
            self.assertIsInstance(name, str)
            self.assertTrue(len(name) > 0, "Generated name should not be empty")

if __name__ == "__main__":
    unittest.main()
