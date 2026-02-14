
from sentence_transformers import SentenceTransformer
import torch

try:
    model = SentenceTransformer("google/embeddinggemma-300m", trust_remote_code=True)
    if hasattr(model, "tokenizer"):
        print("SentenceTransformer has .tokenizer attribute.")
    else:
        print("SentenceTransformer DOES NOT have .tokenizer attribute.")
        # Check components
        print(f"Modules: {model._modules.keys()}")
        if '0' in model._modules:
            mod0 = model._modules['0']
            if hasattr(mod0, 'tokenizer'):
                print("Module 0 has tokenizer.")
            if hasattr(mod0, 'auto_model'):
                print("Module 0 has auto_model.")

except Exception as e:
    print(f"Error loading model: {e}")
