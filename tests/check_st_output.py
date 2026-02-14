
from sentence_transformers import SentenceTransformer
import torch

try:
    model = SentenceTransformer("google/embeddinggemma-300m", trust_remote_code=True)
    device = model.device
    tokenizer = model.tokenizer
    inputs = tokenizer(["hello world"], return_tensors='pt')
    inputs = {k: v.to(device) for k, v in inputs.items()}
    outputs = model(inputs)
    print(f"Output type: {type(outputs)}")
    if hasattr(outputs, 'last_hidden_state'):
         print("Output HAS .last_hidden_state")
    elif isinstance(outputs, dict) and 'last_hidden_state' in outputs:
         print("Output is dict and has 'last_hidden_state' key")
    elif isinstance(outputs, dict) and 'token_embeddings' in outputs:
         print("Output is dict and has 'token_embeddings' key")
         # AdapterX expects .last_hidden_state
    else:
         print(f"Output keys: {outputs.keys() if isinstance(outputs, dict) else 'Not a dict'}")

except Exception as e:
    print(f"Error: {e}")
