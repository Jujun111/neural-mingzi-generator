import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

import torch
import pickle
from main import get_model
from inference import generate_name
from evaluate import evaluate_generation_diversity
from markov_chain import ChineseNameMarkov
from db_loader import get_cbdb_names

def evaluate_markov(count=1000):
    print(f"\n--- Generating {count} names with MARKOV ---")
    surnames, given_names = get_cbdb_names(limit=None)
    
    markov = ChineseNameMarkov(surnames_list=surnames, boost_compound=False)
    markov.train(given_names)
    
    generations = []
    for _ in range(count):
        generations.append(markov.generate(max_length=2))
        
    evaluate_generation_diversity(generations)

def evaluate_model(model_type, count=1000):
    model_path = f"{model_type}_weights.pth"
    vocab_path = "vocab.pkl" if model_type == "lstm" else f"{model_type}_vocab.pkl"
    
    if not os.path.exists(model_path):
        print(f"Error: {model_path} not found.")
        return
        
    with open(vocab_path, 'rb') as f:
        vocab = pickle.load(f)
        
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    class ArgsMock:
        pass
    args = ArgsMock()
    args.model_type = model_type
    
    model = get_model(args, len(vocab), vocab.PAD_IDX, device, is_inference=True)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    
    print(f"\n--- Generating {count} names with {model_type.upper()} ---")
    
    generations = []
    # Generation might take a few seconds on CPU
    for i in range(count):
        name = generate_name(model, vocab, seed_text=None, max_length=4, temperature=0.8, device=device)
        generations.append(name)
        
    evaluate_generation_diversity(generations)

if __name__ == "__main__":
    print("Initiating Comparative Diversity Evaluation on Sequence Models...")
    evaluate_model("lstm", count=1000)
    evaluate_model("transformer", count=1000)
    evaluate_markov(count=1000)
