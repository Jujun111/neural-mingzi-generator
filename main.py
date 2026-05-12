import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

import argparse
import pickle
import torch
from torch.utils.data import DataLoader

from db_loader import get_cbdb_names
from dataset import Vocabulary, ChineseNameDataset
from model import ChineseNameLSTM, ChineseNameTransformer
from train import train_model
from inference import generate_name

def get_model(args, vocab_size, pad_idx, device, is_inference=False):
    """Instantiates the selected model architecture."""
    dropout_prob = 0.0 if is_inference else 0.4
    
    if args.model_type == "lstm":
        return ChineseNameLSTM(
            vocab_size=vocab_size,
            embedding_dim=256,
            hidden_size=512,
            num_layers=2,
            pad_idx=pad_idx,
            dropout_prob=dropout_prob
        ).to(device)
    elif args.model_type == "transformer":
        return ChineseNameTransformer(
            vocab_size=vocab_size,
            embedding_dim=256,
            num_heads=8,
            hidden_dim=512,
            num_layers=2,
            max_seq_len=10,
            pad_idx=pad_idx,
            dropout_prob=dropout_prob
        ).to(device)

def run_training(args):
    model_path = f"{args.model_type}_weights.pth"
    vocab_path = "vocab.pkl" if args.model_type == "lstm" else f"{args.model_type}_vocab.pkl"
    
    print("=== Phase 1: Data Ingestion ===")
    surnames, given_names = get_cbdb_names(dynasty_id=args.dynasty, limit=args.limit)
    full_names = [s + g for s, g in zip(surnames, given_names) if s and g]
    if not full_names:
        print("Error: No valid names retrieved from the database.")
        return

    print(f"Loaded {len(full_names)} complete names. Building vocabulary...")
    vocab = Vocabulary(frequency_threshold=1)
    vocab.build_vocabulary(full_names)

    print("=== Phase 2: Dataset Construction ===")
    dataset = ChineseNameDataset(full_names, vocab, max_seq_len=6)
    dataloader = DataLoader(dataset, batch_size=128, shuffle=True, drop_last=True)

    print(f"=== Phase 3: Model Initialization ({args.model_type.upper()}) ===")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = get_model(args, len(vocab), vocab.PAD_IDX, device, is_inference=False)

    print("=== Phase 4: Training Optimization ===")
    train_model(
        model=model, 
        dataloader=dataloader, 
        vocab_size=len(vocab), 
        pad_idx=vocab.PAD_IDX, 
        num_epochs=args.epochs, 
        learning_rate=args.lr,
        device=device
    )

    print("=== Phase 5: Artifact Serialization ===")
    torch.save(model.state_dict(), model_path)
    with open(vocab_path, 'wb') as f:
        pickle.dump(vocab, f)
    print(f"Success: Weights saved to {model_path}, Vocabulary saved to {vocab_path}.")

def run_inference(args):
    model_path = f"{args.model_type}_weights.pth"
    vocab_path = "vocab.pkl" if args.model_type == "lstm" else f"{args.model_type}_vocab.pkl"

    if not os.path.exists(model_path) or not os.path.exists(vocab_path):
        print(f"Error: Pre-trained artifacts not found for '{args.model_type}'. Please run with --train first.")
        return

    print("Loading vocabulary...")
    with open(vocab_path, 'rb') as f:
        vocab = pickle.load(f)

    print(f"Initializing {args.model_type.upper()} architecture and loading weights...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = get_model(args, len(vocab), vocab.PAD_IDX, device, is_inference=True)
    
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    
    print(f"\n--- Generating {args.count} Names ({args.model_type.upper()}) ---")
    print(f"Parameters: Temperature = {args.temperature}, Seed = {args.seed if args.seed else 'None'}")
    
    for i in range(args.count):
        name = generate_name(
            model=model, 
            vocab=vocab, 
            seed_text=args.seed, 
            max_length=4, 
            temperature=args.temperature,
            device=device
        )
        print(f"{i+1}. {name}")

def main():
    parser = argparse.ArgumentParser(description="Neural Chinese Name Generator.")
    
    # Mode flags
    parser.add_argument("--train", action="store_true", help="Execute the training pipeline.")
    parser.add_argument("--generate", action="store_true", help="Execute inference using saved weights.")
    
    # Model architecture selector
    parser.add_argument("--model_type", type=str, default="lstm", choices=["lstm", "transformer"], help="Choose the neural architecture.")
    
    # Training hyper-parameters
    parser.add_argument("--dynasty", type=int, default=None, help="Filter by Dynasty ID for training.")
    parser.add_argument("--limit", type=int, default=None, help="Limit DB records (for rapid testing).")
    parser.add_argument("--epochs", type=int, default=30, help="Number of training epochs. Default scaled to 30.")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate for Adam optimizer.")
    
    # Inference parameters
    parser.add_argument("--count", type=int, default=10, help="Number of names to generate.")
    parser.add_argument("--temperature", type=float, default=0.8, help="Softmax temperature (higher = more random).")
    parser.add_argument("--seed", type=str, default=None, help="Seed text to condition generation.")

    args = parser.parse_args()

    if args.train:
        run_training(args)
    elif args.generate:
        run_inference(args)
    else:
        print("Please specify an operational mode: --train or --generate.")
        parser.print_help()

if __name__ == "__main__":
    main()
