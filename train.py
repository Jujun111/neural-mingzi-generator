import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import List

def train_model(model: nn.Module, 
                dataloader: DataLoader, 
                vocab_size: int, 
                pad_idx: int, 
                num_epochs: int = 10, 
                learning_rate: float = 0.001, 
                device: str = 'cuda' if torch.cuda.is_available() else 'cpu') -> List[float]:
    """
    Executes the standard training loop for the autoregressive LSTM.
    
    Args:
        model (nn.Module): The instantiated ChineseNameLSTM.
        dataloader (DataLoader): PyTorch DataLoader yielding (X, Y) batches.
        vocab_size (int): Total vocabulary size (for reshaping).
        pad_idx (int): The integer index of the <PAD> token.
        num_epochs (int): Number of complete passes through the dataset.
        learning_rate (float): Step size for the Adam optimizer.
        device (str): Hardware accelerator ('cuda' or 'cpu').
        
    Returns:
        List[float]: A history of the average loss per epoch.
    """
    print(f"Initiating training on device: {device.upper()}")
    model.to(device)
    model.train() # Set module to training mode (activates Dropout)

    # 1. The Optimizer
    # Adam computes individual adaptive learning rates for different parameters
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # 2. The Loss Function
    # ignore_index ensures predictions made on <PAD> indices do not contribute to the loss gradient
    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)
    
    loss_history = []

    for epoch in range(num_epochs):
        total_loss = 0.0
        batch_count = 0

        for x_batch, y_batch in dataloader:
            # Transfer tensors to the appropriate hardware memory
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            # 1. Zero the gradients from the previous iteration
            optimizer.zero_grad()

            # 2. Forward Pass
            # logits shape: (batch_size, seq_len, vocab_size)
            logits, _ = model(x_batch)

            # 3. Tensor Reshaping for CrossEntropyLoss
            # PyTorch's CrossEntropyLoss expects inputs of shape (N, C) where C is the number of classes.
            # We must flatten the batch and sequence dimensions.
            # Flattened logits shape: (batch_size * seq_len, vocab_size)
            logits_flat = logits.view(-1, vocab_size)
            
            # Flattened targets shape: (batch_size * seq_len)
            y_flat = y_batch.view(-1)

            # 4. Compute Loss
            loss = criterion(logits_flat, y_flat)

            # 5. Backward Pass (Compute Gradients)
            loss.backward()

            # 6. Gradient Clipping
            # Prevents the "exploding gradient" problem inherent to unrolled RNNs/LSTMs
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)

            # 7. Optimizer Step (Update Weights)
            optimizer.step()

            total_loss += loss.item()
            batch_count += 1

        avg_epoch_loss = total_loss / batch_count
        loss_history.append(avg_epoch_loss)
        print(f"Epoch [{epoch+1}/{num_epochs}] - Average Loss: {avg_epoch_loss:.4f}")

    print("Training concluded.")
    return loss_history
