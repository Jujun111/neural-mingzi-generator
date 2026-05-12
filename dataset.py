import torch
from torch.utils.data import Dataset, DataLoader
from collections import Counter

class Vocabulary:
    """
    Manages the bidirectional mapping between Chinese characters and integer indices.
    Implements special tokens required for sequence-to-sequence modeling.
    """
    def __init__(self, frequency_threshold=1):
        # 1. Define Reserved Special Tokens
        self.PAD_IDX = 0  # Padding token to ensure uniform tensor shapes in batches
        self.SOS_IDX = 1  # Start Of Sequence (initiates generation)
        self.EOS_IDX = 2  # End Of Sequence (terminates generation)
        self.UNK_IDX = 3  # Unknown token (handles Out-Of-Vocabulary characters)

        self.idx2char = {
            self.PAD_IDX: '<PAD>', 
            self.SOS_IDX: '<SOS>', 
            self.EOS_IDX: '<EOS>', 
            self.UNK_IDX: '<UNK>'
        }
        self.char2idx = {v: k for k, v in self.idx2char.items()}
        self.frequency_threshold = frequency_threshold

    def build_vocabulary(self, full_names_list):
        """
        Builds the character dictionary based on empirical frequencies.
        """
        char_counts = Counter()
        for name in full_names_list:
            char_counts.update(name)

        idx = 4 # Start indexing after special tokens
        for char, count in char_counts.items():
            if count >= self.frequency_threshold:
                self.char2idx[char] = idx
                self.idx2char[idx] = char
                idx += 1
                
        print(f"Vocabulary built. Total unique tokens: {len(self.char2idx)}")

    def encode(self, text):
        """Converts a string sequence to an integer list."""
        return [self.char2idx.get(char, self.UNK_IDX) for char in text]

    def decode(self, indices):
        """Converts an integer list back to a string, ignoring special tokens."""
        special_indices = {self.PAD_IDX, self.SOS_IDX, self.EOS_IDX, self.UNK_IDX}
        return "".join([self.idx2char[idx] for idx in indices if idx not in special_indices])

    def __len__(self):
        return len(self.char2idx)


class ChineseNameDataset(Dataset):
    """
    PyTorch Dataset wrapper for historical Chinese names.
    Prepares input (X) and target (Y) tensors for autoregressive LSTM training.
    """
    def __init__(self, full_names_list, vocab, max_seq_len=6):
        """
        Args:
            full_names_list (list): List of full strings (e.g., ["欧阳修", "李白"]).
            vocab (Vocabulary): The initialized and populated Vocabulary instance.
            max_seq_len (int): The maximum length of a name + special tokens.
        """
        self.vocab = vocab
        self.max_seq_len = max_seq_len
        self.X = []
        self.Y = []

        self._process_data(full_names_list)

    def _process_data(self, full_names_list):
        for name in full_names_list:
            if not name:
                continue
                
            # 1. Encode string to integers
            encoded_name = self.vocab.encode(name)
            
            # 2. Add structural tokens
            # Format: <SOS> + Surname + Given Name + <EOS>
            sequence = [self.vocab.SOS_IDX] + encoded_name + [self.vocab.EOS_IDX]
            
            # 3. Truncate if anomalously long (defensive programming)
            if len(sequence) > self.max_seq_len:
                sequence = sequence[:self.max_seq_len]
                sequence[-1] = self.vocab.EOS_IDX # Ensure it always ends with EOS
                
            # 4. Shift sequences for Teacher Forcing
            # Input X: <SOS> 欧 阳 修
            # Target Y: 欧 阳 修 <EOS>
            x_seq = sequence[:-1]
            y_seq = sequence[1:]
            
            # 5. Pad sequences to guarantee identical tensor dimensions for batching
            pad_len_x = self.max_seq_len - 1 - len(x_seq)
            x_padded = x_seq + [self.vocab.PAD_IDX] * pad_len_x
            
            pad_len_y = self.max_seq_len - 1 - len(y_seq)
            y_padded = y_seq + [self.vocab.PAD_IDX] * pad_len_y
            
            self.X.append(x_padded)
            self.Y.append(y_padded)

        # Convert to immutable PyTorch tensors in memory
        self.X = torch.tensor(self.X, dtype=torch.long)
        self.Y = torch.tensor(self.Y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]
