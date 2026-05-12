from typing import Iterable, Literal, Optional, Tuple

import opencc
import torch
import torch.nn.functional as F


ConstraintPosition = Literal["any", "start", "middle", "end"]


def normalize_prompt_text(
    prompt_text: Optional[str], converter: Optional[opencc.OpenCC] = None
) -> Tuple[Optional[str], bool]:
    """Normalize simplified Chinese prompts into the training domain."""
    if not prompt_text:
        return None, False

    active_converter = converter or opencc.OpenCC("s2t")
    normalized_text = active_converter.convert(prompt_text)
    return normalized_text, normalized_text != prompt_text


def normalize_seed_text(
    seed_text: Optional[str], converter: Optional[opencc.OpenCC] = None
) -> Tuple[Optional[str], bool]:
    return normalize_prompt_text(seed_text, converter=converter)


def token_matches_position(name: str, token: Optional[str], position: ConstraintPosition = "any") -> bool:
    if not token:
        return True

    if position == "any":
        return token in name

    if position == "start":
        return name.startswith(token)

    if position == "end":
        return name.endswith(token)

    if position == "middle":
        token_length = len(token)
        max_start = len(name) - token_length - 1
        if max_start < 1:
            return False
        return any(name[index : index + token_length] == token for index in range(1, max_start + 1))

    raise ValueError(f"Unsupported token position '{position}'.")


def token_is_single_occurrence(name: str, token: Optional[str]) -> bool:
    if not token:
        return True
    return name.count(token) == 1


def _is_stateful_model(model: torch.nn.Module) -> bool:
    """Use model structure instead of forward signature duck-typing."""
    return hasattr(model, "lstm") and not hasattr(model, "transformer")


def generate_name(
    model: torch.nn.Module,
    vocab,
    seed_text: Optional[str] = None,
    max_length: int = 4,
    temperature: float = 0.8,
    device: str = "cpu",
    converter: Optional[opencc.OpenCC] = None,
) -> str:
    """
    Autoregressively generate a Chinese name from either an LSTM or a Transformer.
    """
    model.eval()
    model.to(device)

    normalized_seed, _ = normalize_seed_text(seed_text, converter=converter)
    seed_indices = vocab.encode(normalized_seed or "")
    sequence_indices = [vocab.SOS_IDX, *seed_indices]
    is_stateful = _is_stateful_model(model)

    with torch.no_grad():
        hidden_state = None

        while len(sequence_indices) < max_length:
            if is_stateful:
                if hidden_state is None:
                    x_tensor = torch.tensor([sequence_indices], dtype=torch.long).to(device)
                else:
                    x_tensor = torch.tensor([[sequence_indices[-1]]], dtype=torch.long).to(device)

                logits, hidden_state = model(x_tensor, hidden=hidden_state)
            else:
                x_tensor = torch.tensor([sequence_indices], dtype=torch.long).to(device)
                logits, _ = model(x_tensor)

            next_char_logits = logits[0, -1, :]
            active_temperature = max(temperature, 1e-4)
            scaled_logits = next_char_logits / active_temperature

            # Force at least one character beyond the prompt.
            if len(sequence_indices) == len(seed_indices) + 1:
                scaled_logits[vocab.EOS_IDX] = -float("inf")

            probabilities = F.softmax(scaled_logits, dim=-1)
            next_index = torch.multinomial(probabilities, num_samples=1).item()

            if next_index == vocab.EOS_IDX:
                break

            sequence_indices.append(next_index)

    return vocab.decode(sequence_indices)


def build_fim_prompt(fixed_token: str, position: ConstraintPosition, seed_text: Optional[str]) -> str:
    seed_value = seed_text or ""
    return f"<TASK_INFILL> <TOKEN> {fixed_token} <POSITION> {position} <SEED> {seed_value} <SEP>"


def generate_fim_name(
    model: torch.nn.Module,
    vocab,
    fixed_token: str,
    position: ConstraintPosition = "any",
    seed_text: Optional[str] = None,
    max_new_tokens: int = 6,
    min_new_tokens: int = 2,
    temperature: float = 0.8,
    device: str = "cpu",
    converter: Optional[opencc.OpenCC] = None,
    allowed_output_indices: Optional[Iterable[int]] = None,
) -> str:
    """Generate a full name from a template FIM prompt."""
    model.eval()
    model.to(device)

    normalized_token, _ = normalize_prompt_text(fixed_token, converter=converter)
    normalized_seed, _ = normalize_seed_text(seed_text, converter=converter)
    prompt = build_fim_prompt(normalized_token or "", position, normalized_seed)
    prompt_indices = vocab.encode(prompt)
    sequence_indices = [vocab.SOS_IDX, *prompt_indices]
    generated_indices = []
    allowed_set = set(allowed_output_indices or [])

    with torch.no_grad():
        max_positions = getattr(getattr(model, "pos_embedding", None), "num_embeddings", None)
        for step in range(max_new_tokens):
            if max_positions is not None and len(sequence_indices) >= max_positions:
                break
            x_tensor = torch.tensor([sequence_indices], dtype=torch.long).to(device)
            logits, _ = model(x_tensor)
            scaled_logits = logits[0, -1, :] / max(temperature, 1e-4)

            for special_idx in (vocab.PAD_IDX, vocab.SOS_IDX, vocab.UNK_IDX):
                scaled_logits[special_idx] = -float("inf")

            if step < min_new_tokens:
                scaled_logits[vocab.EOS_IDX] = -float("inf")

            if allowed_set:
                mask = torch.full_like(scaled_logits, -float("inf"))
                for index in allowed_set:
                    if 0 <= index < mask.numel():
                        mask[index] = scaled_logits[index]
                if step > 0:
                    mask[vocab.EOS_IDX] = scaled_logits[vocab.EOS_IDX]
                scaled_logits = mask

            probabilities = F.softmax(scaled_logits, dim=-1)
            next_index = torch.multinomial(probabilities, num_samples=1).item()
            if next_index == vocab.EOS_IDX:
                break
            sequence_indices.append(next_index)
            generated_indices.append(next_index)

    return vocab.decode(generated_indices)
