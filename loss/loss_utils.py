import torch


def compute_entropy(
    logits: torch.FloatTensor,
    labels: torch.LongTensor,
) -> torch.FloatTensor:
    """Compute the entropy of the policy distribution over tokens.

    Args:
        logits: Logits of the model (unnormalized). Shape: (batch_size, sequence_length, vocab_size)
        labels: Labels to determine which positions to compute entropy for.
                Label tokens with a value of -100 are ignored. Shape: (batch_size, sequence_length)

    Returns:
        A tensor of shape (batch_size, sequence_length-1) containing the entropy at each token position.
        Higher entropy = more uncertain/uniform distribution over vocabulary.
    """
    assert logits.shape[:-1] == labels.shape

    labels = labels[:, 1:].clone()
    logits = logits[:, :-1, :]
    loss_mask = labels != -100

    # Compute probabilities
    probs = logits.softmax(-1)  # Shape: (batch_size, seq_len, vocab_size)
    log_probs = logits.log_softmax(-1)

    # Entropy = -sum(p * log(p))
    entropy = -(probs * log_probs).sum(-1)  # Shape: (batch_size, seq_len)

    # Apply mask to ignore padding/special tokens
    entropy = entropy * loss_mask

    return entropy, loss_mask


def compute_kl(
    logits: torch.FloatTensor,
    reference_logits: torch.FloatTensor,
    labels: torch.LongTensor,
    direction: str = "ref_to_policy",
) -> tuple[torch.FloatTensor, torch.BoolTensor]:
    """Compute per-token KL divergence between policy and reference distributions.

    Args:
        logits: Policy logits. Shape: (batch_size, sequence_length, vocab_size)
        reference_logits: Reference logits. Shape: (batch_size, sequence_length, vocab_size)
        labels: Labels to determine which positions to compute KL for.
                Label tokens with a value of -100 are ignored. Shape: (batch_size, sequence_length)
        direction: Which KL to compute.
            - "ref_to_policy": KL(reference || policy)
            - "policy_to_ref": KL(policy || reference)

    Returns:
        A tuple (kl, loss_mask) where:
          - kl has shape (batch_size, sequence_length-1) and is zero at masked positions.
          - loss_mask has shape (batch_size, sequence_length-1) and is True where labels != -100.
    """
    assert logits.shape[:-1] == labels.shape
    assert reference_logits.shape[:-1] == labels.shape

    labels = labels[:, 1:].clone()
    logits = logits[:, :-1, :]
    reference_logits = reference_logits[:, :-1, :]
    loss_mask = labels != -100

    policy_log_probs = logits.log_softmax(-1)
    reference_log_probs = reference_logits.log_softmax(-1)

    direction = direction.lower()
    if direction in {"ref_to_policy", "reference_to_policy", "ref"}:
        reference_probs = reference_log_probs.exp()
        kl = (reference_probs * (reference_log_probs - policy_log_probs)).sum(-1)
    elif direction in {"policy_to_ref", "policy"}:
        policy_probs = policy_log_probs.exp()
        kl = (policy_probs * (policy_log_probs - reference_log_probs)).sum(-1)
    else:
        raise ValueError(
            f"Unknown KL direction: {direction}. Use 'ref_to_policy' or 'policy_to_ref'."
        )

    kl = kl * loss_mask
    return kl, loss_mask


def Q_tbpo_get_batch_logps(
    logits: torch.FloatTensor,
    reference_logits: torch.FloatTensor,
    labels: torch.LongTensor,
):
    assert logits.shape[:-1] == labels.shape
    assert reference_logits.shape[:-1] == labels.shape

    labels = labels[:, 1:].clone()
    logits = logits[:, :-1, :]
    reference_logits = reference_logits[:, :-1, :]

    loss_mask = labels != -100

    # dummy token; we'll ignore the losses on these tokens later
    labels[labels == -100] = 0

    per_token_logps = torch.gather(
        logits.log_softmax(-1), dim=2, index=labels.unsqueeze(2)
    ).squeeze(2)
    per_reference_token_logps = torch.gather(
        reference_logits.log_softmax(-1), dim=2, index=labels.unsqueeze(2)
    ).squeeze(2)

    logps_margin = per_token_logps - per_reference_token_logps

    return (
        logps_margin * loss_mask,
        per_token_logps * loss_mask,
    )


def A_tbpo_get_batch_logps(
    logits: torch.FloatTensor,
    reference_logits: torch.FloatTensor,
    labels: torch.LongTensor,
    average_log_prob: bool = False,
):
    """also return token KL"""
    pass


def _tdpo_get_batch_logps(
    logits: torch.FloatTensor,
    reference_logits: torch.FloatTensor,
    labels: torch.LongTensor,
    average_log_prob: bool = False,
):
    """Compute the kl divergence/log probabilities of the given labels under the given logits.

    Args:
        logits: Logits of the model (unnormalized). Shape: (batch_size, sequence_length, vocab_size)
        reference_logits: Logits of the reference model (unnormalized). Shape: (batch_size, sequence_length, vocab_size)
        labels: Labels for which to compute the log probabilities. Label tokens with a value of -100 are ignored. Shape: (batch_size, sequence_length)
        average_log_prob: If True, return the average log probability per (non-masked) token. Otherwise, return the sum of the log probabilities of the (non-masked) tokens.

    Returns:
        Several tensors of shape (batch_size,) containing the average/sum kl divergence/log probabilities of the given labels under the given logits.
    """
    assert logits.shape[:-1] == labels.shape
    assert reference_logits.shape[:-1] == labels.shape

    labels = labels[:, 1:].clone()
    logits = logits[:, :-1, :]
    reference_logits = reference_logits[:, :-1, :]

    loss_mask = labels != -100

    # dummy token; we'll ignore the losses on these tokens later
    labels[labels == -100] = 0

    vocab_logps = logits.log_softmax(-1)

    reference_vocab_ps = reference_logits.softmax(-1)
    reference_vocab_logps = reference_vocab_ps.log()

    per_position_kl = (reference_vocab_ps * (reference_vocab_logps - vocab_logps)).sum(-1)
    per_token_logps = torch.gather(vocab_logps, dim=2, index=labels.unsqueeze(2)).squeeze(2)
    per_reference_token_logps = torch.gather(
        reference_vocab_logps, dim=2, index=labels.unsqueeze(2)
    ).squeeze(2)

    logps_margin = per_token_logps - per_reference_token_logps

    if average_log_prob:
        return (
            (logps_margin * loss_mask).sum(-1) / loss_mask.sum(-1),
            (per_position_kl * loss_mask).sum(-1) / loss_mask.sum(-1),
            (per_token_logps * loss_mask).sum(-1) / loss_mask.sum(-1),
        )
    else:
        return (
            (logps_margin * loss_mask).sum(-1),
            (per_position_kl * loss_mask).sum(-1),
            (per_token_logps * loss_mask).sum(-1),
        )


def _get_batch_logps(
    logits: torch.FloatTensor,
    labels: torch.LongTensor,
    average_log_prob: bool = False,
    token_level: bool = False,
) -> torch.FloatTensor:
    """Compute the log probabilities of the given labels under the given logits.

    Args:
        logits: Logits of the model (unnormalized). Shape: (batch_size, sequence_length, vocab_size)
        labels: Labels for which to compute the log probabilities. Label tokens with a value of -100 are ignored. Shape: (batch_size, sequence_length)
        average_log_prob: If True, return the average log probability per (non-masked) token. Otherwise, return the sum of the log probabilities of the (non-masked) tokens.

    Returns:
        A tensor of shape (batch_size,) containing the average/sum log probabilities of the given labels under the given logits.
    """
    assert logits.shape[:-1] == labels.shape

    labels = labels[:, 1:].clone()
    logits = logits[:, :-1, :]
    loss_mask = labels != -100

    # dummy token; we'll ignore the losses on these tokens later
    labels[labels == -100] = 0

    per_token_logps = torch.gather(
        logits.log_softmax(-1), dim=2, index=labels.unsqueeze(2)
    ).squeeze(2)
    # import ipdb; ipdb.set_trace()
    batch_logps = (per_token_logps * loss_mask).sum(-1)

    if average_log_prob:
        return batch_logps / loss_mask.sum(-1)
    else:
        return batch_logps


def _get_batch_logps_tisdpo(
    logits: torch.FloatTensor,
    reference_logits: torch.FloatTensor,
    labels: torch.LongTensor,
    weights: torch.FloatTensor = None,
    average_log_prob: bool = False,
) -> torch.FloatTensor:
    """Compute the log probabilities of the given labels under the given logits.

    Args:
        logits: Logits of the model (unnormalized). Shape: (batch_size, sequence_length, vocab_size)
        reference_logits: Logits of the reference model (unnormalized). Shape: (batch_size, sequence_length, vocab_size)
        labels: Labels for which to compute the log probabilities. Label tokens with a value of -100 are ignored. Shape: (batch_size, sequence_length)
        weights: Weights for each token. Shape: (batch_size, sequence_length)
        average_log_prob: If True, return the average log probability per
        (non-masked) token. Otherwise, return the sum of the log probabilities of the (non-masked) tokens.
        token_level: If True, return the log probability per (non-masked) token. Otherwise, return the sum of the log probabilities of the (non-masked) tokens.

    Returns:
        A tensor of shape (batch_size,) containing the average/sum log probabilities of the given labels under the given logits.
    """
    assert logits.shape[:-1] == labels.shape

    labels = labels[:, 1:].clone()
    logits = logits[:, :-1, :]
    reference_logits = reference_logits[:, :-1, :]

    loss_mask = labels != -100

    labels[labels == -100] = 0

    vocab_ps = logits.softmax(-1)
    vocab_logps = vocab_ps.log()

    reference_vocab_ps = reference_logits.softmax(-1)
    reference_vocab_logps = reference_vocab_ps.log()

    per_position_kl = (vocab_ps * (vocab_logps - reference_vocab_logps)).sum(-1)

    per_token_logps = torch.gather(vocab_logps, dim=2, index=labels.unsqueeze(2)).squeeze(2)
    per_reference_token_logps = torch.gather(
        reference_vocab_logps, dim=2, index=labels.unsqueeze(2)
    ).squeeze(2)

    logps_margin = per_token_logps - per_reference_token_logps
    weights = weights[:, 1:].clone()

    if average_log_prob:
        return (
            (logps_margin * weights * loss_mask).sum(-1) / loss_mask.sum(-1),
            (per_position_kl * weights * loss_mask).sum(-1) / loss_mask.sum(-1),
            (per_token_logps * weights * loss_mask).sum(-1) / loss_mask.sum(-1),
        )
    else:
        return (
            (logps_margin * weights * loss_mask).sum(-1),
            (per_position_kl * weights * loss_mask).sum(-1),
            (per_token_logps * weights * loss_mask).sum(-1),
        )
