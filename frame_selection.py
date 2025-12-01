import torch
import torch.nn.functional as F


# Computes a differentiable summary vector by softly weighting frames.
def soft_select_frames(frames, scores, mask=None, temperature=1.0):
    if temperature <= 0:
        raise ValueError("temperature must be > 0")

    weights = scores / temperature
    if mask is not None:
        weights = weights.masked_fill(~mask, float("-inf"))

    weights = F.softmax(weights, dim=-1)
    summary = torch.bmm(weights.unsqueeze(1), frames)
    return summary.squeeze(1)


# Returns the top-k frame features per video along with a mask of valid picks.
def hard_topk_frames(frames, scores, mask=None, k=5):
    if k <= 0:
        raise ValueError("k must be > 0 for hard_topk_frames")

    if mask is not None:
        scores = scores.masked_fill(~mask, float("-inf"))

    max_k = min(k, scores.size(1))
    topk_idx = torch.topk(scores, max_k, dim=-1).indices

    batch_range = torch.arange(frames.size(0), device=frames.device).unsqueeze(-1)
    selected = frames[batch_range, topk_idx]

    if mask is not None:
        topk_mask = mask[batch_range, topk_idx]
        return selected, topk_mask

    return selected, None
