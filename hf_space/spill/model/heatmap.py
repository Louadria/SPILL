"""Keypoint extraction from heatmaps.

Adapted from the keypoint_detection package.
"""
from typing import List, Optional, Tuple
import torch


def get_keypoints_from_heatmap_batch_maxpool(
    heatmap: torch.Tensor,
    max_keypoints: int = 20,
    min_keypoint_pixel_distance: int = 1,
    abs_max_threshold: Optional[float] = None,
    rel_max_threshold: Optional[float] = None,
    return_scores: bool = False,
) -> List[List[List[Tuple[int, int]]]]:
    """Fast extraction of keypoints from a batch of heatmaps using maxpooling.

    Args:
        heatmap: NxCxHxW heatmap batch
        max_keypoints: max number of keypoints to extract per channel
        min_keypoint_pixel_distance: NMS distance
        abs_max_threshold: absolute score threshold
        rel_max_threshold: relative score threshold
        return_scores: whether to return scores alongside keypoints

    Returns:
        Nested list: [batch][channel][keypoint]
    """
    batch_size, n_channels, _, width = heatmap.shape

    kernel = min_keypoint_pixel_distance * 2 + 1
    pad = min_keypoint_pixel_distance
    padded_heatmap = torch.nn.functional.pad(heatmap, (pad, pad, pad, pad), mode="constant", value=1.0)
    max_pooled_heatmap = torch.nn.functional.max_pool2d(padded_heatmap, kernel, stride=1, padding=0)
    local_maxima = max_pooled_heatmap == heatmap
    heatmap = heatmap * local_maxima

    scores, indices = torch.topk(heatmap.view(batch_size, n_channels, -1), max_keypoints, sorted=True)
    indices = torch.stack([torch.div(indices, width, rounding_mode="floor"), indices % width], dim=-1)

    indices = indices.detach().cpu().numpy()
    scores = scores.detach().cpu().numpy()
    filtered_indices = [[[] for _ in range(n_channels)] for _ in range(batch_size)]
    filtered_scores = [[[] for _ in range(n_channels)] for _ in range(batch_size)]

    threshold = 0.01
    if abs_max_threshold is not None:
        threshold = max(threshold, abs_max_threshold)
    if rel_max_threshold is not None:
        threshold = max(threshold, rel_max_threshold * heatmap.max())

    for batch_idx in range(batch_size):
        for channel_idx in range(n_channels):
            candidates = indices[batch_idx, channel_idx]
            for candidate_idx in range(candidates.shape[0]):
                if scores[batch_idx, channel_idx, candidate_idx] > threshold:
                    filtered_indices[batch_idx][channel_idx].append(candidates[candidate_idx][::-1].tolist())
                    filtered_scores[batch_idx][channel_idx].append(scores[batch_idx, channel_idx, candidate_idx])

    if return_scores:
        return filtered_indices, filtered_scores
    else:
        return filtered_indices
