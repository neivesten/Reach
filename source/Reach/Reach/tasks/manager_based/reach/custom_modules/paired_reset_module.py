"""Utilities for resetting multiple prims to share a random XY position."""

from __future__ import annotations

from typing import Sequence, Tuple

import torch


def reset_prims_to_shared_xy(
    env,
    env_ids: torch.Tensor,
    prim_names: Sequence[str],
    x_range: Tuple[float, float] = (-0.5, 0.5),
    y_range: Tuple[float, float] = (-0.5, 0.5),
) -> None:
    """Repositions the specified prims so they share a randomly sampled XY location.

    The function samples a single XY target per environment, within the provided ranges,
    and teleports all requested prims in that environment to the same location while
    keeping their original Z height and orientation. Root linear and angular velocities
    are cleared to avoid residual motion after teleporting.

    Args:
        env: The manager-based environment providing access to the scene and simulator.
        env_ids: Tensor of environment indices that should be affected.
        prim_names: Names of the prims to reposition. Each name must correspond to an
            entry registered in ``env.scene``.
        x_range: Inclusive range ``(min, max)`` in which to sample the X coordinate.
        y_range: Inclusive range ``(min, max)`` in which to sample the Y coordinate.
    """
    if not prim_names:
        return

    device = env.device
    env_ids = env_ids.to(device)
    if env_ids.numel() == 0:
        return

    x_min, x_max = x_range
    y_min, y_max = y_range

    # Sample a single XY per environment and broadcast it to every prim.
    xs = torch.rand((env_ids.numel(),), device=device) * (x_max - x_min) + x_min
    ys = torch.rand((env_ids.numel(),), device=device) * (y_max - y_min) + y_min

    for name in prim_names:
        obj = env.scene[name]

        pose = obj.data.root_state_w[env_ids, :7].clone()
        pose[:, 0] = xs
        pose[:, 1] = ys

        obj.write_root_pose_to_sim(pose, env_ids=env_ids)

        zeros = torch.zeros((env_ids.numel(), 6), device=device, dtype=pose.dtype)
        obj.write_root_velocity_to_sim(zeros, env_ids=env_ids)
