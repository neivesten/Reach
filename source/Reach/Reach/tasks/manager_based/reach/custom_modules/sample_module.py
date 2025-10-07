from isaaclab.managers import SceneEntityCfg
import torch
from isaaclab.utils.math import quat_from_euler_xyz, quat_mul
import isaacsim.core.utils.prims as prim_utils


# Randomizes the Yaw (Z Axis) of a Prim
def yaw_only_randomize(
    env,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg = SceneEntityCfg(name="battery"),
):
    obj = env.scene[asset_cfg.name]
    device = env.device
    B = env_ids.numel()

    # 1) Start from current root pose [x y z qx qy qz qw]
    pose = obj.data.root_state_w[env_ids, :7].clone()   # (B,7), keep dtype/device

    # 2) Sample yaw in [-pi, pi] and build world-Z rotation (xyzw)
    yaw = (torch.rand(B, device=device) * 2.0 - 1) * (torch.pi / 18) # ~10 degrees
    qz  = quat_from_euler_xyz(torch.zeros_like(yaw), torch.zeros_like(yaw), yaw)  # (B,4)

    # 3) Apply world-frame yaw: new_q = qz ⊗ q_current
    q_cur = pose[:, 3:7]                 # (B,4)
    q_new = quat_mul(qz, q_cur)          # (B,4)

    # 4) Write back orientation (position unchanged)
    pose[:, 3:7] = q_new
    obj.write_root_pose_to_sim(pose, env_ids=env_ids)

    # 5) Zero root velocities (good hygiene after teleport)
    zeros = torch.zeros((B, 6), device=device, dtype=pose.dtype)
    obj.write_root_velocity_to_sim(zeros, env_ids=env_ids)

    # Teleports a prim on call. Has flags to prevent 


# moves a prim downwards on the Z axis
def jump_table_once_interval(
    env,
    env_ids,                     # no default; Event Manager injects this
    asset_cfg,                   # SceneEntityCfg("table")
    delta_z: float = -1.0,
    hide_visual: bool = False,
):
    # one-shot flag
    device = env.device
    if not hasattr(env, "_table_jumped"):
        env._table_jumped = torch.zeros(env.num_envs, dtype=torch.bool, device=device)

    # clear on reset
    # if hasattr(env, "reset_buf"):
    #     ids = torch.nonzero(env.reset_buf, as_tuple=False).flatten()
    #     if ids.numel():
    #         env._table_jumped[ids] = False

    # act only on triggered envs that haven’t jumped
    env_ids = env_ids.to(device)
    act_ids = env_ids[~env._table_jumped[env_ids]]
    if act_ids.numel() == 0:
        return

    # resolve the rigid object (SceneEntityCfg is supported as a param)
    asset_name = getattr(asset_cfg, "name", asset_cfg)
    table = env.scene[asset_name]

    # write pose for only the active envs (supported API)
    pose = table.data.root_state_w[act_ids, :7].clone()
    pose[:, 2] += delta_z
    table.write_root_pose_to_sim(pose, env_ids=act_ids)

    # kinematic objects can retain old velocity: zero it to avoid artifacts
    zeros = torch.zeros((act_ids.numel(), 6), device=device)
    table.write_root_velocity_to_sim(zeros, env_ids=act_ids)

    env._table_jumped[act_ids] = True

    if hide_visual:
        try:
            # cosmetic; optional
            for i in act_ids.tolist():
                prim_utils.set_prim_visibility(table.prim_paths[i], "invisible")
        except Exception:
            pass


# Resets a custom Flag used in jump_table_once_interval
def clear_table_jump_flag(env, env_ids):
    if hasattr(env, "_table_jumped"):
        env._table_jumped[env_ids] = False


def print_filtered_contact_debug(env, sensor_name: str, env_ids=(0,), aggregate="max"):
    """
    Prints filtered (battery-only) and total (unfiltered) contact magnitudes
    for a given sensor after the scene has been updated.
    Call via an EventTerm running at a fixed interval.
    """
    s = env.scene[sensor_name]

    # --- FILTERED (battery-only) ---
    fm = getattr(s.data, "force_matrix_w", None)  # (E,B,M,3) or None
    if fm is None:
        print(f"[contact-debug] sensor={sensor_name} force_matrix_w=None")
    else:
        fm = torch.nan_to_num(fm, nan=0.0)
        if fm.ndim == 4:
            mags = fm.norm(dim=-1)  # (E,B,M)
            if aggregate == "max":
                filt = mags.amax(dim=(1,2))  # (E,)
            elif aggregate == "mag_sum":
                filt = mags.sum(dim=(1,2))   # (E,)
            else:  # vector_sum
                filt = fm.sum(dim=2).norm(dim=-1).amax(dim=1)  # (E,)
        elif fm.ndim == 3:
            filt = fm.norm(dim=-1).amax(dim=1)   # (E,)
        else:
            filt = fm.norm(dim=-1)               # (E,)

        for e in env_ids:
            print(f"[contact-debug] sensor={sensor_name} env={e} "
                  f"filtered_mag={float(filt[e]):.4f}")

    # --- TOTAL (unfiltered) ---
    net = getattr(s.data, "net_forces_w", None)  # (E,B,3) or (E,3)
    if net is not None:
        net = torch.nan_to_num(net, nan=0.0)
        if net.ndim == 3:
            net_mag = net.norm(dim=-1).amax(dim=1)  # (E,)
        else:
            net_mag = net.norm(dim=-1)              # (E,)
        for e in env_ids:
            print(f"[contact-debug] sensor={sensor_name} env={e} "
                  f"total_mag={float(net_mag[e]):.4f}")
 