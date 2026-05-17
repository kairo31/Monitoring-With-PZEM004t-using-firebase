"""Budget-aware RL v2 utilities for five household devices.

Model v2 uses a custom PyTorch actor/critic checkpoint instead of the
Stable-Baselines3 zip format used by the previous PPO agent.  The public
inference API intentionally mimics SB3's ``predict`` method so the worker can
use either model family while the application migrates.
"""

from __future__ import annotations

import calendar
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical

TARIF_PER_KWH = float(os.getenv("TARIF_PER_KWH", "1352.0"))
DEFAULT_MONTHLY_TARGET_RP = float(os.getenv("DEFAULT_MONTHLY_TARGET_RP", "250000"))
CALIBRATION_CURRENT_CORRECTION = float(os.getenv("PZEM_CURRENT_CORRECTION", "1.018"))
CALIBRATION_VOLTAGE_CORRECTION = float(os.getenv("PZEM_VOLTAGE_CORRECTION", "1.0"))
HIDDEN_DIM = int(os.getenv("RL_HIDDEN_DIM", "128"))
OBS_DIM = 16
N_ACTIONS = 6
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DEVICE_ORDER = ("AC", "Waterheater", "TV", "Magicom", "Laptop")
ACTION_LABELS = {
    0: "Tidak ada tindakan",
    1: "Matikan / kurangi AC",
    2: "Matikan / kurangi Waterheater",
    3: "Mode hemat TV",
    4: "Mode hemat Magicom",
    5: "Mode hemat Laptop",
}

# Raw-state scale used by the network.  The worker and trainer both pass raw
# values in rupiah/watt/hour units; normalization happens inside the network.
OBS_SCALE = np.array(
    [
        5000.0,  # daya
        60.0,  # suhu
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,  # five device flags
        23.0,  # jam
        6.0,  # day_of_week
        1.0,  # is_weekend
        1.0,  # is_peak_hour
        1.0,  # day_progress
        31.0,  # days_remaining
        5000.0,  # prediksi_watt
        2_000_000.0,  # monthly target
        2_000_000.0,  # gap to target
    ],
    dtype=np.float32,
)


@dataclass
class TrainRow:
    daya: float
    suhu: float
    arus: float
    tegangan: float
    jam: float
    day_of_week: float
    is_weekend: float
    day_progress: float
    days_remaining: float


class ActorNetwork(nn.Module):
    """Policy network π(a|s) for the v2 checkpoint."""

    def __init__(self, obs_dim: int = OBS_DIM, n_actions: int = N_ACTIONS, hidden: int = HIDDEN_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, n_actions),
        )

    def forward(self, x: torch.Tensor) -> Categorical:
        scale = torch.as_tensor(OBS_SCALE, dtype=x.dtype, device=x.device)
        x_norm = torch.clamp(x / scale, min=-1.5, max=1.5)
        return Categorical(logits=self.net(x_norm))

    def get_action(self, state: np.ndarray) -> tuple[int, torch.Tensor, torch.Tensor]:
        t = torch.as_tensor(state, dtype=torch.float32, device=DEVICE).unsqueeze(0)
        dist = self(t)
        action = dist.sample()
        return int(action.item()), dist.log_prob(action), dist.entropy()


class CriticNetwork(nn.Module):
    """Value network V(s) for the v2 training loop."""

    def __init__(self, obs_dim: int = OBS_DIM, hidden: int = HIDDEN_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = torch.as_tensor(OBS_SCALE, dtype=x.dtype, device=x.device)
        x_norm = torch.clamp(x / scale, min=-1.5, max=1.5)
        return self.net(x_norm).squeeze(-1)


class ObservationSpace:
    """Small compatibility shim matching the SB3 attribute used by the worker."""

    shape = (OBS_DIM,)


class RLBudgetV2Agent:
    """Inference wrapper with a Stable-Baselines-like ``predict`` method."""

    observation_space = ObservationSpace()

    def __init__(self, actor: ActorNetwork, model_path: str | Path):
        self.actor = actor
        self.model_path = str(model_path)

    @classmethod
    def load(cls, model_path: str | Path, device: torch.device | None = None) -> "RLBudgetV2Agent":
        runtime_device = device or DEVICE
        checkpoint = torch.load(str(model_path), map_location=runtime_device)
        actor = ActorNetwork(
            obs_dim=int(checkpoint.get("obs_dim", OBS_DIM)),
            n_actions=int(checkpoint.get("n_actions", N_ACTIONS)),
            hidden=int(checkpoint.get("hidden_dim", HIDDEN_DIM)),
        ).to(runtime_device)
        actor.load_state_dict(checkpoint["actor"])
        actor.eval()
        return cls(actor=actor, model_path=model_path)

    def predict(self, state: np.ndarray | Iterable[float], deterministic: bool = True) -> tuple[int, None]:
        arr = np.asarray(state, dtype=np.float32)
        if arr.shape[-1] != OBS_DIM:
            raise ValueError(f"RL v2 membutuhkan state {OBS_DIM} dimensi, menerima {arr.shape[-1]} dimensi.")
        with torch.no_grad():
            t = torch.as_tensor(arr, dtype=torch.float32, device=DEVICE).unsqueeze(0)
            dist = self.actor(t)
            if deterministic:
                action = int(torch.argmax(dist.probs, dim=-1).item())
            else:
                action = int(dist.sample().item())
        return action, None


def device_flags_from_power(daya: float) -> tuple[int, int, int, int, int]:
    """Infer AC, Waterheater, TV, Magicom, Laptop flags from a single watt value.

    Because one aggregate PZEM reading cannot perfectly disaggregate overlapping
    appliances, ranges can overlap.  Runtime inference prefers explicit
    ``log_konfirmasi`` device states from Firebase.
    """

    flag_ac = 1 if daya > 300 else 0
    flag_waterheater = 1 if 220 < daya <= 300 else 0
    flag_tv = 1 if 35 < daya <= 100 else 0
    flag_magicom = 1 if 120 < daya <= 450 else 0
    flag_laptop = 1 if 60 < daya <= 220 else 0
    return flag_ac, flag_waterheater, flag_tv, flag_magicom, flag_laptop


def prediksi_watt_from_row(row: TrainRow) -> float:
    if row.tegangan > 0 and row.arus > 0:
        return max(row.tegangan * CALIBRATION_VOLTAGE_CORRECTION * row.arus * CALIBRATION_CURRENT_CORRECTION, 0.0)
    return max(row.daya, 0.0)


def build_state_from_values(
    *,
    daya: float,
    suhu: float,
    device_state: dict[str, int] | None,
    jam: float,
    day_of_week: float,
    is_weekend: float,
    day_progress: float,
    days_remaining: float,
    prediksi_watt: float,
    monthly_target_rp: float,
    gap_to_target: float,
) -> np.ndarray:
    if device_state is None:
        ac, waterheater, tv, magicom, laptop = device_flags_from_power(daya)
    else:
        ac = 1 if device_state.get("AC") else 0
        waterheater = 1 if device_state.get("Waterheater") else 0
        tv = 1 if device_state.get("TV") else 0
        magicom = 1 if device_state.get("Magicom") else 0
        laptop = 1 if device_state.get("Laptop") else 0

    is_peak_hour = 1.0 if 17 <= float(jam) <= 22 else 0.0
    return np.array(
        [
            float(max(daya, 0.0)),
            float(np.clip(suhu, 10.0, 60.0)),
            ac,
            waterheater,
            tv,
            magicom,
            laptop,
            float(jam),
            float(day_of_week),
            float(is_weekend),
            is_peak_hour,
            float(day_progress),
            float(days_remaining),
            float(max(prediksi_watt, 0.0)),
            float(max(monthly_target_rp, 10_000.0)),
            float(gap_to_target),
        ],
        dtype=np.float32,
    )


def build_state_from_row(row: TrainRow, monthly_target_rp: float) -> np.ndarray:
    prediksi_watt = prediksi_watt_from_row(row)
    est_daily_cost = (prediksi_watt / 1000.0) * 24 * TARIF_PER_KWH
    gap_to_target = (est_daily_cost * 30) - monthly_target_rp
    return build_state_from_values(
        daya=row.daya,
        suhu=row.suhu,
        device_state=None,
        jam=row.jam,
        day_of_week=row.day_of_week,
        is_weekend=row.is_weekend,
        day_progress=row.day_progress,
        days_remaining=row.days_remaining,
        prediksi_watt=prediksi_watt,
        monthly_target_rp=monthly_target_rp,
        gap_to_target=gap_to_target,
    )


def checkpoint_metadata() -> dict[str, Any]:
    return {
        "version": "budget_rl_v2_five_devices",
        "obs_dim": OBS_DIM,
        "n_actions": N_ACTIONS,
        "hidden_dim": HIDDEN_DIM,
        "device_order": list(DEVICE_ORDER),
        "action_labels": ACTION_LABELS,
        "obs_scale": OBS_SCALE.tolist(),
        "tarif_per_kwh": TARIF_PER_KWH,
    }


def save_checkpoint(path: str | Path, actor: ActorNetwork, critic: CriticNetwork, training_logs: list[dict[str, Any]]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            **checkpoint_metadata(),
            "actor": actor.state_dict(),
            "critic": critic.state_dict(),
            "training_logs_tail": training_logs[-20:],
        },
        str(output),
    )


def write_training_log(path: str | Path, training_logs: list[dict[str, Any]]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        json.dump(training_logs, handle, indent=2)


def month_progress(waktu) -> tuple[float, float]:
    days = calendar.monthrange(waktu.year, waktu.month)[1]
    return float(waktu.day / max(days, 1)), float(max(days - waktu.day, 0))
