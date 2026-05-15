

from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Any

import firebase_admin
import numpy as np
import pandas as pd
import pytz
import torch
import torch.nn as nn
import torch.optim as optim
from firebase_admin import credentials, db

from rl_budget_v2 import (
    ACTION_LABELS,
    DEVICE,
    HIDDEN_DIM,
    N_ACTIONS,
    OBS_DIM,
    TARIF_PER_KWH,
    ActorNetwork,
    CriticNetwork,
    TrainRow,
    build_state_from_row,
    month_progress,
    prediksi_watt_from_row,
    save_checkpoint,
    write_training_log,
)

DB_URL = os.getenv(
    "FIREBASE_DB_URL",
    "https://test-reading-the-pzem-default-rtdb.asia-southeast1.firebasedatabase.app",
)
HISTORY_NODE = os.getenv("FIREBASE_HISTORY_NODE", "history")
FIREBASE_CREDENTIALS = os.getenv(
    "FIREBASE_CREDENTIALS",
    "/content/drive/MyDrive/SKRIPSHIT/firebase-project/firebase-project/serviceAccountKey.json",
)
MODEL_OUTPUT_PATH = Path(os.getenv("RL_MODEL_OUTPUT", "model_ai/model_rl_budget_v2.pt"))
DEFAULT_MONTHLY_TARGET_RP = float(os.getenv("DEFAULT_MONTHLY_TARGET_RP", "250000"))
TOTAL_TIMESTEPS = int(os.getenv("RL_TOTAL_TIMESTEPS", "40000"))
LR_ACTOR = float(os.getenv("RL_LR_ACTOR", "3e-4"))
LR_CRITIC = float(os.getenv("RL_LR_CRITIC", "1e-3"))
GAMMA = float(os.getenv("RL_GAMMA", "0.99"))
LAMBDA_GAE = float(os.getenv("RL_LAMBDA_GAE", "0.95"))
CLIP_EPS = float(os.getenv("RL_CLIP_EPS", "0.2"))
ENTROPY_COEF = float(os.getenv("RL_ENTROPY_COEF", "0.01"))
BATCH_SIZE = int(os.getenv("RL_BATCH_SIZE", "512"))
N_EPOCHS = int(os.getenv("RL_N_EPOCHS", "10"))
MINIBATCH_SIZE = int(os.getenv("RL_MINIBATCH_SIZE", "128"))


class BudgetEnergyEnvV2:
    """Budget-aware environment for aggregate PZEM data and five devices."""

    IDX_PREDIKSI_WATT = 13
    IDX_GAP_TO_TARGET = 15
    REDUCTION = {
        0: 0.00,  # no action
        1: 0.30,  # AC
        2: 0.20,  # Waterheater
        3: 0.05,  # TV
        4: 0.08,  # Magicom
        5: 0.05,  # Laptop
    }

    def __init__(self, rows: list[TrainRow], monthly_target_rp: float):
        self.rows = rows
        self.monthly_target_rp = max(float(monthly_target_rp), 10_000.0)
        self.i = 0
        self.prev_action = 0

    def reset(self, random_start: bool = True) -> np.ndarray:
        max_start = max(len(self.rows) - 2, 0)
        self.i = random.randint(0, max_start) if random_start and max_start > 0 else 0
        self.prev_action = 0
        return build_state_from_row(self.rows[self.i], self.monthly_target_rp)

    def step(self, action: int) -> tuple[np.ndarray, float, bool, dict[str, Any]]:
        row = self.rows[self.i]
        state = build_state_from_row(row, self.monthly_target_rp)
        prediksi_watt = float(state[self.IDX_PREDIKSI_WATT])
        gap_to_target = float(state[self.IDX_GAP_TO_TARGET])

        action = int(action)
        reduction = self.REDUCTION[action]
        watt_after = prediksi_watt * (1.0 - reduction)
        cost_after = (watt_after / 1000.0) * 24 * TARIF_PER_KWH
        gap_after = (cost_after * 30) - self.monthly_target_rp

        reward = -abs(gap_after) / max(self.monthly_target_rp, 1.0)
        if gap_to_target > 0 and action == 0:
            reward -= 0.20
        if gap_to_target <= 0 and action != 0:
            reward -= 0.10
        if action != self.prev_action:
            reward -= 0.03
        if row.suhu >= 30 and action == 1:
            reward -= 0.08
        if row.daya < 120 and action in {1, 2, 4}:
            reward -= 0.05
        if row.daya < 60 and action in {3, 5}:
            reward -= 0.03

        self.i += 1
        done = self.i >= len(self.rows) - 1
        next_state = build_state_from_row(self.rows[self.i if not done else -1], self.monthly_target_rp)
        self.prev_action = action
        return next_state, float(reward), done, {
            "prediksi_watt": prediksi_watt,
            "gap_before": gap_to_target,
            "gap_after": gap_after,
            "action_label": ACTION_LABELS[action],
        }


def initialize_firebase() -> None:
    if firebase_admin._apps:
        return
    firebase_key_json = os.getenv("FIREBASE_KEY")
    if firebase_key_json:
        cred = credentials.Certificate(json.loads(firebase_key_json))
    else:
        cred_path = Path(FIREBASE_CREDENTIALS)
        if not cred_path.exists():
            raise FileNotFoundError(
                "Firebase credentials tidak ditemukan. Isi FIREBASE_KEY atau FIREBASE_CREDENTIALS."
            )
        cred = credentials.Certificate(str(cred_path))
    firebase_admin.initialize_app(cred, {"databaseURL": DB_URL})


def load_training_rows() -> list[TrainRow]:
    raw_data = db.reference(HISTORY_NODE).get()
    if not raw_data:
        raise ValueError(f"Node '{HISTORY_NODE}' kosong atau tidak tersedia.")

    df = pd.DataFrame.from_dict(raw_data, orient="index")
    if "waktu" not in df.columns:
        raise ValueError("Kolom 'waktu' tidak ditemukan di data Firebase.")

    for col in ["Daya", "Suhu", "Arus", "Tegangan"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df["waktu"] = pd.to_datetime(df["waktu"], unit="ms", errors="coerce")
    df = df.dropna(subset=["waktu", "Daya", "Suhu", "Arus", "Tegangan"]).sort_values("waktu")
    if df.empty:
        raise ValueError("Data kosong setelah cleaning untuk retrain RL v2.")

    tz = pytz.timezone("Asia/Jakarta")
    rows: list[TrainRow] = []
    for _, row in df.tail(5000).iterrows():
        waktu = row["waktu"]
        if waktu.tzinfo is None:
            waktu = waktu.tz_localize("UTC").tz_convert(tz)
        else:
            waktu = waktu.tz_convert(tz)
        day_progress, days_remaining = month_progress(waktu)
        rows.append(
            TrainRow(
                daya=float(max(row["Daya"], 0.0)),
                suhu=float(np.clip(row["Suhu"], 10.0, 60.0)),
                arus=float(max(row["Arus"], 0.0)),
                tegangan=float(max(row["Tegangan"], 0.0)),
                jam=float(waktu.hour),
                day_of_week=float(waktu.weekday()),
                is_weekend=float(1 if waktu.weekday() >= 5 else 0),
                day_progress=day_progress,
                days_remaining=days_remaining,
            )
        )

    if len(rows) < 200:
        raise ValueError(f"Data untuk retrain RL v2 terlalu sedikit ({len(rows)}), butuh >= 200 sampel.")
    return rows


def resolve_monthly_target() -> float:
    prefs = db.reference("user_preferences").get() or {}
    target = float(prefs.get("monthly_budget_rp", DEFAULT_MONTHLY_TARGET_RP) or DEFAULT_MONTHLY_TARGET_RP)
    return max(target, 10_000.0)


def collect_batch(env: BudgetEnergyEnvV2, actor: ActorNetwork, batch_size: int) -> dict[str, np.ndarray]:
    states: list[np.ndarray] = []
    actions: list[int] = []
    rewards: list[float] = []
    old_log_probs: list[float] = []
    dones: list[float] = []
    next_states: list[np.ndarray] = []
    gaps_after: list[float] = []

    state = env.reset(random_start=True)
    while len(rewards) < batch_size:
        action, log_prob, _ = actor.get_action(state)
        next_state, reward, done, info = env.step(action)
        states.append(state)
        actions.append(action)
        rewards.append(reward)
        old_log_probs.append(float(log_prob.item()))
        dones.append(float(done))
        next_states.append(next_state)
        gaps_after.append(float(info["gap_after"]))
        state = env.reset(random_start=True) if done else next_state

    return {
        "states": np.asarray(states, dtype=np.float32),
        "actions": np.asarray(actions, dtype=np.int64),
        "rewards": np.asarray(rewards, dtype=np.float32),
        "old_log_probs": np.asarray(old_log_probs, dtype=np.float32),
        "dones": np.asarray(dones, dtype=np.float32),
        "next_states": np.asarray(next_states, dtype=np.float32),
        "gaps_after": np.asarray(gaps_after, dtype=np.float32),
    }


def compute_advantages_and_returns(
    rewards: np.ndarray,
    dones: np.ndarray,
    values: np.ndarray,
    next_values: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    advantages = np.zeros_like(rewards, dtype=np.float32)
    gae = 0.0
    for t in reversed(range(len(rewards))):
        not_done = 1.0 - dones[t]
        delta = rewards[t] + GAMMA * next_values[t] * not_done - values[t]
        gae = delta + GAMMA * LAMBDA_GAE * not_done * gae
        advantages[t] = gae
    returns = advantages + values
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    return advantages.astype(np.float32), returns.astype(np.float32)


def train(rows: list[TrainRow], monthly_target_rp: float) -> list[dict[str, Any]]:
    env = BudgetEnergyEnvV2(rows, monthly_target_rp)
    actor = ActorNetwork(OBS_DIM, N_ACTIONS, HIDDEN_DIM).to(DEVICE)
    critic = CriticNetwork(OBS_DIM, HIDDEN_DIM).to(DEVICE)
    optimizer_actor = optim.Adam(actor.parameters(), lr=LR_ACTOR)
    optimizer_critic = optim.Adam(critic.parameters(), lr=LR_CRITIC)

    if MODEL_OUTPUT_PATH.exists():
        checkpoint = torch.load(str(MODEL_OUTPUT_PATH), map_location=DEVICE)
        actor.load_state_dict(checkpoint["actor"])
        critic.load_state_dict(checkpoint["critic"])
        print(f"  → Melanjutkan model RL v2 dari {MODEL_OUTPUT_PATH}")

    timesteps_done = 0
    iteration = 0
    logs: list[dict[str, Any]] = []

    while timesteps_done < TOTAL_TIMESTEPS:
        iteration += 1
        batch = collect_batch(env, actor, BATCH_SIZE)
        timesteps_done += len(batch["rewards"])

        states_t = torch.as_tensor(batch["states"], dtype=torch.float32, device=DEVICE)
        actions_t = torch.as_tensor(batch["actions"], dtype=torch.long, device=DEVICE)
        old_lp_t = torch.as_tensor(batch["old_log_probs"], dtype=torch.float32, device=DEVICE)
        next_states_t = torch.as_tensor(batch["next_states"], dtype=torch.float32, device=DEVICE)

        with torch.no_grad():
            values_np = critic(states_t).cpu().numpy()
            next_values_np = critic(next_states_t).cpu().numpy()

        advantages_np, returns_np = compute_advantages_and_returns(
            batch["rewards"], batch["dones"], values_np, next_values_np
        )
        advantages_t = torch.as_tensor(advantages_np, dtype=torch.float32, device=DEVICE)
        returns_t = torch.as_tensor(returns_np, dtype=torch.float32, device=DEVICE)

        critic_losses: list[float] = []
        actor_losses: list[float] = []
        indices = np.arange(len(batch["rewards"]))
        for _ in range(N_EPOCHS):
            np.random.shuffle(indices)
            for start in range(0, len(indices), MINIBATCH_SIZE):
                mb = indices[start : start + MINIBATCH_SIZE]
                mb_t = torch.as_tensor(mb, dtype=torch.long, device=DEVICE)

                value_pred = critic(states_t[mb_t])
                critic_loss = nn.functional.mse_loss(value_pred, returns_t[mb_t])
                optimizer_critic.zero_grad()
                critic_loss.backward()
                nn.utils.clip_grad_norm_(critic.parameters(), max_norm=0.5)
                optimizer_critic.step()

                dist = actor(states_t[mb_t])
                new_lp = dist.log_prob(actions_t[mb_t])
                entropy = dist.entropy().mean()
                ratio = torch.exp(new_lp - old_lp_t[mb_t])
                surr1 = ratio * advantages_t[mb_t]
                surr2 = torch.clamp(ratio, 1 - CLIP_EPS, 1 + CLIP_EPS) * advantages_t[mb_t]
                actor_loss = -torch.min(surr1, surr2).mean() - ENTROPY_COEF * entropy
                optimizer_actor.zero_grad()
                actor_loss.backward()
                nn.utils.clip_grad_norm_(actor.parameters(), max_norm=0.5)
                optimizer_actor.step()

                critic_losses.append(float(critic_loss.item()))
                actor_losses.append(float(actor_loss.item()))

        log_entry = {
            "iter": iteration,
            "timesteps": timesteps_done,
            "reward": round(float(np.mean(batch["rewards"])), 4),
            "loss_critic": round(float(np.mean(critic_losses)), 4),
            "loss_actor": round(float(np.mean(actor_losses)), 4),
            "mean_abs_gap_after_rp": round(float(np.mean(np.abs(batch["gaps_after"]))), 2),
            "mean_prediksi_watt": round(float(np.mean(batch["states"][:, BudgetEnergyEnvV2.IDX_PREDIKSI_WATT])), 2),
        }
        logs.append(log_entry)
        print(
            f"Iter {iteration:4d} | Timesteps {timesteps_done:6d}/{TOTAL_TIMESTEPS} "
            f"| Reward {log_entry['reward']:+.4f} | Gap Rp {log_entry['mean_abs_gap_after_rp']:,.0f} "
            f"| C {log_entry['loss_critic']:.4f} | A {log_entry['loss_actor']:.4f}"
        )

    save_checkpoint(MODEL_OUTPUT_PATH, actor, critic, logs)
    write_training_log(MODEL_OUTPUT_PATH.parent / "training_log_rl_v2.json", logs)
    return logs


def main() -> None:
    print("=" * 72)
    print("  RL Budget Energy v2 — PPO-style Actor-Critic lima perangkat")
    print("=" * 72)
    print(f"  Device         : {DEVICE}")
    print(f"  Output         : {MODEL_OUTPUT_PATH}")
    print(f"  Observasi      : {OBS_DIM} fitur")
    print(f"  Aksi           : {N_ACTIONS} ({', '.join(ACTION_LABELS.values())})")
    print(f"  Total timesteps: {TOTAL_TIMESTEPS}")
    print("=" * 72)

    initialize_firebase()
    rows = load_training_rows()
    target_rp = resolve_monthly_target()
    print(f"Data training: {len(rows)} baris | Target bulanan: Rp {target_rp:,.0f}")
    train(rows, target_rp)
    print(f"✅ RL v2 berhasil disimpan ke {MODEL_OUTPUT_PATH}")


if __name__ == "__main__":
    main()
