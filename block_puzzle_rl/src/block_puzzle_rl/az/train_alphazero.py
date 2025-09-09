from __future__ import annotations

import argparse
from typing import List, Tuple
import os
import random
from copy import deepcopy

import numpy as np
import torch
import torch.nn.functional as F
from torch import optim
from tqdm import trange
from torch.utils.tensorboard import SummaryWriter

from block_puzzle_rl.blockpuzzle import BlockPuzzleGame, GameConfig
from .net import AlphaZeroNet, encode_state_planes, Device
from .mcts import MCTS, unflatten_action
from .scoring import ScoreNormalizer


class ReplayBuffer:
    def __init__(self, capacity: int = 100_000):
        self.capacity = int(capacity)
        self.data: List[Tuple[np.ndarray, np.ndarray, float]] = []

    def push(self, x: np.ndarray, pi: np.ndarray, z: float) -> None:
        if len(self.data) >= self.capacity:
            self.data.pop(0)
        self.data.append((x, pi, float(z)))

    def sample(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch = random.sample(self.data, batch_size)
        xs = torch.from_numpy(np.stack([b[0] for b in batch], axis=0)).to(Device)
        pis = torch.from_numpy(np.stack([b[1] for b in batch], axis=0)).to(Device)
        zs = torch.tensor([b[2] for b in batch], dtype=torch.float32, device=Device)
        return xs, pis, zs

    def __len__(self) -> int:
        return len(self.data)


def play_one_game(mcts: MCTS, temp: float = 1.0) -> Tuple[List[np.ndarray], List[np.ndarray], List[float], dict]:
    cfg = GameConfig(grid_size=5)  # enforce 5x5
    game = BlockPuzzleGame(cfg)
    size = int(game.grid.size)
    k = int(game.config.pieces_per_set)
    x_list: List[np.ndarray] = []
    pi_list: List[np.ndarray] = []
    r_list: List[float] = []

    while not game.game_over:
        pi = mcts.run(deepcopy(game), add_root_noise=True)
        # Temperature sampling
        if temp > 1e-8:
            probs = (pi ** (1.0 / temp)).astype(np.float64)
            probs = np.maximum(probs, 0.0)
            s = float(np.sum(probs))
            if not np.isfinite(s) or s <= 0.0:
                probs = None
            else:
                probs /= s
                # ensure exact sum and non-negativity
                idx = int(np.argmax(probs))
                probs[idx] += 1.0 - float(np.sum(probs))
        else:
            probs = np.zeros_like(pi, dtype=np.float64)
            if pi.sum() > 0:
                probs[int(np.argmax(pi))] = 1.0
        # Save training pair (state, pi)
        x = encode_state_planes(game, k).numpy()  # (C,H,W)
        x_list.append(x)
        pi_list.append(probs.astype(np.float32))

        # Sample and play action
        if probs is None:
            flat_idx = int(np.random.choice(len(pi)))
        else:
            flat_idx = int(np.random.choice(len(probs), p=probs))
        piece, x_pos, y_pos, r = unflatten_action(flat_idx, size, n_rot=4)
        ok, gained, _ = game.place_piece(piece, x_pos, y_pos, r)
        if not ok:
            # Fallback: if somehow invalid, end the game
            game.game_over = True
            r_list.append(0.0)
            break
        r_list.append(float(gained))

        if game.step_count >= 500:  # safety cap for tiny boards
            break

        if game.game_over:
            break

    # Terminal normalized score as value target for all states in this episode
    final_score = float(game.score)
    z = float(mcts.score_norm.normalize(final_score))
    # Update running stats after producing the target (avoid leakage within episode)
    mcts.score_norm.update(final_score)
    zs: List[float] = [z] * len(x_list)

    assert len(x_list) == len(pi_list) == len(zs), (len(x_list), len(pi_list), len(zs))
    return x_list, pi_list, zs, game.get_game_stats()


def train(net: AlphaZeroNet, buffer: ReplayBuffer, epochs: int, batch_size: int, lr: float, l2: float = 1e-4) -> tuple[float, float]:
    net.train().to(Device)
    opt = optim.Adam(net.parameters(), lr=lr, weight_decay=l2)
    steps_per_epoch = max(1, len(buffer) // batch_size)
    avg_pol: float = 0.0
    avg_val: float = 0.0
    total_steps = 0
    for _ in range(epochs):
        for _ in range(steps_per_epoch):
            xs, pis, zs = buffer.sample(batch_size)
            logits, v = net(xs)

            # Policy loss: cross-entropy with target distribution pi
            logp = torch.log_softmax(logits, dim=1)
            pol_loss = -(pis * logp).sum(dim=1).mean()

            # Value loss: MSE to tanh-normalized terminal score
            val_loss = torch.mean((v - zs) ** 2)

            loss = pol_loss + val_loss

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            avg_pol += float(pol_loss.item())
            avg_val += float(val_loss.item())
            total_steps += 1
    if total_steps > 0:
        avg_pol /= float(total_steps)
        avg_val /= float(total_steps)
    return avg_pol, avg_val


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", type=int, default=20)
    p.add_argument("--sims", type=int, default=128, help="MCTS simulations per move")
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--temp", type=float, default=1.0)
    p.add_argument("--temp_min", type=float, default=0.1)
    p.add_argument("--temp_decay", type=float, default=0.999)
    p.add_argument("--heuristic_eps_start", type=float, default=0.5)
    p.add_argument("--heuristic_eps_end", type=float, default=0.1)
    p.add_argument("--heuristic_eps_decay", type=float, default=0.999)
    p.add_argument("--channels", type=int, default=64)
    p.add_argument("--blocks", type=int, default=4)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--epochs", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--save_path", type=str, default="./models/az_blockpuzzle_5x5.pt")
    p.add_argument("--ckpt_every", type=int, default=100, help="Episodes between checkpoints (0 to disable)")
    p.add_argument("--logdir", type=str, default="", help="TensorBoard log dir (optional)")
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--heuristic_prior_eps", type=float, default=0.0)
    # Fixed mapping flags for ScoreNormalizer
    p.add_argument("--value_fixed_min", type=float, default=None, help="If set with --value_fixed_max, linearly map [min,max] -> [-1,1]")
    p.add_argument("--value_fixed_max", type=float, default=None, help="If set with --value_fixed_min, linearly map [min,max] -> [-1,1]")
    args = p.parse_args()

    size = 5
    k = 3
    net = AlphaZeroNet(board_size=size, k=k, n_rot=4, channels=args.channels, blocks=args.blocks).to(Device)
    # Initialize with starting heuristic mixing; we'll anneal per episode
    mcts = MCTS(
        net,
        n_simulations=args.sims,
        heuristic_prior_eps=float(args.heuristic_eps_start if args.heuristic_prior_eps == 0.0 else args.heuristic_prior_eps),
        score_norm=ScoreNormalizer(fixed_min=args.value_fixed_min, fixed_max=args.value_fixed_max),
    )

    buf = ReplayBuffer(capacity=50_000)

    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)

    writer = SummaryWriter(args.logdir) if args.logdir else None

    temp = float(args.temp)
    heur_eps = float(mcts.heuristic_prior_eps)
    for ep in trange(args.episodes, desc="Self-play"):
        xs, pis, zs, stats = play_one_game(mcts, temp=temp)
        for x, pi, z in zip(xs, pis, zs):
            buf.push(x, pi, z)
        if args.verbose:
            print(f"Episode {ep}: steps={stats['steps_taken']} score={stats['final_score']} lines={stats['lines_cleared']} buffer={len(buf)}")
        if writer is not None:
            writer.add_scalar("episode/final_score", stats["final_score"], ep)
            writer.add_scalar("episode/steps", stats["steps_taken"], ep)
            writer.add_scalar("episode/lines_cleared", stats["lines_cleared"], ep)
            writer.add_scalar("buffer/size", len(buf), ep)
        if len(buf) >= args.batch_size:
            pol_loss, val_loss = train(net, buf, epochs=args.epochs, batch_size=args.batch_size, lr=args.lr)
            if args.verbose:
                print(f"Train: pol_loss={pol_loss:.4f} val_loss={val_loss:.4f}")
            if writer is not None:
                writer.add_scalar("train/policy_loss", pol_loss, ep)
                writer.add_scalar("train/value_loss", val_loss, ep)
        # Anneal temperature and heuristic eps
        temp = max(args.temp_min, temp * args.temp_decay)
        if args.heuristic_prior_eps == 0.0:
            heur_eps = max(args.heuristic_eps_end, heur_eps * args.heuristic_eps_decay)
            mcts.heuristic_prior_eps = heur_eps
        # Periodic checkpointing
        if args.ckpt_every > 0 and (ep + 1) % int(args.ckpt_every) == 0:
            root, ext = os.path.splitext(args.save_path)
            ckpt_path = f"{root}_ep{ep+1}{ext}"
            torch.save(net.state_dict(), ckpt_path)
            if args.verbose:
                print(f"Checkpoint saved: {ckpt_path}")

    torch.save(net.state_dict(), args.save_path)
    print(f"Saved AlphaZeroNet to {args.save_path}")
    if writer is not None:
        writer.close()


if __name__ == "__main__":  # pragma: no cover
    main()



