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
import torch.multiprocessing as mp

from block_puzzle_rl.blockpuzzle import BlockPuzzleGame, GameConfig
from block_puzzle_rl.blockpuzzle.logic import ValueNormalizer
from .net import AlphaZeroNet, encode_state_planes, Device
from .mcts import MCTS, unflatten_action


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


def play_one_game(mcts: MCTS, temp: float = 1.0, gamma: float = 0.99, value_normalizer: ValueNormalizer | None = None) -> Tuple[List[np.ndarray], List[np.ndarray], List[float], dict]:
    cfg = GameConfig(grid_size=5)  # enforce 5x5
    game = BlockPuzzleGame(cfg)
    # Share a global value normalizer across episodes (if provided)
    if value_normalizer is not None:
        game.value_normalizer = value_normalizer
    size = int(game.grid.size)
    k = int(game.config.pieces_per_set)
    x_list: List[np.ndarray] = []
    pi_list: List[np.ndarray] = []
    r_list: List[float] = []

    while not game.game_over:
        pi = mcts.run(deepcopy(game), add_root_noise=True)
        # Temperature sampling
        if temp > 1e-8:
            probs = pi ** (1.0 / temp)
            probs = probs / (np.sum(probs) + 1e-8)
        else:
            probs = np.zeros_like(pi)
            if pi.sum() > 0:
                probs[np.argmax(pi)] = 1.0
        # Save training pair (state, pi)
        x = encode_state_planes(game, k).numpy()  # (C,H,W)
        x_list.append(x)
        pi_list.append(probs.astype(np.float32))

        # Sample and play action
        flat_idx = int(np.random.choice(len(probs), p=probs if probs.sum() > 0 else None))
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

    # Use terminal normalized value target for all states in the episode
    stats = game.get_game_stats()
    z_terminal = game.get_value_target()
    zs: List[float] = [float(z_terminal)] * len(x_list)

    # Update global baseline AFTER computing targets
    if value_normalizer is not None:
        value_normalizer.update(float(stats.get("final_score", 0.0)))

    assert len(x_list) == len(pi_list) == len(zs), (len(x_list), len(pi_list), len(zs))
    return x_list, pi_list, zs, stats


def _state_dict_cpu(sd: dict) -> dict:
    """Move a state_dict to CPU tensors for cheaper pickling between processes."""
    return {k: v.detach().cpu() for k, v in sd.items()}


def self_play_worker(worker_id: int, state_dict: dict, sims: int, temp: float, episodes: int, heuristic_prior_eps: float, channels: int, blocks: int, gamma: float) -> List[Tuple[List[np.ndarray], List[np.ndarray], float, dict]]:
    """Run self-play episodes with a given model snapshot.

    Returns a list of tuples (xs, pis, final_score, stats) per episode.
    Value targets are computed in the main process using a shared normalizer.
    """
    size = 5
    k = 3
    net = AlphaZeroNet(board_size=size, k=k, n_rot=4, channels=channels, blocks=blocks)
    net.load_state_dict(state_dict)
    net.eval()
    # Force CPU in workers to avoid GPU contention
    net.to(torch.device("cpu"))
    mcts = MCTS(net, n_simulations=int(sims), gamma=float(gamma), heuristic_prior_eps=float(heuristic_prior_eps))

    results: List[Tuple[List[np.ndarray], List[np.ndarray], float, dict]] = []
    for _ in range(int(episodes)):
        xs, pis, _zs_unused, stats = play_one_game(mcts, temp=temp, gamma=gamma, value_normalizer=None)
        results.append((xs, pis, float(stats.get("final_score", 0.0)), stats))
    return results


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

            # Value loss: MSE to terminal tanh-normalized target in [-1, 1]
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
    p.add_argument("--channels", type=int, default=64)
    p.add_argument("--blocks", type=int, default=4)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--epochs", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--save_path", type=str, default="./models/az_blockpuzzle_5x5.pt")
    p.add_argument("--logdir", type=str, default="", help="TensorBoard log dir (optional)")
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--heuristic_prior_eps", type=float, default=0.0)
    p.add_argument("--workers", type=int, default=1, help="Number of self-play worker processes")
    p.add_argument("--episodes_per_update", type=int, default=10, help="Episodes per training update across workers")
    args = p.parse_args()

    size = 5
    k = 3
    net = AlphaZeroNet(board_size=size, k=k, n_rot=4, channels=args.channels, blocks=args.blocks).to(Device)
    # Shared value normalizer across episodes
    value_normalizer = ValueNormalizer()

    buf = ReplayBuffer(capacity=50_000)

    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)

    writer = SummaryWriter(args.logdir) if args.logdir else None

    if int(args.workers) <= 1:
        mcts = MCTS(net, n_simulations=args.sims, gamma=args.gamma, heuristic_prior_eps=float(args.heuristic_prior_eps))
        for ep in trange(args.episodes, desc="Self-play"):
            xs, pis, zs, stats = play_one_game(mcts, temp=args.temp, gamma=args.gamma, value_normalizer=value_normalizer)
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
    else:
        # Multiprocessing self-play loop
        try:
            mp.set_start_method("spawn", force=True)
        except RuntimeError:
            pass
        total_eps = int(args.episodes)
        done = 0
        update_idx = 0
        while done < total_eps:
            remaining = total_eps - done
            n_workers = min(int(args.workers), max(1, (remaining + args.episodes_per_update - 1) // args.episodes_per_update))
            sd_cpu = _state_dict_cpu(net.state_dict())
            tasks = []
            issued = 0
            for w in range(n_workers):
                ep_w = min(int(args.episodes_per_update), remaining - issued)
                if ep_w <= 0:
                    break
                tasks.append((w, sd_cpu, int(args.sims), float(args.temp), int(ep_w), float(args.heuristic_prior_eps), int(args.channels), int(args.blocks), float(args.gamma)))
                issued += ep_w
            with mp.Pool(processes=n_workers) as pool:
                results = pool.starmap(self_play_worker, tasks)

            batch_scores: List[float] = []
            batch_steps: List[int] = []
            for r in results:
                for xs, pis, final_score, stats in r:
                    z = value_normalizer.value_target(final_score)
                    for x, pi in zip(xs, pis):
                        buf.push(x, pi, z)
                    batch_scores.append(float(final_score))
                    batch_steps.append(int(stats.get("steps_taken", 0)))
                    done += 1

            # Update normalizer after targets computed
            for s in batch_scores:
                value_normalizer.update(float(s))

            if len(buf) >= args.batch_size:
                pol_loss, val_loss = train(net, buf, epochs=args.epochs, batch_size=args.batch_size, lr=args.lr)
                if args.verbose:
                    print(f"Update {update_idx}: episodes={done} pol_loss={pol_loss:.4f} val_loss={val_loss:.4f} buffer={len(buf)}")
                if writer is not None:
                    writer.add_scalar("train/policy_loss", pol_loss, update_idx)
                    writer.add_scalar("train/value_loss", val_loss, update_idx)

            if writer is not None and batch_scores:
                writer.add_scalar("episode/final_score_mean", float(np.mean(batch_scores)), update_idx)
                writer.add_scalar("episode/steps_mean", float(np.mean(batch_steps)), update_idx)
                writer.add_scalar("buffer/size", len(buf), update_idx)
            update_idx += 1

    torch.save(net.state_dict(), args.save_path)
    print(f"Saved AlphaZeroNet to {args.save_path}")
    if writer is not None:
        writer.close()


if __name__ == "__main__":  # pragma: no cover
    main()



