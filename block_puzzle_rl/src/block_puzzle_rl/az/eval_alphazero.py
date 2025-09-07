from __future__ import annotations

import argparse
from typing import Dict, Any, List

import numpy as np
import torch

from block_puzzle_rl.blockpuzzle import BlockPuzzleGame, GameConfig
from .net import AlphaZeroNet, Device
from .mcts import MCTS, unflatten_action


def play_episode(net: AlphaZeroNet, sims: int, temp: float = 0.0, step_cap: int = 500, verbose: bool = False, heuristic_prior_eps: float = 0.0) -> Dict[str, Any]:
    cfg = GameConfig(grid_size=5)
    game = BlockPuzzleGame(cfg)
    mcts = MCTS(net, n_simulations=int(sims), heuristic_prior_eps=float(heuristic_prior_eps))
    size = int(game.grid.size)

    while not game.game_over and game.step_count < step_cap:
        pi = mcts.run(game, add_root_noise=False)
        if float(pi.sum()) <= 0.0:
            break
        if temp > 1e-8:
            probs = pi ** (1.0 / temp)
            probs = probs / (np.sum(probs) + 1e-8)
            idx = int(np.random.choice(len(probs), p=probs))
        else:
            idx = int(np.argmax(pi))
        p, x, y, r = unflatten_action(idx, size=size, n_rot=4)
        ok, gained, lines = game.place_piece(p, x, y, r)
        if verbose:
            print(f"step={game.step_count} action=(p={p},x={x},y={y},r={r}) gained={gained} lines={lines} score={game.score}")
        if not ok:
            break

    return game.get_game_stats()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="Path to saved AlphaZeroNet .pt file")
    parser.add_argument("--sims", type=int, default=128, help="MCTS simulations per move")
    parser.add_argument("--episodes", type=int, default=1, help="Number of evaluation episodes")
    parser.add_argument("--temp", type=float, default=0.0, help="Action temperature; 0 for greedy")
    parser.add_argument("--channels", type=int, default=64, help="CNN channels (must match checkpoint)")
    parser.add_argument("--blocks", type=int, default=4, help="Residual blocks (must match checkpoint)")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--heuristic_prior_eps", type=float, default=0.0)
    args = parser.parse_args()

    size = 5
    k = 3
    net = AlphaZeroNet(board_size=size, k=k, n_rot=4, channels=args.channels, blocks=args.blocks).to(Device)
    state = torch.load(args.model, map_location=Device)
    net.load_state_dict(state)
    net.eval()

    results: List[Dict[str, Any]] = []
    for ep in range(int(args.episodes)):
        stats = play_episode(net, sims=args.sims, temp=args.temp, verbose=args.verbose, heuristic_prior_eps=args.heuristic_prior_eps)
        results.append(stats)
        if args.verbose:
            print(f"episode {ep}: score={stats['final_score']} steps={stats['steps_taken']} lines={stats['lines_cleared']}")

    if not results:
        print("No results.")
        return

    avg_score = sum(r["final_score"] for r in results) / len(results)
    avg_steps = sum(r["steps_taken"] for r in results) / len(results)
    avg_lines = sum(r["lines_cleared"] for r in results) / len(results)
    print(f"Episodes: {len(results)}")
    print(f"Avg score: {avg_score:.2f}")
    print(f"Avg steps: {avg_steps:.2f}")
    print(f"Avg lines: {avg_lines:.2f}")
    print("Last episode:", results[-1])


if __name__ == "__main__":  # pragma: no cover
    main()



