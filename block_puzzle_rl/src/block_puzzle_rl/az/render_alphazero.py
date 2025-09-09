from __future__ import annotations

import argparse
from typing import Tuple
import random

import numpy as np
import pygame
import torch

from block_puzzle_rl.blockpuzzle import BlockPuzzleGame, GameConfig
from .net import AlphaZeroNet, Device
from .mcts import MCTS, unflatten_action


def _color_for_value(v: int) -> Tuple[int, int, int]:
    return (15, 15, 20) if v == 0 else (70, 200, 120)


def draw_board(screen: pygame.Surface, grid, cell_size: int, margin: int) -> None:
    h, w = grid.shape
    screen.fill((10, 10, 14))
    for y in range(h):
        for x in range(w):
            rect = pygame.Rect(margin + x * cell_size, margin + y * cell_size, cell_size - 1, cell_size - 1)
            pygame.draw.rect(screen, _color_for_value(int(grid[y, x])), rect)


def run(model_path: str | None, sims: int, channels: int, blocks: int, temp: float = 0.0, heuristic_prior_eps: float = 0.0, random_play: bool = False, board_size: int = 5, heuristic_only: bool = False) -> None:
    pygame.init()
    try:
        cfg = GameConfig(grid_size=int(board_size))
        game = BlockPuzzleGame(cfg)
        size = int(game.grid.size)
        k = int(game.config.pieces_per_set)
        mcts = None
        if not random_play:
            if heuristic_only:
                mcts = MCTS(net=None, n_simulations=int(sims), heuristic_prior_eps=1.0, use_heuristic_everywhere=True)
            else:
                if not model_path:
                    raise RuntimeError("--model is required unless --random or --heuristic_only is set")
                net = AlphaZeroNet(board_size=size, k=k, n_rot=4, channels=channels, blocks=blocks).to(Device)
                state = torch.load(model_path, map_location=Device)
                net.load_state_dict(state)
                net.eval()
                mcts = MCTS(net, n_simulations=int(sims), heuristic_prior_eps=float(heuristic_prior_eps))

        cell_size = 64
        margin = 20
        board_px_w = game.grid.size * cell_size
        board_px_h = game.grid.size * cell_size
        width = margin * 2 + board_px_w
        height = margin * 2 + board_px_h + 40
        screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption(f"Block Puzzle {game.grid.size}x{game.grid.size} - AlphaZero Render")
        font = pygame.font.SysFont(None, 28)
        clock = pygame.time.Clock()

        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.key == pygame.K_n:
                        game.reset()

            if not game.game_over:
                if random_play:
                    actions = game.get_valid_actions()
                    if actions:
                        p, x, y, r = random.choice(actions)
                        ok, gained, lines = game.place_piece(p, x, y, r)
                    else:
                        game.game_over = True
                else:
                    pi = mcts.run(game, add_root_noise=False)
                    if float(pi.sum()) > 0.0:
                        if temp > 1e-8:
                            probs = pi ** (1.0 / temp)
                            probs = probs / (np.sum(probs) + 1e-8)
                            idx = int(np.random.choice(len(probs), p=probs))
                        else:
                            idx = int(np.argmax(pi))
                        p, x, y, r = unflatten_action(idx, size=size, n_rot=4)
                        ok, gained, lines = game.place_piece(p, x, y, r)
                    else:
                        game.game_over = True

            draw_board(screen, game.grid.grid, cell_size, margin)
            status = f"Score: {game.score}  Steps: {game.step_count}  Pieces left: {len(game.current_pieces)}"
            img = font.render(status, True, (230, 230, 230))
            screen.blit(img, (margin, margin + board_px_h + 10))
            if game.game_over:
                over = font.render("Game Over - Press N to reset", True, (255, 120, 120))
                screen.blit(over, (margin, margin - 16))
            pygame.display.flip()
            clock.tick(6)
    finally:
        pygame.quit()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=str, default="", help="Path to model (.pt). Optional for --random or --heuristic_only")
    p.add_argument("--sims", type=int, default=64)
    p.add_argument("--channels", type=int, default=64)
    p.add_argument("--blocks", type=int, default=4)
    p.add_argument("--temp", type=float, default=0.0)
    p.add_argument("--heuristic_prior_eps", type=float, default=0.0)
    p.add_argument("--size", type=int, default=5)
    p.add_argument("--heuristic_only", action="store_true")
    p.add_argument("--random", action="store_true")
    args = p.parse_args()
    run(args.model or None, args.sims, args.channels, args.blocks, args.temp, args.heuristic_prior_eps, args.random, args.size, args.heuristic_only)


if __name__ == "__main__":  # pragma: no cover
    main()


