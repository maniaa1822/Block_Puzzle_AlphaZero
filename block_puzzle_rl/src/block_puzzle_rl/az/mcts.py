from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
from copy import deepcopy
import math
import numpy as np

from block_puzzle_rl.blockpuzzle import BlockPuzzleGame, GameAnalytics
from .scoring import ScoreNormalizer
from .net import AlphaZeroNet, Device


def flatten_action(piece: int, x: int, y: int, r: int, size: int, n_rot: int) -> int:
    return (((piece * size + y) * size + x) * n_rot) + r


def unflatten_action(idx: int, size: int, n_rot: int) -> tuple[int, int, int, int]:
    r = idx % n_rot
    idx //= n_rot
    x = idx % size
    idx //= size
    y = idx % size
    piece = idx // size
    return int(piece), int(x), int(y), int(r)


def legal_action_indices(game: BlockPuzzleGame, size: int, k: int, n_rot: int) -> tuple[np.ndarray, list[tuple[int, int, int, int]]]:
    actions = game.get_valid_actions()
    idxs: list[int] = []
    tuples: list[tuple[int, int, int, int]] = []
    for (p, x, y, r) in actions:
        if 0 <= p < k and 0 <= x < size and 0 <= y < size and 0 <= r < n_rot:
            idxs.append(flatten_action(p, x, y, r, size, n_rot))
            tuples.append((p, x, y, r))
    if len(idxs) == 0:
        return np.zeros((0,), dtype=np.int64), []
    arr = np.array(sorted(idxs), dtype=np.int64)
    tuples_sorted = [unflatten_action(i, size, n_rot) for i in arr]
    return arr, tuples_sorted


@dataclass
class Node:
    game: BlockPuzzleGame
    prior: Dict[int, float]
    N: Dict[int, int]
    W: Dict[int, float]
    Q: Dict[int, float]
    children: Dict[int, "Node"]
    legal_idxs: np.ndarray
    terminal: bool

    @classmethod
    def create_root(cls, game: BlockPuzzleGame, priors: Dict[int, float], legal_idxs: np.ndarray, terminal: bool) -> "Node":
        return cls(game=game, prior=priors, N={}, W={}, Q={}, children={}, legal_idxs=legal_idxs, terminal=terminal)

    def best_ucb(self, c_puct: float) -> int:
        total_N = sum(self.N.get(a, 0) for a in self.legal_idxs.tolist()) + 1
        best_score = -1e9
        best_a: Optional[int] = None
        for a in self.legal_idxs.tolist():
            q = self.Q.get(a, 0.0)
            p = self.prior.get(a, 0.0)
            n = self.N.get(a, 0)
            u = c_puct * p * math.sqrt(total_N) / (1 + n)
            s = q + u
            if s > best_score:
                best_score = s
                best_a = a
        assert best_a is not None
        return best_a


class MCTS:
    def __init__(
        self,
        net: Optional[AlphaZeroNet],
        c_puct: float = 1.5,
        dirichlet_alpha: float = 0.3,
        dirichlet_eps: float = 0.25,
        n_simulations: int = 200,
        heuristic_prior_eps: float = 0.0,
        heuristic_weights: Optional[Dict[str, float]] = None,
        score_norm: Optional[ScoreNormalizer] = None,
        use_heuristic_everywhere: bool = False,
    ):
        self.net = net.to(Device) if net is not None else None
        self.c_puct = float(c_puct)
        self.dirichlet_alpha = float(dirichlet_alpha)
        self.dirichlet_eps = float(dirichlet_eps)
        self.n_simulations = int(n_simulations)
        # Heuristic prior mixing at root (0 disables)
        self.heuristic_prior_eps = float(heuristic_prior_eps)
        self.use_heuristic_everywhere = bool(use_heuristic_everywhere)
        # Default weights inspired by env shaping
        self.heuristic_weights: Dict[str, float] = {
            "cells": 0.05,
            "lines": 10.0,
            "holes": 0.1,
            "bumpiness": 0.01,
            "height": 0.02,
            "almost": 0.5,
        }
        if heuristic_weights:
            self.heuristic_weights.update({k: float(v) for k, v in heuristic_weights.items()})
        # Score normalizer for terminal value targets
        self.score_norm: ScoreNormalizer = score_norm or ScoreNormalizer()

    def _masked_softmax(self, logits: np.ndarray, legal_idxs: np.ndarray, N_out: int) -> Dict[int, float]:
        masked = np.full((N_out,), -1e9, dtype=np.float32)
        masked[legal_idxs] = logits[legal_idxs].astype(np.float32)
        m = np.max(masked)
        exps = np.exp(masked - m)
        probs = exps / np.sum(exps)
        return {int(i): float(probs[int(i)]) for i in legal_idxs.tolist()}

    def _expand(self, node: Node) -> tuple[Dict[int, float], float]:
        size = int(node.game.grid.size)
        k = int(node.game.config.pieces_per_set)
        n_rot = 4 if self.net is None else int(self.net.n_rot)
        N_out = k * size * size * n_rot if self.net is None else int(self.net.N_out)
        if self.use_heuristic_everywhere or self.net is None:
            priors = self._heuristic_prior(node.game, node.legal_idxs, size, n_rot)
            return priors, 0.0
        logits, v = self.net.predict(node.game)
        priors = self._masked_softmax(logits, node.legal_idxs, N_out)
        return priors, v

    def _add_dirichlet(self, priors: Dict[int, float], legal_idxs: np.ndarray) -> Dict[int, float]:
        if legal_idxs.size == 0:
            return priors
        noise = np.random.dirichlet([self.dirichlet_alpha] * legal_idxs.size)
        mixed: Dict[int, float] = {}
        for i, a in enumerate(legal_idxs.tolist()):
            p = priors.get(a, 0.0)
            mixed[a] = float(self.dirichlet_eps * noise[i] + (1.0 - self.dirichlet_eps) * p)
        s = sum(mixed.values()) or 1.0
        for a in mixed:
            mixed[a] /= s
        return mixed

    def _heuristic_prior(self, game: BlockPuzzleGame, legal_idxs: np.ndarray, size: int, n_rot: int) -> Dict[int, float]:
        if legal_idxs.size == 0:
            return {}
        scores: Dict[int, float] = {}
        w = self.heuristic_weights
        for a in legal_idxs.tolist():
            p, x, y, r = unflatten_action(a, size, n_rot)
            # Evaluate placement quality analytically (no rollout)
            piece = game.current_pieces[p]
            shape = piece.get_shape(r)
            q = GameAnalytics.evaluate_placement_quality(game.grid, shape, x, y)
            if not q.get("valid", False):
                s = -1e9
            else:
                s = (
                    w["cells"] * float(q.get("cells_placed", 0))
                    + w["lines"] * float(q.get("lines_cleared", 0))
                    - w["holes"] * float(max(0, q.get("holes_created", 0)))
                    - w["bumpiness"] * float(max(0, q.get("bumpiness_change", 0)))
                    - w["height"] * float(max(0, q.get("height_increase", 0)))
                    + w["almost"] * float(q.get("almost_complete_lines_after", 0))
                )
            scores[a] = float(s)
        # Convert scores -> distribution via softmax; guard for -inf
        arr_idx = legal_idxs.tolist()
        vec = np.array([scores[i] for i in arr_idx], dtype=np.float32)
        m = np.max(vec)
        exps = np.exp(vec - m)
        probs = exps / (np.sum(exps) + 1e-8)
        return {int(i): float(p) for i, p in zip(arr_idx, probs.tolist())}

    def run(self, root_game: BlockPuzzleGame, add_root_noise: bool = True) -> np.ndarray:
        size = int(root_game.grid.size)
        k = int(root_game.config.pieces_per_set)
        n_rot = int(self.net.n_rot) if self.net is not None else 4
        N_out = int(self.net.N_out) if self.net is not None else int(k * size * size * n_rot)

        legal_idxs, _ = legal_action_indices(root_game, size, k, n_rot)
        terminal = bool(root_game.game_over or legal_idxs.size == 0)
        priors, _ = self._expand(Node.create_root(root_game, {}, legal_idxs, terminal))
        # Mix in heuristic prior at root if enabled and not already using heuristic everywhere
        if (self.heuristic_prior_eps > 0.0) and (not self.use_heuristic_everywhere):
            h_prior = self._heuristic_prior(root_game, legal_idxs, size, n_rot)
            mixed: Dict[int, float] = {}
            for a in legal_idxs.tolist():
                p_net = priors.get(a, 0.0)
                p_h = h_prior.get(a, 0.0)
                mixed[a] = float((1.0 - self.heuristic_prior_eps) * p_net + self.heuristic_prior_eps * p_h)
            s = sum(mixed.values()) or 1.0
            for a in mixed:
                mixed[a] /= s
            priors = mixed
        if add_root_noise:
            priors = self._add_dirichlet(priors, legal_idxs)
        root = Node.create_root(root_game, priors, legal_idxs, terminal)

        for _ in range(self.n_simulations):
            node = root
            path: List[tuple[Node, int]] = []
            sim_game = deepcopy(root_game)

            # Selection
            while not node.terminal:
                a = node.best_ucb(self.c_puct)
                p, x, y, r = unflatten_action(a, size, n_rot)
                ok, gained, _ = sim_game.place_piece(p, x, y, r)
                if not ok:
                    node.N[a] = node.N.get(a, 0) + 1
                    node.W[a] = node.W.get(a, 0.0)
                    node.Q[a] = node.W[a] / node.N[a]
                    break
                path.append((node, a))

                # Expand new child if needed
                if a not in node.children:
                    child_legal, _ = legal_action_indices(sim_game, size, k, n_rot)
                    child_terminal = bool(sim_game.game_over or child_legal.size == 0)
                    child_priors, v = self._expand(Node.create_root(sim_game, {}, child_legal, child_terminal))
                    child = Node.create_root(deepcopy(sim_game), child_priors, child_legal, child_terminal)
                    node.children[a] = child
                    # Backup: r + v for non-terminal; normalized final score for terminal
                    if child_terminal:
                        G = float(self.score_norm.normalize(sim_game.score))
                        for parent, act in reversed(path):
                            parent.N[act] = parent.N.get(act, 0) + 1
                            parent.W[act] = parent.W.get(act, 0.0) + G
                            parent.Q[act] = parent.W[act] / parent.N[act]
                    else:
                        # r + v backup: implicit r is the incremental score gained at each step
                        # We can approximate by using the network value at child and not discount (single-player episodic)
                        G = float(v)
                        for parent, act in reversed(path):
                            parent.N[act] = parent.N.get(act, 0) + 1
                            parent.W[act] = parent.W.get(act, 0.0) + G
                            parent.Q[act] = parent.W[act] / parent.N[act]
                    break
                else:
                    node = node.children[a]
                    if node.terminal:
                        # Terminal reached; backup normalized final score from sim_game
                        G = float(self.score_norm.normalize(sim_game.score))
                        for parent, act in reversed(path):
                            parent.N[act] = parent.N.get(act, 0) + 1
                            parent.W[act] = parent.W.get(act, 0.0) + G
                            parent.Q[act] = parent.W[act] / parent.N[act]
                        break

        pi = np.zeros((N_out,), dtype=np.float32)
        total = 0
        for a in root.legal_idxs.tolist():
            c = root.N.get(a, 0)
            pi[a] = float(c)
            total += c
        if total > 0:
            pi /= float(total)
        return pi



