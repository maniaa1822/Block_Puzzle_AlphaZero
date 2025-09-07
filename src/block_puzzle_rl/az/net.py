from __future__ import annotations

from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from block_puzzle_rl.blockpuzzle import BlockPuzzleGame, PieceType


Device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ResidualBlock(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.c1 = nn.Conv2d(ch, ch, kernel_size=3, padding=1)
        self.c2 = nn.Conv2d(ch, ch, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(ch)
        self.bn2 = nn.BatchNorm2d(ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D401
        h = F.relu(self.bn1(self.c1(x)))
        h = self.bn2(self.c2(h))
        return F.relu(x + h)


def encode_state_planes(game: BlockPuzzleGame, k: int | None = None) -> torch.Tensor:
    """
    Build input planes:
    - 1 grid plane (H,W) with 0/1
    - k * 7 piece-type planes (broadcast, one-hot per slot)
    - 1 plane = pieces_remaining / k
    """
    size = int(game.grid.size)
    k = int(k if k is not None else game.config.pieces_per_set)
    grid = game.grid.grid.astype(np.float32)  # (H,W)

    # Slot-wise one-hot for available pieces (7 types). Missing slots -> zeros.
    types = [p.piece_type.value for p in game.current_pieces]
    planes = [grid[None, :, :]]  # (1,H,W)

    num_types = len(PieceType)
    for slot in range(k):
        onehot = np.zeros((num_types,), dtype=np.float32)
        if slot < len(types):
            t = int(types[slot])
            if 0 <= t < num_types:
                onehot[t] = 1.0
        planes.append(np.broadcast_to(onehot[:, None, None], (num_types, size, size)))

    prem = np.full((1, size, size), fill_value=(len(types) / max(1, k)), dtype=np.float32)
    planes.append(prem)

    x = np.concatenate(planes, axis=0)  # (C,H,W)
    return torch.from_numpy(x)


class AlphaZeroNet(nn.Module):
    """
    Small CNN trunk + policy/value heads.
    Policy head outputs logits of size N = k * size * size * 4 (piece,y,x,r).
    """

    def __init__(self, board_size: int, k: int, n_rot: int = 4, channels: int = 64, blocks: int = 4):
        super().__init__()
        self.size = int(board_size)
        self.k = int(k)
        self.n_rot = int(n_rot)

        in_channels = 1 + self.k * len(PieceType) + 1
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )
        self.res = nn.Sequential(*[ResidualBlock(channels) for _ in range(blocks)])

        # Policy head
        self.p_conv = nn.Conv2d(channels, 2, kernel_size=1)
        self.p_bn = nn.BatchNorm2d(2)
        flat_dim = 2 * self.size * self.size
        self.N_out = int(self.k * self.size * self.size * self.n_rot)
        self.p_fc = nn.Linear(flat_dim, self.N_out)

        # Value head
        self.v_conv = nn.Conv2d(channels, 1, kernel_size=1)
        self.v_bn = nn.BatchNorm2d(1)
        self.v_fc1 = nn.Linear(self.size * self.size, 64)
        self.v_fc2 = nn.Linear(64, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.res(self.stem(x))
        # policy
        p = F.relu(self.p_bn(self.p_conv(h)))
        p = torch.flatten(p, 1)
        policy_logits = self.p_fc(p)  # (B,N)
        # value
        v = F.relu(self.v_bn(self.v_conv(h)))
        v = torch.flatten(v, 1)
        v = F.relu(self.v_fc1(v))
        value = torch.tanh(self.v_fc2(v)).squeeze(-1)
        return policy_logits, value

    @torch.no_grad()
    def predict(self, game: BlockPuzzleGame) -> tuple[np.ndarray, float]:
        x = encode_state_planes(game, self.k).unsqueeze(0).to(Device)  # (1,C,H,W)
        self.eval()
        logits, v = self(x)
        return logits.squeeze(0).cpu().numpy(), float(v.item())



