# Block Puzzle RL

Project scaffold for a Gymnasium environment and RL agents for Block Puzzle.

## Quickstart

- Create venv and install deps:

	cd block_puzzle_rl
	uv sync

- Run a quick import check:

	uv run python -c "import gymnasium, numpy, pygame; print('OK')"

## Layout

- src/block_puzzle_rl: package code
- pyproject.toml: project metadata

## Run commands

- Play Tetris variant (keyboard):

\tuv run block-puzzle-play

- Play Block Puzzle (9x9) human UI:

\tuv run block-puzzle-sudoku-play

- Random agent (env sanity):

\tuv run block-puzzle-random

- PPO training (vanilla):

\tuv run block-puzzle-train-ppo --algo ppo --timesteps 200000 --n_envs 4

- Maskable PPO (requires sb3-contrib):

\tuv run block-puzzle-train-ppo --algo maskable --timesteps 200000 --n_envs 4

- Linear Q-learning baseline (CPU):

\tuv run block-puzzle-train-linear --episodes 500

- Evaluate saved PPO model visually:

\tuv run block-puzzle-eval --algo maskable --model models/ppo_blockpuzzle.zip --steps 500 --fps 10

- TensorBoard logs:

\tuv run tensorboard --logdir logs/ppo

## Experiments

- Log your runs in `docs/EXPERIMENTS.md`:
  - Date, commit, grid size, algo, steps, reward weights, n_envs
  - Results: `rollout/ep_rew_mean`, `rollout/ep_len_mean`
  - Artifacts: model path and logs path

## AlphaZero Runbook (5x5 simplified set)

- Train (heuristic mixed):

	/home/matteo/Block_Puzzle_RL/block_puzzle_rl/.venv/bin/python -m block_puzzle_rl.az.train_alphazero \
	  --episodes 2000 --sims 128 --epochs 1 --batch_size 256 \
	  --channels 16 --blocks 1 --lr 5e-4 \
	  --heuristic_prior_eps 0.7 \
	  --logdir /home/matteo/Block_Puzzle_RL/logs/az_5x5_simple_heur \
	  --save_path /home/matteo/Block_Puzzle_RL/models/az_5x5_simple_heur.pt \
	  --verbose

- Eval (greedy):

	/home/matteo/Block_Puzzle_RL/block_puzzle_rl/.venv/bin/python -m block_puzzle_rl.az.eval_alphazero \
	  --model /home/matteo/Block_Puzzle_RL/models/az_5x5_simple.pt \
	  --sims 96 --episodes 100 --channels 16 --blocks 1 --temp 0.0 \
	  --heuristic_prior_eps 0.3

- Render:

	/home/matteo/Block_Puzzle_RL/block_puzzle_rl/.venv/bin/python -m block_puzzle_rl.az.render_alphazero \
	  --model /home/matteo/Block_Puzzle_RL/models/az_5x5_simple.pt \
	  --sims 64 --channels 16 --blocks 1 --heuristic_prior_eps 0.3

- Heuristic gist: promote lines, mobility, 2×2 windows; penalize holes, singletons, height/bumpiness.

## AlphaZero Runbook (5x5 stochastic pieces)

- Train (annealed heuristic mix, score-normalized value, checkpoints every 100 eps):

	/home/matteo/Block_Puzzle_RL/block_puzzle_rl/.venv/bin/python -m block_puzzle_rl.az.train_alphazero \
	  --episodes 1500 --sims 192 --epochs 1 --batch_size 256 \
	  --channels 32 --blocks 3 --lr 5e-4 \
	  --heuristic_prior_eps 0.0 --heuristic_eps_start 0.5 --heuristic_eps_end 0.05 --heuristic_eps_decay 0.997 \
	  --temp 1.0 --temp_min 0.1 --temp_decay 0.997 \
	  --logdir /home/matteo/Block_Puzzle_RL/logs/az_5x5_stoch \
	  --save_path /home/matteo/Block_Puzzle_RL/models/az_5x5_stoch.pt \
	  --ckpt_every 100 --verbose

- Train (bigger model, more sims):

	/home/matteo/Block_Puzzle_RL/block_puzzle_rl/.venv/bin/python -m block_puzzle_rl.az.train_alphazero \
	  --episodes 2000 --sims 256 --epochs 1 --batch_size 256 \
	  --channels 64 --blocks 4 --lr 3e-4 \
	  --heuristic_prior_eps 0.0 --heuristic_eps_start 0.6 --heuristic_eps_end 0.05 --heuristic_eps_decay 0.997 \
	  --temp 1.0 --temp_min 0.1 --temp_decay 0.997 \
	  --logdir /home/matteo/Block_Puzzle_RL/logs/az_5x5_stoch_big \
	  --save_path /home/matteo/Block_Puzzle_RL/models/az_5x5_stoch_big.pt \
	  --ckpt_every 100 --verbose

- Train (bigger model, fixed value mapping 0→-1, 1000→+1):

	PYTHONPATH=/home/matteo/Block_Puzzle_RL/block_puzzle_rl/src /home/matteo/Block_Puzzle_RL/block_puzzle_rl/.venv/bin/python -m block_puzzle_rl.az.train_alphazero \
	  --episodes 2000 --sims 256 --epochs 1 --batch_size 256 \
	  --channels 64 --blocks 4 --lr 3e-4 \
	  --heuristic_prior_eps 0.0 --heuristic_eps_start 0.6 --heuristic_eps_end 0.05 --heuristic_eps_decay 0.997 \
	  --temp 1.0 --temp_min 0.1 --temp_decay 0.997 \
	  --value_fixed_min 0 --value_fixed_max 1000 \
	  --logdir /home/matteo/Block_Puzzle_RL/logs/az_5x5_stoch_big_no_heur_fixed_1000 \
	  --save_path /home/matteo/Block_Puzzle_RL/models/az_5x5_stoch_big_no_heur_fixed_1000.pt \
	  --ckpt_every 100 --verbose

- Eval (net-only, greedy):

	/home/matteo/Block_Puzzle_RL/block_puzzle_rl/.venv/bin/python -m block_puzzle_rl.az.eval_alphazero \
	  --model /home/matteo/Block_Puzzle_RL/models/az_5x5_stoch.pt \
	  --sims 64 --episodes 50 --channels 32 --blocks 3 --temp 0.0 --heuristic_prior_eps 0.0 --size 5

- Eval (heuristic-only baseline, no model):

	/home/matteo/Block_Puzzle_RL/block_puzzle_rl/.venv/bin/python -m block_puzzle_rl.az.eval_alphazero \
	  --sims 8 --episodes 50 --heuristic_only --size 5

- Render (net-only):

	/home/matteo/Block_Puzzle_RL/block_puzzle_rl/.venv/bin/python -m block_puzzle_rl.az.render_alphazero \
	  --model /home/matteo/Block_Puzzle_RL/models/az_5x5_stoch.pt \
	  --sims 64 --channels 32 --blocks 3 --temp 0.0

- Render (heuristic-only, no model; increase `--sims` for slower but stronger decisions):

	/home/matteo/Block_Puzzle_RL/block_puzzle_rl/.venv/bin/python -m block_puzzle_rl.az.render_alphazero \
	  --size 9 --sims 2 --heuristic_only --temp 0.0

- Render (random agent):

	/home/matteo/Block_Puzzle_RL/block_puzzle_rl/.venv/bin/python -m block_puzzle_rl.az.render_alphazero --random --sims 1

## What changed vs the simple setup

- Value targets: tanh-normalized terminal score via a running `ScoreNormalizer` (Welford mean/std).
- MCTS backups: r+v for non-terminal nodes; normalized terminal score at terminal nodes.
- Heuristic mixing: root prior mix with annealing (`--heuristic_eps_start/end/decay`).
- Temperature schedule: `--temp`, `--temp_min`, `--temp_decay` per episode.
- Stochastic pieces: each set is sampled with replacement (prevents deterministic loops); `max_episode_steps` enforced.
- Robust sampling: temperature distribution is clamped/renormalized to avoid `np.random.choice` errors.
- Checkpoints: `--ckpt_every N` saves `{save_path}_ep{N}.pt` periodically.
- Heuristic-only mode: pass `--heuristic_only` (no `--model` required). For 9×9 tests, use `--size 9`.

### Fixed terminal score mapping (optional)

- Enable absolute value targets by mapping final score linearly to [-1, 1], with 0 → -1 and 1000 → +1:

	Use `--value_fixed_min 0 --value_fixed_max 1000` when running the stochastic setup trainer under `block_puzzle_rl/src`.

- Why: stationary, interpretable targets that bias learning toward higher absolute scores. Trade-off: if scores exceed the cap, targets clip at +1; set a slightly higher max if needed.

### MCTS simulations guidance

- More `--sims` reduces policy target noise, deepens lookahead, and averages stochastic piece draws, improving sample efficiency. Returns diminish beyond ~512 sims on 5×5.
- Practical cadence: 128–256 early; 512 once policy stabilizes; 1024 only if compute allows. If using a nonzero root heuristic mix, you can often use fewer sims for similar quality.

### Learning signals and evaluation

- Signs of learning (low-heuristic runs): scores rising to 400–500 by ~300 episodes indicate the net is guiding search, despite variance from stochastic pieces.
- Evaluate with temp 0.0 and no heuristic mixing over ≥200 episodes; track mean, median, p95, and hit-rates (≥400/600/800/1000).
- Value calibration: correlation between predicted v and final score (r > 0.3–0.4 early is a good sign). A freeze test (stop training, continue self-play) should stall improvements if learning was the driver.

### Difficulty and sample size (5×5 stochastic pieces)

- Reaching occasional 1000s: ~20k–40k self-play episodes (depends on sims/model size).
- Hitting 1000 regularly (e.g., >10% of episodes): ~50k–100k episodes.
- Median ≈1000 likely requires >100k episodes. Parallel workers scale throughput; episode counts are approximate and sensitive to configs.

## Notes

- Renderer controls: ESC to quit, N to reset.
- If the window is black, reduce `--sims` (e.g., 1–4) for responsiveness.
- Shape mismatch errors happen if you load a 5×5 checkpoint with a 9×9 net; use `--heuristic_only` or retrain for the new size.

## AlphaZero: Improvements and Tests

### Improvements to try

- **Increase MCTS sims**: 128→256→512 as training stabilizes; expect diminishing returns beyond ~512 on 5×5. Combine with more `--workers` to offset throughput loss.
- **Decouple root noise from targets**: add Dirichlet noise only for action selection, compute/store training π without noise to reduce target variance.
- **Soften π early**: raise sampling temperature (e.g., `--temp 1.3–1.8`) for the probabilities used as targets; anneal to 0.1.
- **Heuristic prior mix schedule**: start `--heuristic_eps_start` at 0.3–0.6 and anneal to ~0.05; improves early priors and stabilizes π.
- **Label smoothing on π**: blend ε of uniform mass over legal actions into the target; reduces overconfidence and policy loss.
- **Loss weighting**: downweight value slightly (e.g., `loss = policy + 0.5 * value`) so policy gets more gradient when value converges fast.
- **More optimization per data**: increase `--epochs` (1→2) or batch size moderately to reduce gradient noise.
- **Curriculum on value cap**: train with `--value_fixed_max 600–800`, bump to 1000 when p95 score crosses ~70% of the cap.
- **Ablate fixed vs running normalization**: compare `--value_fixed_min/max` vs default Welford+tanh for sample efficiency.

### Tests and evaluations

- **Greedy eval (net-only)**: run 200–500 episodes with `--temp 0.0`, `--heuristic_prior_eps 0.0`; report mean, median, p95, max, and hit-rates (≥400/600/800/1000).
- **Heuristic baseline**: compare against `--heuristic_only` at similar `--sims` to quantify net contribution.
- **Value calibration**: Pearson r between predicted v at root and final score; early r > 0.3–0.4 indicates useful signal. Optionally bin by v and plot calibration.
- **Freeze test**: freeze model weights and continue self-play; improvements should stall if learning was driving gains.
- **Sims sweep**: 64/128/256/512 sims; measure policy loss, eval scores, and episodes-to-threshold.
- **Heuristic mix sweep**: vary `--heuristic_eps_start/end/decay` and measure stability and sample efficiency.
- **Root noise on/off for stored π**: quantify policy loss and eval differences.
- **Value cap sensitivity**: test `--value_fixed_max` 800/1000/1200; check saturation/clipping effects.

Tip: compare policy CE to uniform baseline ln(num_legal). Being below that and trending down indicates learning even with high variance.
