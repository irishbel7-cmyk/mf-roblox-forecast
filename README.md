# mf-roblox-forecast

Entry for make.fun's "Predict Roblox's Up-and-Coming top 10" competition ($500).
A Python program that reads the Up-and-Coming chart history and writes one
forecast — the probability of each game holding each of the top 10 places over
the next 72 hours — to `/output/forecast.json`.

## Model

Spike-and-slab rank forecast with per-game volatility:

- Each game's rank at a future hour is a mixture: with probability
  `exp(-h/24)` it **stays** near its current rank (narrow Gaussian, width
  scaled by that game's own historical rank volatility); otherwise it
  **moves** to a wide Gaussian around a slowly fading rank (games drift down
  as they age). The mixture is averaged over the scored window's hours.
- A game whose 28-day eligibility ends inside the window (from
  `details.created`, when the house recorder provides it) contributes no mass
  after its exit hour. Games that fell off the chart recently are decayed.
- Each place's probabilities are normalised to leave a share unnamed (for
  games the history cannot name — new entrants), then a water-filling step
  names only the games whose probability beats the unnamed share: the optimal
  allocation under a proper log score.

## Practice score (public archive)

- **This entry: -34.28** (mean log score over the 10 practice reads)
- House baseline (persistence): -43.51
- House model forecast: -37.2

## Run

```sh
mkdir -p out
MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
# writes out/forecast.json
```

Standard library only, no network, no randomness: two runs on the same
history write byte-identical output. Runs in well under a second.

License: MIT
