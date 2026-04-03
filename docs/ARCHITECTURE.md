# Poker Engine — Architecture & Baseline Reference

> Generated 2026-04-03. Benchmarks run on a clean Release+LTCG build with 5 repetitions each.
> Machine: 16 × 3394 MHz cores, L1-D 32 KiB/core, L1-I 32 KiB/core, L2 512 KiB/core, L3 32 MiB shared.

---

> **Performance regression policy:** Any change that may affect performance — including modifications to `hand.hpp`, `intrinsics.hpp`, `deck.hpp`, `game.hpp`, lookup tables, or any hot-path code — **must be verified against the benchmark baseline in Section 10** before merging. A clean Release+LTCG build with `--benchmark_repetitions=5` is required. Regressions are not acceptable, with one explicit exception: **bug fixes may regress benchmarks**. If a fix is a correctness change (not an optimization), document the regression and the bug it resolves in Section 12.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Repository Layout](#2-repository-layout)
3. [Core Engine: Card & Deck Representation](#3-core-engine-card--deck-representation)
4. [Core Engine: Hand Classification](#4-core-engine-hand-classification)
5. [Core Engine: Probability Simulation](#5-core-engine-probability-simulation)
6. [Full Game Engine](#6-full-game-engine)
7. [ML/RL System](#7-mlrl-system)
8. [Build System](#8-build-system)
9. [Test Suite](#9-test-suite)
10. [Benchmark Baseline](#10-benchmark-baseline)
11. [Key Design Decisions](#11-key-design-decisions)
12. [Performance Optimization History](#12-performance-optimization-history)

---

## 1. Project Overview

A high-performance Texas Hold'em poker engine written in C++23, with:

- **O(1) hand classification** via precomputed lookup tables and bitmask arithmetic
- **Monte Carlo equity simulation** using multithreaded XoroShiro128Plus PRNG
- **Full game engine** (state machine with side-pot handling)
- **Actor-critic RL agent** trained via self-play, using dlib neural networks
- **Web API** (Drogon framework) for external equity queries
- **Google Benchmark** suite for both core engine and ML pipeline

**Key dependencies** (managed via Conan):

| Package | Purpose |
|---|---|
| `GTest` | Test framework |
| `benchmark` | Google Benchmark |
| `bshoshany-thread-pool` | `BS::thread_pool` for multithreaded simulation/featurization |
| `dlib` | Neural network (policy + value networks) |
| `ftxui` | Terminal UI for interactive policy runner |
| `Drogon` | Web framework for REST API |
| `glaze` | JSON serialization |

---

## 2. Repository Layout

```
Poker/
├── include/                              # All headers (header-only library)
│   ├── card_enums.hpp                    # Suit, Rank, Classification enums (bitmask)
│   ├── card.hpp                          # Card struct: getSuit/getRank/getMask, parseCard
│   ├── classification_result.hpp         # ClassificationResult: (Classification<<13)|rank_bits
│   ├── deck.hpp                          # Deck: 64-bit bitmask, popRandomCards, iterator
│   ├── hand.hpp                          # Hand::classify() — O(1) lookup table approach
│   ├── intrinsics.hpp                    # pdep (BMI2), keepTopBits, software fallback
│   ├── game.hpp                          # probabilityOfWinning, compareHands (Monte Carlo)
│   ├── random.hpp                        # XoroShiro128Plus PRNG, FastUniformIntDistribution
│   ├── game/
│   │   ├── game.hpp                      # Game class: state machine, applyAction, showdown
│   │   ├── player.hpp                    # Player struct: hole, chips, committed, invested
│   │   ├── blinds.hpp                    # Blinds struct: smallBlind, bigBlind
│   │   ├── poker_enums.hpp               # GameState, ActionType, ActionStruct enums
│   │   └── pot_manager.hpp               # PotManager::build() — side-pot construction
│   └── neural_network/
│       ├── dlib_policy.hpp               # policy_net / value_net typedefs + inference helpers
│       ├── rl_actions.hpp                # ActIdx enum, LegalActions, legal_actions(), to_engine_action()
│       ├── rl_featurizer.hpp             # featurize() — 32-feature state vector
│       └── rl_trainer.hpp                # Actor-critic training loop, GAE, batch construction
├── src/
│   ├── main.cpp                          # CLI: compute win probability from command line
│   ├── game.cpp                          # 3-player game simulation loop
│   ├── benchmarks.cpp                    # Google Benchmark suite (core engine)
│   ├── ml_benchmarks.cpp                 # Google Benchmark suite (ML pipeline)
│   ├── train_rl.cpp                      # RL training entry point + hyperparameters
│   ├── run_policy.cpp                    # Interactive human-vs-bot + benchmark mode
│   └── api.cpp                           # REST API server (Drogon + glaze)
├── test/
│   ├── poker_tests.cpp                   # 190+ static_assert classification tests
│   ├── game_test.cpp                     # Game setup and basic action tests
│   ├── game_logic_test.cpp               # Full game logic (streets, pots, chip conservation)
│   ├── execution_tests.cpp               # Monte Carlo probability integration tests
│   └── ml_test.cpp                       # RL pipeline unit tests (softmax, GAE, featurize, etc.)
├── neural_network/
│   └── __main__.py                       # Alternative Python/Gymnasium/PPO prototype
├── CMakeLists.txt                        # Build configuration
└── conanfile.txt                         # Conan dependency manifest
```

---

## 3. Core Engine: Card & Deck Representation

### 3.1 Enumerations (`include/card_enums.hpp`)

**`Suit` (`uint8_t`, 4 bits):**
```cpp
Hearts = 1<<0, Diamonds = 1<<1, Clubs = 1<<2, Spades = 1<<3
```

**`Rank` (`uint32_t`, power-of-2 bitmask):**
```cpp
Two = 1<<0, Three = 1<<1, ..., Ace = 1<<12
LowStraight = Two|Three|Four|Five|Ace  // wheel: A-2-3-4-5
HighStraight = Ten|Jack|Queen|King|Ace
```
Index 0 = Two, Index 12 = Ace. All bitwise operators (`|`, `&`, `^`, `~`, `<<`, `>>`) are overloaded.

**`Classification` (`uint16_t`, power-of-2 bitmask):**
```cpp
HighCard=1<<0, Pair=1<<1, TwoPair=1<<2, ThreeOfAKind=1<<3,
Straight=1<<4, Flush=1<<5, FullHouse=1<<6, FourOfAKind=1<<7,
StraightFlush=1<<8, RoyalFlush=1<<9
```

### 3.2 Card (`include/card.hpp`)

Single 32-bit mask: lower 4 bits = suit, upper bits = rank flag.

```cpp
Card c = Card::parseCard("As");  // Ace of spades
c.getSuit()   // → Suit::Spades
c.getRank()   // → Rank::Ace
c.getMask()   // → raw uint32_t
```

### 3.3 Deck (`include/deck.hpp`)

**Storage:** single `uint64_t` — 52 bits used, grouped as 4 × 13-bit suit lanes:
- Bits 0–12: Hearts ranks
- Bits 13–25: Diamonds ranks
- Bits 26–38: Clubs ranks
- Bits 39–51: Spades ranks

A card at (suit_index, rank_index) occupies bit `suit_index * 13 + rank_index`.

**Key operations:**
```cpp
Deck d = Deck::createFullDeck();         // all 52 cards
Deck h = Deck::parseHand("As Kh Qd");   // from notation
Deck two = d.popRandomCards(rng, 2);     // extract 2 random cards (removes them)
Card c   = d.popRandomCard(rng);         // single card extraction
Deck pair = d.popPair(rng);              // optimized 2-card extraction
d.addCard(c);                            // OR-based insertion
d.removeCard(c);                         // AND-NOT-based removal
d.size();                                // popcount(mask)
```

**`DeckIterator`:** standard forward iterator using `m &= m-1` (BLSR) to visit each set bit.

---

## 4. Core Engine: Hand Classification

**File:** `include/hand.hpp`

All logic is in the single static method `Hand::classify(Deck)`. It is `constexpr` and `[[nodiscard]]`, runs in O(1), and returns a `ClassificationResult`.

### 4.1 ClassificationResult (`include/classification_result.hpp`)

Single `uint32_t` encoding:
```
bits 31–13: Classification value (upper 16 bits, shifted left 13)
bits 12– 0: Rank flags (13 bits, lower bitmask of relevant ranks)
```

Direct integer comparison `m_mask < other.m_mask` gives correct hand ordering because:
- Classification occupies the high bits (higher hand type → larger integer)
- Within the same type, rank bits in the low part break ties (higher kicker → larger integer)

**Special rank encoding for complex hands:**
- `FullHouse`: rank_bits = `(trips_rank_index << 4) | pair_rank_index` — not a bitmask, but an index pair packed into 8 bits
- `Pair`: rank_bits = `(pair_rank_index << 9) | kicker_bits` — pair index in high bits, kicker bitmask in low bits

### 4.2 Lookup Tables

Both tables are `constexpr` (computed at compile time):

**`straightTable`** (8192 × `uint16_t` = 16 KB):
```
index: 13-bit rank bitmask (anySuit)
value: 0 if no straight; else the high-rank bit of the straight
```
Wheel (A-2-3-4-5) handled as special case → returns `Rank::Five`.

**`flushTable`** (8192 × `uint16_t` = 16 KB):
```
index: 13-bit rank bitmask of one suit
value: 0 if fewer than 5 cards; else same mask (all 5+ suited cards kept)
```

Total table size: **32 KB** — exactly fills the 32 KB L1 Data Cache.

### 4.3 Internal Structures

**`SuitMasks`:** four `uint16_t` values — one 13-bit rank bitmask per suit, extracted from the 64-bit deck mask via shift+mask:
```cpp
s0 = mask & 0x1FFF;          // Hearts
s1 = (mask >> 13) & 0x1FFF;  // Diamonds
s2 = (mask >> 26) & 0x1FFF;  // Clubs
s3 = (mask >> 39) & 0x1FFF;  // Spades
```
`anySuit()` = `s0|s1|s2|s3` (OR-union of all ranks present).

**`CountInfo`:** returned by `topTwoCounts()`:
```cpp
struct CountInfo {
    uint8_t maxCount;        // 1–4: highest multiplicity
    uint8_t secondMaxCount;  // second-highest multiplicity
    uint16_t pairs;          // pair rank bits (or pair rank for FH)
    uint16_t majorRank;      // trips or quads rank bit
};
```

`topTwoCounts()` uses only bitwise AND/OR/XOR of the four suit masks to identify coincident ranks across suits — no loops:
- `p01 = s0 & s1` — ranks present in both Hearts and Diamonds
- `p23 = s2 & s3` — ranks present in both Clubs and Spades
- `all4 = p01 & p23` — ranks present in all 4 suits (quads)
- `three = (p01 & (s2|s3)) | (p23 & (s0|s1))` — ranks in 3 suits (trips)
- `two = p01 | p23 | ((s0^s1) & (s2^s3))` — ranks in exactly 2 suits (pairs)

### 4.4 `classify()` Flow

```cpp
ClassificationResult Hand::classify(Deck cards) noexcept
```

1. Extract `SuitMasks` from 64-bit deck mask.
2. Compute `anySuit = s0|s1|s2|s3`.
3. `flushMask = getFlush(suits)` — 4 parallel L1 table lookups OR'd together.
4. `straightVal = straightTable[anySuit]` — independent of flush (no data dependency).
5. `[maxCount, secondMaxCount, pairs, majorRank] = topTwoCounts(suits)`.
6. Decision tree (annotated with `[[unlikely]]` where appropriate):

```
if flushMask:
  straightValFlush = straightTable[flushMask]
  if straightValFlush:
    if Ace-high → RoyalFlush
    else        → StraightFlush
  if quads    → FourOfAKind (+ best kicker)
  if FH       → FullHouse (trips_idx<<4 | pair_idx)
  else        → Flush (keepTopBits(flushMask, 5))
else:
  if quads    → FourOfAKind (+ best kicker)
  if FH       → FullHouse (trips_idx<<4 | pair_idx)
  if straight → Straight (straightVal)
  if trips    → ThreeOfAKind (trips + top 2 kickers via BLSR)
  if maxCount != 2 → HighCard (top 5 via 2 conditional BLSRs)
  if TwoPair  → TwoPair (top 2 pairs + 1 kicker)
  else        → Pair (makePairMask: pair rank index + top 3 kickers)
```

### 4.5 Kicker Normalization

**`keepTopBits(m, keep)`** (`include/intrinsics.hpp`): extracts the top `keep` set bits from mask `m` using the BMI2 PDEP hardware instruction:
```cpp
int excess = popcount(m) - keep;
uint32_t mask = (1u << keep) - 1u;
return pdep(mask << excess, m);
```
Used for Flush (top 5), Trips (top 2 kickers), TwoPair (top 2 pairs + kicker).

**Bounded BLSR** for HighCard: `popcount(hc) - 5` excess bits removed via at most 2 `x &= x-1` operations — avoids PDEP overhead (6 vs 8 cycle critical path).

**`makePairMask(anySuit, pairs)`**: strips the pair rank from `anySuit` using XOR, then applies 2 conditional BLSRs for kicker normalization. Encodes pair rank as index (not bit) in the high bits of the result.

---

## 5. Core Engine: Probability Simulation

**File:** `include/game.hpp`

### 5.1 Core Functions

```cpp
// Deterministic: compare known hands against fixed board
WinResult compareHands(Deck heroHole, Deck board, span<Deck> oppHoles);

// Stochastic: random opponents, random remaining board cards
WinResult compareRandomHands(Deck heroHole, Deck board, int numOpponents, Rng &rng);

// Boolean shortcut (no Tie tracking — faster)
bool playerWinsRandomGame(Deck heroHole, Deck board, int numOpponents, Rng &rng);

// Monte Carlo win probability — single-threaded
double probabilityOfWinning(Deck hole, Deck board, int simulations, int numOpponents, Rng &rng);

// Monte Carlo win probability — multithreaded (BS::thread_pool)
double probabilityOfWinning(Deck hole, Deck board, int simulations, int numOpponents, BS::thread_pool &pool);
```

### 5.2 PRNG (`include/random.hpp`)

**`omp::XoroShiro128Plus`**: two `uint64_t` state words, period 2^128−1, seeded via SplitMix64.

**`omp::FastUniformIntDistribution<T, tBits>`**: division-free uniform distribution using Lemire's algorithm. Buffers `tBits`-wide chunks from a single 64-bit RNG output, reusing them across calls of the same distribution.

### 5.3 Multithreading

The multithreaded `probabilityOfWinning` splits `simulations` evenly across all worker threads in the pool. Each thread gets an independent RNG state (different seeds derived from a shared counter). Results (win counts) are summed and divided by `simulations`. This is embarrassingly parallel — no synchronization needed during the simulation loop.

---

## 6. Full Game Engine

**Files:** `include/game/game.hpp`, `player.hpp`, `blinds.hpp`, `poker_enums.hpp`, `pot_manager.hpp`

### 6.1 Data Structures

**`ActionType` / `ActionStruct` (`poker_enums.hpp`):**
```cpp
enum class ActionType { Fold, Check, Call, Bet, Raise, AllIn };
struct ActionStruct { ActionType type; uint32_t amount; };
enum class GameState { PreDeal, PreFlop, Flop, Turn, River, Showdown, Finished };
```

**`Player` (`player.hpp`):**
```cpp
struct Player {
    size_t   id;
    Deck     hole;        // 2 hole cards
    uint32_t chips;
    uint32_t committed;   // chips put in this betting round
    uint32_t invested;    // total chips put in this hand (for side pots)
    bool     folded;
    bool     all_in;
    bool     has_hole;
    bool eligible() const; // alive && !all_in
    bool alive()    const; // !folded && has_hole
};
```

**`BetData`:** `{ pot, currentBet, minRaise }` — all `uint32_t`.

**`PlayersData`:** `{ dealer, current, lastAggressor, toAct }` — tracks position.

### 6.2 `Game` Class (`include/game/game.hpp`)

Template-parameterized on `TRng` for `constexpr`-friendly operations.

**Hand lifecycle:**
```cpp
game.startNewHand(rng);       // deal holes, post blinds, set action order
while (!game.applyAction(rng, action)) {
    // poll game.currentPlayer(), game.betData(), game.board()
    // game.dealer() returns the current dealer seat index
    // construct an ActionStruct and call applyAction again
}
// hand is finished
```

**`applyAction(rng, a)`**: processes one player action and advances the game state. Returns `true` when the hand is over (one winner by fold, or showdown reached). Handles:
- `Fold`: mark player folded, check if only one alive
- `Check`: legal only if `amount_to_call == 0`
- `Call`: `commit(min(to_call, chips))`
- `Bet/Raise`: enforces `minRaise`, sets new `currentBet`, resets `toAct`
- `AllIn`: commits all remaining chips; if sub-minimum, does NOT reopen action to other players

**`commit(player, amount)`**: transfers `min(amount, chips)` → `committed += pay`, `invested += pay`, `pot += pay`; sets `all_in = true` if `chips == 0`.

**Street progression**: `PreFlop → Flop (3 cards) → Turn (1) → River (1) → showdownAndPayout()`.

### 6.3 Showdown & Payout

1. Classify each alive player's 7-card hand: `Hand::classify(hole + board)`.
2. Build side pots: `PotManager::build(players)` — groups chips by investment level.
3. For each pot: find the best `ClassificationResult` among eligible players; split equally among all tied winners (remainder chips go to earliest winner by index).

### 6.4 PotManager (`include/game/pot_manager.hpp`)

Creates side pots for all-in scenarios. Each `Pot` has:
- `amount`: chips in this pot
- `eligiblePlayers`: indices of players eligible to win it

Algorithm: sorts players by `invested` level; each all-in creates a new pot boundary. Players who invested less than the boundary are not eligible for pots above their level.

---

## 7. ML/RL System

**Files:** `include/neural_network/`, `src/train_rl.cpp`, `src/run_policy.cpp`

### 7.1 Architecture Overview

**Algorithm:** PPO (Proximal Policy Optimization) with self-play + Opponent Pool + Generalized Advantage Estimation (GAE, λ=0.95).

**Action space** (5 discrete actions, `include/neural_network/rl_actions.hpp`):
```cpp
enum ActIdx : unsigned {
    A_Fold       = 0,   // fold (only legal when facing a bet)
    A_CheckCall  = 1,   // check if no bet; call otherwise (auto-converts to all-in if needed)
    A_BetHalfPot = 2,   // bet/raise 0.5× pot (only offered when a distinct non-shove size)
    A_BetPot     = 3,   // bet/raise 1.0× pot (same constraint)
    A_AllIn      = 4,   // push all chips (surfaced when SPR is low or stack is short)
    A_COUNT      = 5
};
```

`legal_actions(game, heroIdx, blinds)` returns a `LegalActions` struct (stack-allocated array, no heap allocation). Sized bets are only offered when they produce a distinct chip level that is not equivalent to shoving.

`to_engine_action(idx, game, heroIdx, blinds)` translates abstract index → `ActionStruct`. Short calls become `AllIn`; sized bets respect `minRaise`; if the computed bet equals the full stack it becomes `AllIn`.

`sized_bet_target(bd, add)`:
- No current bet: `max(minRaise, add)`
- Facing bet: `max(currentBet + minRaise, currentBet + add)`

### 7.2 Neural Network Architectures (`include/neural_network/dlib_policy.hpp`)

Constants:
```cpp
constexpr int kInputDims  = 32;
constexpr int kNumActions = 5;
```

**PPO label type:**
```cpp
struct PPOLabel {
    unsigned action;      // index of action taken (0..kNumActions-1)
    float    old_log_prob; // log π_θ_old(a|s) at collection time
    float    advantage;   // normalized advantage (post entropy-bonus)
};
```

**Custom PPO loss layer (`loss_ppo_`):** Implements the clipped surrogate objective:
```
J_CLIP = E[ min(ρ·A, clip(ρ, 1−ε, 1+ε)·A) ]   where ρ = exp(new_log_prob − old_log_prob)
```
Clip ε = 0.2.  Gradient: `g_scale · (softmax_j − δ_{j,a})` where `g_scale = A·ρ` when unclipped, 0 when clipped. The log-ratio is clamped to [−5, 5] for numerical stability.

`loss_ppo_` must expose `training_label_type = PPOLabel` and `output_label_type = PPOLabel` so dlib's `add_loss_layer` dispatches `trainer.train()` and `to_label()` correctly.

**Policy network:**
```
input(32) → FC(512) → ReLU → FC(256) → ReLU → FC(128) → ReLU → FC(5)
Loss: loss_ppo (PPO clipped surrogate)
```

**Value network:**
```
input(32) → FC(128) → ReLU → FC(64) → ReLU → FC(1)
Loss: mean squared error (dlib::loss_mean_squared)
```

**Inference helpers** (all use `forward_single` which bypasses the loss layer):
- `get_action_logits(net, s)` → `vector<float>` of 5 raw logits
- `predict_value(vnet, s)` → scalar V(s)
- `softmax_legal(logits, legal)` → masked softmax (illegal actions = 0, numerically stable)
- `policy_sample(net, s, legal, rng, temp)` → sampled action index
- `policy_sample_with_probs(net, s, legal, rng, temp)` → `(action, probs)` in one forward pass
- `policy_greedy(net, s, legal)` → argmax over legal actions
- `compute_entropy(probs)` → H(p) in nats

### 7.3 Feature Engineering (`include/neural_network/rl_featurizer.hpp`)

`featurize(game, heroIdx, blinds, pool)` → `dlib::matrix<float>` of shape (32, 1).

Normalization caps:
```cpp
kMaxToCallBB   = 100.f   // to-call capped at 100 BB
kMaxStackBB    = 200.f   // stack/pot capped at 200 BB
kMaxBetToPot   = 3.f     // bet-to-pot capped at 3×
kMaxSPR        = 20.f    // stack-to-pot ratio capped at 20
kMaxRaiseToPot = 2.f     // min-raise-to-pot capped at 2×
```

**Feature layout (32 features):**

| Index | Name | Description |
|---|---|---|
| 0–3 | street_onehot | One-hot: PreFlop / Flop / Turn / River |
| 4 | can_check | 1 if no bet to call |
| 5 | to_call_bb | Amount to call, normalized to BB (cap 100) |
| 6 | pot_bb | Pot size in BB (cap 200) |
| 7 | pot_odds | `to_call / (pot + to_call)` |
| 8 | facing_bet | 1 if there is a live bet |
| 9 | bet_to_pot | `currentBet / pot` (cap 3×) |
| 10 | hero_stack_bb | Hero chips in BB (cap 200) |
| 11 | spr | Stack-to-pot ratio (cap 20) |
| 12 | committed_ratio | `committed / (chips + committed)` |
| 13 | avg_opp_stack_bb | Average opponent chips in BB |
| 14 | effective_stack_bb | `min(hero, avg_opp)` in BB |
| 15 | can_raise | 1 if hero has chips > to_call |
| 16 | alive_frac | Alive players / total players |
| 17 | elig_frac | Eligible players / total players |
| 18 | position | `(heroIdx − dealer) mod n / (n − 1)` — dealer-relative: 0=BTN, 0.5=SB, 1.0=BB |
| 19 | equity | Monte Carlo win probability (500 sims) |
| 20 | ev_positive | 1 if `equity > pot_odds` |
| 21 | ev_margin | `max(0, equity - pot_odds)` |
| 22 | short_stack_value | 1 if `equity > 0.5 && spr < 4` |
| 23 | fold_indicator | 1 if `equity < 0.3 && facing_bet` |
| 24 | raise_indicator | 1 if `equity > 0.7 && can_raise` |
| 25 | pot_ownership | `hero.invested / pot` (cap 1.0) |
| 26 | min_opp_stack_bb | Shortest opponent stack in BB |
| 27 | min_raise_fraction | `minRaise / hero.chips` (cap 1.0) |
| 28 | min_raise_to_pot | `minRaise / pot / 2.0` (cap 1.0) |
| 29 | opp_stack_ratio | `min_opp / hero.chips / 2.0` (cap 1.0) |
| 30 | hero_stack_invested | `chips / (chips + invested)` |
| 31 | facing_allin | 1 if a call would put hero all-in |

Equity (feature 19) uses 500 Monte Carlo simulations via the multithreaded `probabilityOfWinning`. During interactive play this is raised to 5000 for accuracy.

### 7.4 Training Loop (`include/neural_network/rl_trainer.hpp`, `src/train_rl.cpp`)

**Hyperparameters:**
```
total_epochs         = 2,000
hands_per_epoch      = 300
checkpoint_interval  = 100 epochs
initial_epsilon      = 0.35
initial_temperature  = 1.5
policy_lr            = 1e-4 (SGD + momentum 0.9, base weight decay 5e-4)
value_lr             = 5e-4
mini_batch_size      = min(256, dataset_size)
entropy_coef         = 0.08
action_diversity_coef = 0.10
min_action_prob      = 0.05
allin_penalty        = 0.40
max_allin_ratio      = 25 (%)
gae_lambda           = 0.95
game setup           = 3 players, 10,000 chips, blinds 50/100
```

**Per-epoch flow:**
1. Play `hands_per_epoch` self-play hands. For non-hero seats: with 50% probability, use a policy snapshot sampled from the **opponent pool** instead of the current policy. The pool holds up to 5 past checkpoints (populated every 100 epochs).
2. At each decision point: featurize state, sample action (ε-greedy or temperature-based), record `(state, action, V(s), log_prob, entropy)`. `log_prob` is recorded from whichever policy (current or pool snapshot) made the decision — this becomes `old_log_prob` in `PPOLabel`.
3. **GAE** (backwards induction): `δ_t = r_t + V(s_{t+1}) - V(s_t)`, `A_t = δ_t + λ·A_{t+1}`. Reward = chip delta in BB at terminal state, 0 otherwise. Discount γ = 1.0 within a hand.
4. **PPO batch construction** (`build_training_batch`):
   - Adjusted advantage: `adj = A + entropy_coef * H(probs)`.
   - All-in penalty: if all-in ratio > `max_allin_ratio`, `adj -= allin_penalty * excess`.
   - Normalize `adj` to zero mean / unit variance.
   - **All samples included** in the policy batch — the PPO clipped surrogate provides the trust-region guarantee, eliminating the need for positive-only filtering.
   - All-in samples hard-capped at `max_allin_ratio`% of batch (independent of PPO).
   - Policy label: `PPOLabel{action, old_log_prob, adj_advantage}`.
   - Value batch: all states, target = R (terminal chip delta in BB).
5. **Adaptive epsilon/LR**:
   - `diversity < 0.3`: halve LR, increase ε (explore more).
   - `diversity < 0.5`: slower ε decay (`0.999x` instead of `0.997x`).
6. **Checkpointing**: save `policy_epoch_N.dat` / `value_epoch_N.dat` every 100 epochs; push current policy to opponent pool.
   - Best model: `score = win_rate × 0.7 + action_diversity × 0.3` (only if epoch > 100 and diversity > 0.3).
   - Final model: always saved at end.

**Opponent pool** (`OpponentPool`, `include/neural_network/rl_trainer.hpp`): deque of up to 5 `policy_net` snapshots. Non-hero seats sample uniformly from the pool with 50% probability per hand. Off-policy steps are correctly handled by the PPO importance ratio ρ.

**Serialized model:** `policy_final.dat` / `value_final.dat` (current working directory).

### 7.5 Interactive & Benchmark Modes (`src/run_policy.cpp`)

**Interactive mode** (default): Human as seat 0 vs neural bot (seat 1) in a fullscreen FTXUI terminal UI. Shows hole cards, board, pot, stacks, equity estimate, action history. Bot displays its action probability distribution each turn. Ends when human is busted (< 1 BB).

**Benchmark mode** (`--bench N`): Runs N hands of policy bot vs random players. Reports BB/100, win%, total profit per player, hands/second, final action distribution.

---

## 8. Build System

**File:** `CMakeLists.txt`

**Compiler flags:**
- MSVC: `/W4 /WX /arch:AVX2` (warnings as errors, AVX2 required)
- GCC/Clang: `-Wall -Wextra -Werror -pedantic -march=native -mavx2`
- All: `set(CMAKE_INTERPROCEDURAL_OPTIMIZATION TRUE)` — enables LTCG/LTO

**Executables and their link targets:**

| Target | Source | Key dependencies |
|---|---|---|
| `Poker` | `src/main.cpp` | `bshoshany-thread-pool` |
| `Poker_Game` | `src/game.cpp` | `bshoshany-thread-pool` |
| `Poker_Trainer` | `src/train_rl.cpp` | `bshoshany-thread-pool`, `dlib` |
| `Poker_RunPolicy` | `src/run_policy.cpp` | `bshoshany-thread-pool`, `dlib`, `ftxui` |
| `Poker_Benchmark` | `src/benchmarks.cpp` | `benchmark`, `bshoshany-thread-pool` |
| `Poker_MLBenchmark` | `src/ml_benchmarks.cpp` | `benchmark`, `bshoshany-thread-pool`, `dlib` |
| `Poker_Api` | `src/api.cpp` | `Drogon`, `bshoshany-thread-pool`, `glaze` |
| `PokerTest` | `test/` | `GTest` |

**Build commands:**
```bash
# Tests
cmake --build build --parallel 16 --config Release --target PokerTest
cd build && ctest --build-config Release --output-on-failure

# Benchmarks (--clean-first ensures reliable LTCG numbers)
cmake --build build --parallel 16 --config Release --target Poker_Benchmark --clean-first
build/Release/Poker_Benchmark.exe --benchmark_repetitions=5 --benchmark_display_aggregates_only=true

# ML benchmarks
cmake --build build --parallel 16 --config Release --target Poker_MLBenchmark
build/Release/Poker_MLBenchmark.exe --benchmark_repetitions=3 --benchmark_display_aggregates_only=true
```

---

## 9. Test Suite

**Total: 136 tests, 100% passing** (as of 2026-04-03).

| File | Tests | What it covers |
|---|---|---|
| `test/poker_tests.cpp` | ~90 (many `static_assert`) | All 10 hand types, kicker comparisons, wheel straight/SF, deck parsing, streaming. Bug regressions: FullHouse encoding, double-trips, kicker normalization. |
| `test/game_test.cpp` | ~15 | Blind posting, initial bet state, player counts, heap-up blinds. |
| `test/game_logic_test.cpp` | ~25 | Full hand execution, street progression (Flop/Turn/River card counts), chip conservation, pot/payout, all-in side pots, check/raise rules. |
| `test/execution_tests.cpp` | ~8 | Monte Carlo integration: royal flush = 100% win, unbeatable quads = 100%, J-high flush vs field ≈ 86–89%, two-pair on paired board ≈ 18–22%. |
| `test/ml_test.cpp` | ~36 | `softmax_legal` (sums to 1, illegal = 0), `compute_entropy`, `action_diversity`, `legal_actions`, `featurize` (dims, no NaN/Inf, normalized ranges), `build_training_batch` (advantage normalization, all-in cap, `PPOLabel` field correctness), `policy_sample`, GAE correctness (λ=0 gives TD, λ=1 gives MC). |

---

## 10. Benchmark Baseline

> **Setup:** Windows 10 Pro, 16 × 3394 MHz cores, L1-D 32 KiB/core, L2 512 KiB/core, L3 32 MiB.
> Clean Release+LTCG build. `Poker_Benchmark`: 5 repetitions. `Poker_MLBenchmark`: 3 repetitions.
> All means shown (use these as the reference baseline for optimization comparisons).

### 10.1 Core Engine Benchmarks (`Poker_Benchmark`)

**Deck construction:**

| Benchmark | Mean (ns) |
|---|---|
| `BM_CreateFullDeck` | 1.08 |
| `BM_CreateRandom7Cards` | 8.45 |
| `BM_CreateRandom7CardsSequential` | 21.8 |
| `BM_PopRandomCard` | 2.41 |
| `BM_PopPairOfRandomCards` | 2.39 |
| `BM_DeckIteration` (52 cards) | 70.8 |
| `BM_ParseHand` | 79.2 |

**`BM_PopRandomCards/N` (N cards at once):**

| N cards | Mean (ns) |
|---|---|
| 1 | 4.79 |
| 2 | 4.13 |
| 3 | 6.51 |
| 4 | 7.59 |
| 5 | 8.50 |
| 6 | 9.34 |
| 7 | 10.4 |
| 8 | 11.3 |
| 9 | 12.2 |
| 10 | 13.3 |

**Hand classification (`Hand::classify`):**

| Benchmark | Mean (ns) |
|---|---|
| `BM_Classification` (fixed hand) | **10.8** |
| `BM_ClassificationVaryingHands` (random) | **22.1** |
| `BM_ClassifyRoyalFlush` | 5.47 |
| `BM_ClassifyStraightFlush` | 5.71 |
| `BM_ClassifyFourOfAKind` | 8.26 |
| `BM_ClassifyFullHouse` | 9.15 |
| `BM_ClassifyFlush` | 6.31 |
| `BM_ClassifyStraight` | 5.89 |
| `BM_ClassifyThreeOfAKind` | 9.34 |
| `BM_ClassifyTwoPair` | 12.5 |
| `BM_ClassifyOnePair` | 10.3 |
| `BM_ClassifyHighCard` | 6.15 |

**`BM_ClassificationVaryingHands` = 22.1 ns** is the most representative throughput figure, as it exercises all code paths with random branch prediction. Corresponds to approximately **~45 M classifications/second**.

**Probability simulation:**

| Benchmark | Mean (ns) |
|---|---|
| `BM_CompareHands` (1 opp, known board) | 25.5 |
| `BM_CompareHandsMultipleOpponents` | 66.2 |
| `BM_PlayerWinsRandomGame/2` (1 opp) | 31.6 |
| `BM_PlayerWinsRandomGame/3` (2 opps) | 38.7 |
| `BM_PlayerWinsRandomGame/4` | 41.7 |
| `BM_PlayerWinsRandomGame/5` | 43.8 |
| `BM_PlayerWinsRandomGame/6` | 45.0 |
| `BM_PlayerWinsRandomGame/7` | 44.2 |
| `BM_PlayerWinsRandomGame/10` | 43.4 |
| `BM_PlayerWinsRandomGamePreflop/2` | 46.9 |
| `BM_PlayerWinsRandomGamePreflop/3` | 55.9 |
| `BM_PlayerWinsRandomGamePreflop/10` | 80.5 |
| `BM_PlayerWinsRandomGameFlop/2` | 42.4 |
| `BM_PlayerWinsRandomGameFlop/3` | 48.6 |

### 10.2 ML Pipeline Benchmarks (`Poker_MLBenchmark`)

| Benchmark | Mean |
|---|---|
| `BM_LegalActions` | 21.0 ns |
| `BM_SoftmaxLegal_AllActions` | 60.5 ns |
| `BM_SoftmaxLegal_PartialLegal` | 50.3 ns |
| `BM_ComputeActionDiversity` | 19.3 ns |
| `BM_ValueForwardPass` | 7.03 µs |
| `BM_PolicyForwardPass` | 100 µs |
| `BM_PolicySampleWithProbs` | 99.5 µs |
| `BM_PolicySampleThenGetProbs` | 202 µs |
| `BM_Featurize/1 thread` | 47.9 µs |
| `BM_Featurize/4 threads` | 26.1 µs |
| `BM_Featurize/8 threads` | 25.3 µs |
| `BM_BuildTrainingBatch/256` | 54.7 µs (~4.7 M items/s) |
| `BM_BuildTrainingBatch/1024` | 219 µs (~4.7 M items/s) |
| `BM_BuildTrainingBatch/4096` | 1223 µs (~3.4 M items/s) |
| `BM_PlayOneHand` (full hand, 3 players, neural bots) | 1.23 ms |

**Key observations:**
- Policy network forward pass (100 µs) dominates the per-decision cost — 14× slower than featurization (7 µs at 1 thread).
- Featurization scales reasonably with threads (47.9 µs → 25.3 µs at 8 threads) because equity simulation (500 MC sims) parallelizes well.
- Value network (7 µs) is ~14× faster than policy network, as expected from the size difference (FC 128→64→1 vs FC 512→256→128→5).
- `BM_PlayOneHand` = 1.23 ms means ~813 hands/second at 3 players — sufficient for 2,000 epochs × 300 hands = 600,000 training hands in ~12 minutes.

---

## 11. Key Design Decisions

### Bitmask Representation Throughout

Every entity (suit, rank, card, deck, classification) is a bitmask. This enables:
- O(1) set operations (union, intersection, difference) via bitwise instructions
- Compile-time evaluation via `constexpr` everywhere
- Direct table indexing with rank/suit masks
- Correct comparison encoding via bit-packing in `ClassificationResult`

### Precomputed 32 KB Lookup Tables

`straightTable` and `flushTable` each cost 16 KB (8192 × `uint16_t`). Together they exactly fill the 32 KB L1 Data Cache on this hardware. This is intentional: 2 L1-resident table lookups (~4 cycles each) are much faster than any algorithmic straight/flush detection.

Removing these tables and replacing with inline logic has been tested and regressed OnePair by ~2 ns.

### constexpr Everything

All core logic is `constexpr`. This means:
- Tables are computed at compile time (zero runtime cost)
- `static_assert` tests verify correctness at compile time (no test runner overhead)
- `classify()` is usable in constant expressions

### Self-Play for RL

All 3 seats use the same policy during training. This is critical for stability: the opponent distribution is always "self-similar" and improves as the policy improves, avoiding fixed-strategy overfitting.

### Abstract Action Space

The 5-action abstraction (fold, check/call, ½pot, pot, all-in) avoids the continuous bet-sizing problem entirely. The mapping to concrete chip amounts is handled by `to_engine_action`, which respects `minRaise` and automatically promotes sized bets to all-in when the stack is too short.

### Register Pressure Awareness (MSVC + LTCG)

Several optimization attempts regressed performance due to register spills caused by adding new long-lived variables, even when those variables would run in parallel early in the function. The compiler's register allocator becomes the bottleneck. Key rule: adding a variable that seems free (computed from existing data) can cause spills that hurt other code paths by ~1–2 ns.

---

## 12. Performance Optimization History

Chronological record of `BM_ClassificationThroughput` improvements (fixed hand, Release+LTCG):

| Baseline | Throughput | Change | Description |
|---|---|---|---|
| Pre-optimization | 67.8 M/s | — | Starting point |
| After top5Table removal | ~75 M/s | +7 M/s | `top5Table` (16 KB) replaced with `keepTopBits()` via PDEP |
| After straightTable compaction | +26% | +20 M/s | `StraightInfo{bool,Rank}` = 8 bytes/entry (64 KB due to padding) → `uint16_t` = 2 bytes/entry (16 KB) |
| After hiTable/top2Table/top3Table removal | ~93 M/s | +5 M/s | 48 KB of L2-cold tables replaced by `highBit()` + `keepTopBits()` (2–6 cycle bit ops vs ~10 cycle L2 hit) |
| After topTwoCounts refactor | ~95 M/s | +2 M/s | Reuse `p01 = s0&s1`, `p23 = s2&s3`; replace `popcount >= 2` with `x & (x-1)` |
| After flush→straight dependency break | 97.5 M/s | +4.5 M/s | `getFlush` returns `uint16_t flushMask` directly; `straightTable[anySuit]` computed independently; eliminates data dependency chain |
| After HighCard BLSR optimization | **~108 M/s** | +10 M/s | HighCard uses `popcount(hc)-5` + 2 conditional BLSRs instead of `keepTopBits(anySuit,5)` via PDEP (6 vs 8 cycle critical path); also removed redundant `popcount > 1` guard from HighCard path |

**What was tried and did NOT help:**
- Removing `flushTable` and using inline popcount branches: sequential branch structure hurt OnePair/TwoPair (+2 ns)
- Pre-computing `excess = numCards-5` at top of `classify()`: register pressure from a long-lived variable regressed TwoPair
- Pre-computing `pairExcess = popcount(anySuit)-4` before `topTwoCounts()`: same register pressure issue
- Bounded BLSR for HighCard at 85 M/s baseline: regressed at that point; only worked after surrounding code structure changed and HighCard became better isolated

**Bug fixes that had correctness tests added:**
1. `FullHouse` comparison: must encode as `(tripsIdx << 4) | pairIdx` (index pair), not rank bitmasks, to compare correctly
2. Double-trips (AAA KKK): correctly classifies as FullHouse (highest trips become the trips, lower become the pair)
3. `FourOfAKind` kicker: must normalize to single highest kicker (not all non-quad ranks)
4. `ThreeOfAKind` kickers: must normalize to top 2 kickers
5. `Flush` with 6 suited cards: must trim to top 5 and still detect wheel straight flush correctly
