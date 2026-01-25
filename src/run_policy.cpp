// run_policy.cpp
#include <iostream>
#include <random>
#include <algorithm>
#include <dlib/dnn.h>

#include "../include/neural_network/dlib_policy.hpp"   // policy_net, policy_greedy
#include "../include/neural_network/rl_featurizer.hpp" // featurize(...)
#include "../include/neural_network/rl_actions.hpp"    // legal_actions, to_engine_action
#include "../include/game/game.hpp"
#include "../include/game/blinds.hpp"

// ---------- Config (tweak as you like) ----------
static constexpr int kDefaultPlayers = 3;
static constexpr int kDefaultChips = 10000;
static constexpr int kDefaultHands = 50000;
static constexpr size_t kDefaultBotSeat = 0; // which *seat* uses the policy at the initial lineup
static constexpr uint64_t kDefaultSeed = 12345;
static constexpr int kVerboseEvery = 0; // 0 = silent; else prints every N hands
// -------------------------------------------------

struct StdRng
{
    using result_type = uint64_t;
    omp::XoroShiro128Plus eng{kDefaultSeed};
    static constexpr result_type min() { return 0ULL; }
    static constexpr result_type max() { return ~0ULL; }
    result_type operator()() { return eng(); }
};

static Game make_game(std::size_t nPlayers, int chips, const Blinds &b)
{
    Game g(b);
    for (std::size_t i = 0; i < nPlayers; ++i)
        g.addPlayer(chips);
    return g;
}

static inline unsigned random_legal_action(const std::vector<unsigned> &legal, StdRng &rng)
{
    if (legal.empty())
        return 0u;
    std::uniform_int_distribution<size_t> D(0, legal.size() - 1);
    return legal[D(rng.eng)];
}

// Helper: find the *current* vector index for a given immutable player id
static inline std::size_t find_index_by_id(const std::span<const Player> ps, std::size_t pid)
{
    for (std::size_t i = 0; i < ps.size(); ++i)
    {
        if (ps[i].id == pid)
        {
            return i;
        }
    }
    return 0;
}

int main(int argc, char **argv)
{
    // Basic params (lightweight CLI: argv[1]=hands, argv[2]=players, argv[3]=chips, argv[4]=botSeat)
    const int hands = (argc > 1) ? std::max(1, std::atoi(argv[1])) : kDefaultHands;
    const int players = (argc > 2) ? std::max(2, std::atoi(argv[2])) : kDefaultPlayers;
    const int chips = (argc > 3) ? std::max(1, std::atoi(argv[3])) : kDefaultChips;
    std::size_t botSeat = (argc > 4) ? static_cast<std::size_t>(std::atoi(argv[4])) : kDefaultBotSeat;
    if (botSeat >= static_cast<std::size_t>(players))
        botSeat = 0;

    Blinds blinds{50, 100};

    // Load net
    policy_net net;
    try
    {
        dlib::deserialize("policy_rl.dat") >> net;
    }
    catch (const std::exception &e)
    {
        std::cerr << "Failed to load policy_rl.dat: " << e.what() << "\n";
        return 1;
    }

    // Table + RNG
    auto g = make_game(static_cast<std::size_t>(players), chips, blinds);
    StdRng rng;

    // Capture the *bot identity* once (immutable id), based on initial seating
    const auto &initial_players = g.players();
    const std::size_t bot_initial_index = std::min<std::size_t>(botSeat, initial_players.size() - 1);
    const std::size_t bot_id = initial_players[bot_initial_index].id;

    // Stats
    int64_t sum_bb_for_bot = 0;
    uint64_t action_hist[kNumActions] = {0, 0, 0, 0, 0}; // bot action histogram
    uint64_t bot_decisions = 0;                          // sanity counter

    for (int h = 0; h < hands; ++h)
    {
        // Snapshot stacks BEFORE the hand, keyed by player id
        // (Assumes ids are in a small dense range starting at 0. If not, switch to unordered_map.)
        std::vector<std::uint32_t> chips_before_by_id(g.players().size(), 0);
        for (auto const &p : g.players())
            chips_before_by_id[p.id] = p.chips;

        g.startNewHand(rng);

        while (g.state() != GameState::Finished)
        {
            if (!g.hasCurrentActor())
            {
                g.applyAction(rng, ActionStruct{ActionType::Check, 0});
                continue;
            }

            const std::size_t cur_id = g.currentPlayer().id;
            auto leg = legal_actions(g, cur_id, blinds);

            if (leg.empty())
            {
                g.applyAction(rng, ActionStruct{ActionType::Fold, 0});
                continue;
            }

            // choose action
            unsigned aidx;
            if (cur_id == bot_id)
            {
                auto s = featurize(rng.eng, g, cur_id, blinds);
                aidx = policy_greedy(net, s, leg);
                if (aidx < kNumActions)
                    action_hist[aidx]++;
                ++bot_decisions;
            }
            else
            {
                aidx = random_legal_action(leg, rng);
            }

            // apply
            g.applyAction(rng, to_engine_action(aidx, g, cur_id, blinds));
        }

        // bot result in big blinds (map id -> current index after the hand)
        const auto bot_idx_now = find_index_by_id(g.players(), bot_id);
        const int32_t delta_chips =
            int32_t(g.players()[bot_idx_now].chips) - int32_t(chips_before_by_id[bot_id]);
        const int32_t delta_bb = delta_chips / int32_t(std::max<std::uint32_t>(1, blinds.bigBlind));
        sum_bb_for_bot += delta_bb;

        if constexpr (kVerboseEvery > 0)
        {
            if ((h + 1) % kVerboseEvery == 0)
            {
                std::cout << "Hand " << (h + 1) << "  bot_id=" << bot_id
                          << " Δ=" << delta_bb << "bb  stacks:";
                for (auto const &p : g.players())
                    std::cout << " [" << p.id << ":" << p.chips << "]";
                std::cout << "\n";
            }
        }
    }

    const double bb_per_hand = double(sum_bb_for_bot) / std::max(1, hands);
    const double bb_per_100 = 100.0 * bb_per_hand;

    std::cout << "=========================================\n";
    std::cout << "Eval finished\n"
              << "  hands         : " << hands << "\n"
              << "  players       : " << players << "\n"
              << "  bot seat      : " << botSeat << " (fixed bot_id=" << bot_id << ")\n"
              << "  BB/100 (bot)  : " << bb_per_100 << "\n";

    static const char *kActionName[kNumActions] = {
        "Fold", "Check/Call", "Bet 1/2 pot", "Bet pot", "All-in"};

    std::cout << "  Bot action histogram:\n";
    for (int i = 0; i < kNumActions; ++i)
        std::cout << "    " << kActionName[i] << ": " << action_hist[i] << "\n";

    std::cout << "  bot decisions : " << bot_decisions << "\n";
    std::cout << "  decisions/hand: " << (hands > 0 ? (double)bot_decisions / hands : 0.0) << "\n";
    std::cout << "=========================================\n";

    return 0;
}
