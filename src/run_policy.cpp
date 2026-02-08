// run_policy.cpp
#include <iostream>
#include <random>
#include <algorithm>
#include <thread>
#include <dlib/dnn.h>
#include <BS_thread_pool.hpp>
#include <iomanip> // Adicionado para formatação de tabela

#include "../include/neural_network/dlib_policy.hpp"
#include "../include/neural_network/rl_featurizer.hpp"
#include "../include/neural_network/rl_actions.hpp"
#include "../include/game/game.hpp"
#include "../include/game/blinds.hpp"

// ---------- Configuração ----------
static constexpr int kDefaultPlayers = 3;
static constexpr int kDefaultChips = 10000;
static constexpr int kDefaultHands = 10000; // 10k mãos para ter relevância estatística
static constexpr size_t kDefaultBotSeat = 0; 
static constexpr uint64_t kDefaultSeed = 42; // Seed fixa para reprodutibilidade
// ----------------------------------

struct StdRng
{
    using result_type = uint64_t;
    omp::XoroShiro128Plus eng{kDefaultSeed};
    static constexpr result_type min() { return 0ULL; }
    static constexpr result_type max() { return ~0ULL; }
    result_type operator()() { return eng(); }
};

struct PlayerStats {
    int64_t total_profit_bb = 0;
    int hands_won = 0;
    int hands_played = 0;
    std::string type; // "Neural" ou "Random"
};

static Game make_game(std::size_t nPlayers, int chips, const Blinds &b)
{
    Game g(b);
    for (std::size_t i = 0; i < nPlayers; ++i)
        g.addPlayer(chips);
    return g;
}

static inline unsigned random_legal_action(const std::span<const unsigned> legal, StdRng &rng)
{
    if (legal.empty())
        return 0u;
    std::uniform_int_distribution<size_t> D(0, legal.size() - 1);
    return legal[D(rng.eng)];
}

// Encontra o índice atual no vetor g.players() baseado no ID imutável do jogador
static inline std::size_t find_index_by_id(const std::span<const Player> ps, std::size_t pid)
{
    for (std::size_t i = 0; i < ps.size(); ++i)
    {
        if (ps[i].id == pid)
            return i;
    }
    return 0;
}

int main(int argc, char **argv)
{
    // Parâmetros via linha de comando
    const int hands = (argc > 1) ? std::max(1, std::atoi(argv[1])) : kDefaultHands;
    const std::size_t players = (argc > 2) ? std::max(2, std::atoi(argv[2])) : kDefaultPlayers;
    const int chips = (argc > 3) ? std::max(1, std::atoi(argv[3])) : kDefaultChips;
    std::size_t botSeat = (argc > 4) ? static_cast<std::size_t>(std::atoi(argv[4])) : kDefaultBotSeat;
    std::string model_file = (argc > 5) ? argv[5] : "policy_best.dat"; // Tenta carregar o melhor modelo por padrão

    if (botSeat >= static_cast<std::size_t>(players))
        botSeat = 0;

    Blinds blinds{50, 100};

    // Carregar Rede Neural
    policy_net net;
    try
    {
        dlib::deserialize(model_file) >> net;
        std::cout << ">> Modelo carregado: " << model_file << "\n";
    }
    catch (const std::exception &e)
    {
        std::cerr << "ERRO: Falha ao carregar " << model_file << ": " << e.what() << "\n";
        std::cerr << "Certifique-se de ter treinado o modelo primeiro (execute ./Poker_Trainer).\n";
        return 1;
    }

    // Inicialização do Jogo
    auto g = make_game(static_cast<std::size_t>(players), chips, blinds);
    StdRng rng;
    unsigned int num_threads = std::max(1u, std::thread::hardware_concurrency());
    BS::thread_pool<BS::tp::none> pool(num_threads);

    // Identificar quem é o Bot Neural e quem são os Bots Aleatórios
    const auto &initial_players = g.players();
    const std::size_t bot_initial_index = std::min<std::size_t>(botSeat, initial_players.size() - 1);
    const std::size_t bot_id = initial_players[bot_initial_index].id;

    // Estatísticas por ID de jogador
    std::vector<PlayerStats> stats(players);
    for(size_t i=0; i < players; ++i) {
        stats[i].type = (initial_players[i].id == bot_id) ? "NEURAL (Hero)" : "RANDOM (Villain)";
    }

    uint64_t action_hist[kNumActions] = {0}; 

    std::cout << "\n=== INICIANDO BENCHMARK: Neural vs Random ===\n";
    std::cout << "Mãos: " << hands << " | Players: " << players << " | Bot Seat: " << botSeat << "\n";
    std::cout << "---------------------------------------------\n";

    auto start_time = std::chrono::steady_clock::now();

    for (int h = 0; h < hands; ++h)
    {
        // Resetar stacks se alguém quebrar (para manter o jogo rodando)
        bool needs_reset = false;
        for (const auto &p : g.players())
        {
            if (p.chips < blinds.bigBlind * 3)
            {
                needs_reset = true;
                break;
            }
        }
        if (needs_reset)
        {
            g.resetPlayerChips(chips);
        }

        // Snapshot dos stacks antes da mão
        std::vector<std::uint32_t> chips_before_by_id(players, 0);
        for (auto const &p : g.players())
            chips_before_by_id[p.id] = p.chips;

        g.startNewHand(rng);

        // --- Loop da Mão ---
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

            unsigned aidx;
            // Lógica: Se for o Bot ID, usa a Rede. Se não, usa Random.
            if (cur_id == bot_id)
            {
                auto s = featurize(g, cur_id, blinds, pool);
                aidx = policy_greedy(net, s, leg); // Usa a melhor ação (Greedy) para avaliação
                
                if (aidx < kNumActions) action_hist[aidx]++;
            }
            else
            {
                aidx = random_legal_action(leg, rng);
            }

            g.applyAction(rng, to_engine_action(aidx, g, cur_id, blinds));
        }

        // --- Pós-Mão: Calcular Resultados ---
        for (size_t i = 0; i < players; ++i) {
            size_t pid = initial_players[i].id; // ID original
            size_t current_idx = find_index_by_id(g.players(), pid);
            
            int32_t chips_after = static_cast<int32_t>(g.players()[current_idx].chips);
            int32_t chips_before = static_cast<int32_t>(chips_before_by_id[pid]);
            int32_t delta = chips_after - chips_before;
            
            // Converter para Big Blinds
            int32_t delta_bb = delta / static_cast<int32_t>(blinds.bigBlind);

            stats[i].total_profit_bb += delta_bb;
            stats[i].hands_played++;
            if (delta > 0) {
                stats[i].hands_won++;
            }
        }

        // Feedback visual a cada 10%
        if (hands >= 10 && (h + 1) % (hands / 10) == 0) {
            std::cout << "." << std::flush;
        }
    }

    auto end_time = std::chrono::steady_clock::now();
    double elapsed_sec = std::chrono::duration<double>(end_time - start_time).count();

    // --- Relatório Final ---
    std::cout << "\n\n================ RESULTADOS FINAIS ================\n";
    std::cout << "Tempo: " << std::fixed << std::setprecision(2) << elapsed_sec << "s (" 
              << (hands / elapsed_sec) << " mãos/seg)\n\n";

    std::cout << std::left << std::setw(15) << "Player" 
              << std::setw(20) << "Type" 
              << std::setw(15) << "BB/100" 
              << std::setw(15) << "Win Rate %" 
              << std::setw(15) << "Total Profit (BB)" << "\n";
    std::cout << std::string(80, '-') << "\n";

    for (size_t i = 0; i < players; ++i) {
        double bb_100 = (double)stats[i].total_profit_bb / stats[i].hands_played * 100.0;
        double win_rate = (double)stats[i].hands_won / stats[i].hands_played * 100.0;

        std::cout << std::left << std::setw(15) << ("Seat " + std::to_string(i))
                  << std::setw(20) << stats[i].type
                  << std::setw(15) << std::fixed << std::setprecision(2) << bb_100
                  << std::setw(15) << win_rate
                  << std::setw(15) << stats[i].total_profit_bb << "\n";
    }

    std::cout << "\nDistribuição de Ações do Bot Neural:\n";
    static const char *kActionName[kNumActions] = {"Fold", "Check/Call", "Bet 1/2", "Bet Pot", "All-in"};
    int total_actions = 0;
    for (int i : action_hist) total_actions += i;
    
    for (int i = 0; i < kNumActions; ++i) {
        double pct = (total_actions > 0) ? (100.0 * action_hist[i] / total_actions) : 0.0;
        std::cout << "  " << std::left << std::setw(12) << kActionName[i] 
                  << ": " << std::setw(8) << action_hist[i] 
                  << "(" << std::fixed << std::setprecision(1) << pct << "%)\n";
    }
    std::cout << "===================================================\n";

    return 0;
}