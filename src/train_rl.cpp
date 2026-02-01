#include <iostream>
#include <random>
#include <chrono>
#include <thread>
#include <dlib/dnn.h>
#include <BS_thread_pool.hpp>
#include "../include/neural_network/dlib_policy.hpp"
#include "../include/neural_network/rl_trainer.hpp"

struct StdRng
{
    using result_type = uint64_t;
    omp::XoroShiro128Plus eng;
    StdRng() : eng(std::random_device{}()) {}
    StdRng(uint64_t seed) : eng(seed) {}
    static constexpr result_type min() { return 0ULL; }
    static constexpr result_type max() { return ~0ULL; }
    result_type operator()() { return eng(); }
};

static Game make_game(std::size_t nPlayers, std::uint32_t chips, const Blinds &b)
{
    Game g(b);
    for (std::size_t i = 0; i < nPlayers; ++i)
    {
        g.addPlayer(chips);
    }
    return g;
}

int main()
{
    std::cout << "=== Poker RL Training (Actor-Critic with Self-Play) ===" << std::endl;
    std::cout << "Policy Network: 32 -> 256 -> 128 -> 64 -> 5" << std::endl;
    std::cout << "Value Network:  32 -> 128 -> 64 -> 1" << std::endl;
    std::cout << "Actions: Fold, Check/Call, HalfPot, Pot, AllIn" << std::endl;
    std::cout << "Features: Self-play, entropy bonus, all-in penalty, adaptive epsilon" << std::endl;
    std::cout << std::endl;

    try {
        Blinds blinds{50, 100};
        
        // Training configuration to prevent action collapse
        TrainConfig config;
        config.self_play = true;           // All players use policy (critical!)
        config.entropy_coef = 0.08f;       // Encourage exploration
        config.allin_penalty = 0.4f;       // Penalize excessive all-ins
        config.max_allin_ratio = 25;       // Max 25% all-ins before penalty
        
        std::cout << "Training config:" << std::endl;
        std::cout << "  Self-play: " << (config.self_play ? "yes" : "no") << std::endl;
        std::cout << "  Entropy coefficient: " << config.entropy_coef << std::endl;
        std::cout << "  All-in penalty: " << config.allin_penalty << std::endl;
        std::cout << "  Max all-in ratio: " << config.max_allin_ratio << "%" << std::endl;
        std::cout << std::endl;
        
        std::cout << "Initializing networks..." << std::endl;
        
        // Initialize networks
        policy_net pnet;
        value_net vnet;

        // Initialize policy network with a forward pass
        std::cout << "  Policy network..." << std::endl;
        dlib::matrix<float> dummy_input(kInputDims, 1);
        dummy_input = 0;
        pnet(dummy_input);
        std::cout << "  Policy network initialized." << std::endl;
        
        // Initialize value network with dummy training
        std::cout << "  Value network..." << std::endl;
        {
            std::vector<dlib::matrix<float>> dummy_batch = {dummy_input};
            std::vector<float> dummy_labels = {0.f};
            dlib::dnn_trainer<value_net> vnet_init(vnet);
            vnet_init.set_max_num_epochs(1);
            vnet_init.set_mini_batch_size(1);
            vnet_init.set_iterations_without_progress_threshold(50);
            vnet_init.be_quiet();
            vnet_init.train(dummy_batch, dummy_labels);
        }
        std::cout << "  Value network initialized." << std::endl;

        constexpr std::size_t n_players = 3;
        constexpr std::uint32_t starting_chips = 10000;
        std::cout << "Creating game with " << n_players << " players, " << starting_chips << " chips each..." << std::endl;
        auto g = make_game(n_players, starting_chips, blinds);
        std::cout << "Game created." << std::endl;

        // Create thread pool for parallel equity calculation
        unsigned int num_threads = std::max(1u, std::thread::hardware_concurrency());
        std::cout << "Creating thread pool with " << num_threads << " threads..." << std::endl;
        BS::thread_pool<BS::tp::none> pool(num_threads);
        std::cout << "Thread pool created." << std::endl;

        StdRng rng;
        double epsilon = 0.35;         // Higher initial exploration rate
        float temperature = 1.5f;      // Higher temperature for more diverse sampling

        constexpr int total_epochs = 2000;     // More epochs for self-play learning
        constexpr int hands_per_epoch = 300;   // More hands per epoch
        constexpr int checkpoint_interval = 100;

        std::cout << "Training " << total_epochs << " epochs, "
                  << hands_per_epoch << " hands/epoch" << std::endl;
        std::cout << "Initial epsilon: " << epsilon
                  << ", temperature: " << temperature << std::endl;
        std::cout << std::string(110, '-') << std::endl;

        auto start_time = std::chrono::steady_clock::now();

        float best_win_rate = 0.0f;
        float smoothed_reward = 0.0f;
        float smoothed_win_rate = 0.33f;  // Start at expected random win rate for 3 players

        for (int epoch = 0; epoch < total_epochs; ++epoch)
        {
            // Gradually decrease temperature for exploitation (but keep it higher)
            float current_temp = std::max(1.0f, temperature * static_cast<float>(std::pow(0.999, epoch)));

            auto stats = train_epoch(pnet, vnet, g, rng, blinds, hands_per_epoch,
                                     epsilon, current_temp, starting_chips, pool, config);

            // Exponential moving average
            if (epoch == 0)
            {
                smoothed_reward = stats.mean_reward;
                smoothed_win_rate = stats.win_rate;
            }
            else
            {
                smoothed_reward = 0.95f * smoothed_reward + 0.05f * stats.mean_reward;
                smoothed_win_rate = 0.95f * smoothed_win_rate + 0.05f * stats.win_rate;
            }

            // Print stats
            print_epoch_stats(epoch, stats, epsilon, current_temp);

            // Save best model (after warmup period)
            // Use a combination of win rate and action diversity
            float model_score = stats.win_rate * 0.7f + stats.action_diversity * 0.3f;
            static float best_score = 0.0f;
            if (model_score > best_score && epoch > 100 && stats.action_diversity > 0.3f)
            {
                best_score = model_score;
                best_win_rate = stats.win_rate;
                dlib::serialize("policy_best.dat") << pnet;
                dlib::serialize("value_best.dat") << vnet;
                std::cout << "  >>> New best model! (WinRate: "
                          << (best_win_rate * 100) << "%, Diversity: " 
                          << stats.action_diversity << ")" << std::endl;
            }

            // Periodic checkpoint
            if ((epoch + 1) % checkpoint_interval == 0)
            {
                std::string p_file = "policy_epoch_" + std::to_string(epoch + 1) + ".dat";
                std::string v_file = "value_epoch_" + std::to_string(epoch + 1) + ".dat";
                dlib::serialize(p_file) << pnet;
                dlib::serialize(v_file) << vnet;
                
                auto now = std::chrono::steady_clock::now();
                auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(now - start_time);
                
                std::cout << "  Checkpoint saved | Elapsed: " << elapsed.count() << "s"
                          << " | Smoothed R: " << smoothed_reward
                          << " | Smoothed WR: " << (smoothed_win_rate * 100) << "%" << std::endl;
            }
        }

        auto end_time = std::chrono::steady_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::seconds>(end_time - start_time);

        std::cout << std::string(110, '=') << std::endl;
        std::cout << "Training complete in " << duration.count() << " seconds" << std::endl;
        std::cout << "Best win rate: " << (best_win_rate * 100) << "%" << std::endl;
        std::cout << "Final smoothed reward: " << smoothed_reward << " BB" << std::endl;
        std::cout << "Final smoothed win rate: " << (smoothed_win_rate * 100) << "%" << std::endl;

        dlib::serialize("policy_final.dat") << pnet;
        dlib::serialize("value_final.dat") << vnet;
        std::cout << "Final models saved" << std::endl;
    }
    catch (const std::exception &e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
