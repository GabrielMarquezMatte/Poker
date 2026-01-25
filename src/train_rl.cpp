#include <iostream>
#include <random>
#include <dlib/dnn.h>
#include "../include/neural_network/dlib_policy.hpp"
#include "../include/neural_network/rl_trainer.hpp"
struct StdRng
{
    using result_type = uint64_t;
    omp::XoroShiro128Plus eng{123};
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
    Blinds blinds{50, 100};
    policy_net net;
    auto g = make_game(3, 10000, blinds);

    StdRng rng;
    float moving_baseline = 0.f;
    std::vector<float> seat_baseline;
    double epsilon = 0.10;

    for (int epoch = 0; epoch < 200; ++epoch)
    {
        train_reinforce_epoch(net, g, rng, blinds, 200, seat_baseline, epsilon);
        std::cout << "Epoch " << epoch << " baseline=" << moving_baseline << "\n";
    }

    dlib::serialize("policy_rl.dat") << net;
    std::cout << "Saved to policy_rl.dat\n";
    return 0;
}
