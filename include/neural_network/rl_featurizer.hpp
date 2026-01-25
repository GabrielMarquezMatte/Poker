#ifndef __POKER_RL_FEATURIZER_HPP__
#define __POKER_RL_FEATURIZER_HPP__
#include <dlib/matrix.h>
#include <span>
#include "../game/game.hpp"
#include "../game.hpp"

inline dlib::matrix<float> featurize(omp::XoroShiro128Plus &rng, const Game &g, std::size_t heroIdx, const Blinds &blinds)
{
    const auto bb = std::max<std::uint32_t>(1, blinds.bigBlind);
    auto const &ps = g.players();
    const auto &hero = ps[heroIdx];
    const auto &bd = g.betData();

    // counts
    int alive = 0, elig = 0;
    for (auto const &p : ps)
    {
        if (p.alive())
            ++alive;
        if (p.eligible())
            ++elig;
    }

    // to_call
    std::uint32_t to_call = (bd.currentBet > hero.committed) ? (bd.currentBet - hero.committed) : 0;

    // street
    float street_one_hot[4] = {0, 0, 0, 0};
    switch (g.state())
    {
    case GameState::PreFlop:
        street_one_hot[0] = 1;
        break;
    case GameState::Flop:
        street_one_hot[1] = 1;
        break;
    case GameState::Turn:
        street_one_hot[2] = 1;
        break;
    case GameState::River:
        street_one_hot[3] = 1;
        break;
    default:
        break;
    }

    dlib::matrix<float> x(24, 1);
    x = 0;
    int k = 0;
    for (int i = 0; i < 4; ++i)
        x(k++) = street_one_hot[i];

    x(k++) = (to_call == 0 ? 1.f : 0.f);        // can_check
    x(k++) = float(to_call) / float(bb);        // to_call_bb
    x(k++) = float(bd.pot) / float(bb);         // pot_bb
    x(k++) = float(bd.currentBet) / float(bb);  // currentBet_bb
    x(k++) = float(bd.minRaise) / float(bb);    // minRaise_bb
    x(k++) = float(hero.chips) / float(bb);     // hero_stack_bb
    x(k++) = float(hero.committed) / float(bb); // hero_committed_bb
    x(k++) = float(alive);
    x(k++) = float(elig);
    x(k++) = probabilityOfWinning(rng, hero.hole, g.board(), 50'000, ps.size() - 1);
    int board_cards = 0;
    if (g.state() == GameState::Flop)
    {
        board_cards = 3;
    }
    else if (g.state() == GameState::Turn)
    {
        board_cards = 4;
    }
    else if (g.state() == GameState::River)
    {
        board_cards = 5;
    }

    x(k++) = float(board_cards == 0);
    x(k++) = float(board_cards == 3);
    x(k++) = float(board_cards == 4);
    x(k++) = float(board_cards == 5);

    // padding to 24
    while (k < 24)
        x(k++) = 0.f;
    return x;
}

#endif // __POKER_RL_FEATURIZER_HPP__