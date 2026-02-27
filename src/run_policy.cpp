// run_policy.cpp — Texas Hold'em vs Neural Bot (FTXUI terminal UI)
//
// Default   : interactive — play as seat 0 until bust
// --bench N : run N-hand statistical benchmark vs random players

#ifdef _MSC_VER
#  pragma warning(push, 0)
#endif
#include <ftxui/component/component.hpp>
#include <ftxui/component/event.hpp>
#include <ftxui/component/screen_interactive.hpp>
#include <ftxui/dom/elements.hpp>
#include <ftxui/screen/color.hpp>
#ifdef _MSC_VER
#  pragma warning(pop)
#endif

#include <dlib/dnn.h>
#include <BS_thread_pool.hpp>

#include <cmath>
#include <iostream>
#include <sstream>
#include <string>
#include <deque>
#include <algorithm>
#include <thread>
#include <chrono>
#include <iomanip>

#include "../include/neural_network/dlib_policy.hpp"
#include "../include/neural_network/rl_featurizer.hpp"
#include "../include/neural_network/rl_actions.hpp"
#include "../include/game/game.hpp"
#include "../include/game/blinds.hpp"
#include "../include/hand.hpp"

// ─────────────────────────────── RNG ──────────────────────────────────────────
struct GameRng
{
    using result_type = uint64_t;
    omp::XoroShiro128Plus eng;
    explicit GameRng(uint64_t seed = 0)
        : eng(seed ? seed : static_cast<uint64_t>(std::random_device{}())) {}
    static constexpr result_type min() { return 0ULL; }
    static constexpr result_type max() { return ~0ULL; }
    result_type operator()() { return eng(); }
};

// ────────────────────── Seat constants ────────────────────────────────────────
static constexpr std::size_t kHuman = 0;
static constexpr std::size_t kBot   = 1;

// ────────────────────── Card / deck helpers ────────────────────────────────────
static std::wstring suit_symbol(Suit s) noexcept
{
    switch (s)
    {
    case Suit::Hearts:   return L"\u2665";
    case Suit::Diamonds: return L"\u2666";
    case Suit::Clubs:    return L"\u2663";
    case Suit::Spades:   return L"\u2660";
    default:             return L"?";
    }
}

static bool is_red(const Card &c) noexcept
{
    return c.getSuit() == Suit::Hearts || c.getSuit() == Suit::Diamonds;
}

// "A♠", "10♥", " 2♣"  — always 3 display columns
static std::wstring fmt_card(const Card &c)
{
    std::wostringstream os;
    os << c.getRank();
    std::wstring r = os.str();
    if (r.size() == 1) r = L" " + r;
    return r + suit_symbol(c.getSuit());
}

// ────────────────── FTXUI card widgets ────────────────────────────────────────
static ftxui::Element card_widget(const Card &c)
{
    auto t = ftxui::text(fmt_card(c));
    t = t | (is_red(c) ? ftxui::color(ftxui::Color::Red)
                       : ftxui::color(ftxui::Color::White));
    return t | ftxui::border;
}

static ftxui::Element hidden_card_widget()
{
    return ftxui::text(L" ??") | ftxui::dim | ftxui::border;
}

static ftxui::Element hand_widget(const Deck &d)
{
    if (d.size() == 0)
        return ftxui::text(L"[none]") | ftxui::dim;
    std::vector<ftxui::Element> elems;
    for (const auto &c : d)
    {
        if (!elems.empty()) elems.push_back(ftxui::text(L" "));
        elems.push_back(card_widget(c));
    }
    return ftxui::hbox(std::move(elems));
}

static ftxui::Element hidden_hand_widget(std::size_t n)
{
    std::vector<ftxui::Element> elems;
    for (std::size_t i = 0; i < n; ++i)
    {
        if (i > 0) elems.push_back(ftxui::text(L" "));
        elems.push_back(hidden_card_widget());
    }
    return ftxui::hbox(std::move(elems));
}

// ──────────────────────── Bet helpers ─────────────────────────────────────────
static uint32_t half_pot_target(const BetData &bd) noexcept
{
    uint32_t p = std::max<uint32_t>(1, bd.pot);
    return (bd.currentBet == 0)
        ? std::max<uint32_t>(bd.minRaise, p / 2)
        : std::max<uint32_t>(bd.currentBet + bd.minRaise, bd.currentBet + p / 2);
}

static uint32_t full_pot_target(const BetData &bd) noexcept
{
    uint32_t p = std::max<uint32_t>(1, bd.pot);
    return (bd.currentBet == 0)
        ? std::max<uint32_t>(bd.minRaise, p)
        : std::max<uint32_t>(bd.currentBet + bd.minRaise, bd.currentBet + p);
}

static uint32_t action_cost(unsigned a, const BetData &bd, const Player &p) noexcept
{
    switch (a)
    {
    case A_Fold:       return 0;
    case A_CheckCall:  return std::min((bd.currentBet > p.committed) ? bd.currentBet - p.committed : 0u, p.chips);
    case A_BetHalfPot: { uint32_t t = half_pot_target(bd); return (t > p.committed) ? std::min(t - p.committed, p.chips) : 0; }
    case A_BetPot:     { uint32_t t = full_pot_target(bd);  return (t > p.committed) ? std::min(t - p.committed, p.chips) : 0; }
    default:           return p.chips;
    }
}

static std::wstring action_label(unsigned a, const BetData &bd, const Player &p)
{
    uint32_t cost = action_cost(a, bd, p);
    switch (a)
    {
    case A_Fold:       return L"Fold";
    case A_CheckCall:  return cost ? L"Call " + std::to_wstring(cost) : L"Check";
    case A_BetHalfPot: return L"Bet " + std::to_wstring(cost) + L" (\u00bdpot)";
    case A_BetPot:     return L"Bet " + std::to_wstring(cost) + L" (pot)";
    default:           return L"All-in (" + std::to_wstring(p.chips) + L")";
    }
}

static const wchar_t *street_name(GameState s) noexcept
{
    switch (s)
    {
    case GameState::PreFlop: return L"Pre-Flop";
    case GameState::Flop:    return L"Flop";
    case GameState::Turn:    return L"Turn";
    case GameState::River:   return L"River";
    default:                 return L"---";
    }
}

// ──────────────────────── UI State ────────────────────────────────────────────
enum class Phase { HumanTurn, HandOver, GameOver };

struct UIState
{
    int     hand_num   = 0;
    int64_t net_bb     = 0;
    int32_t hand_result = 0;       // chip delta for human (set in HandOver)

    std::wstring your_class;       // e.g. L"Full House"  — empty until hand ends
    std::wstring bot_class;        // e.g. L"Flush"       — empty until hand ends
    float        equity = -1.f;   // Monte Carlo win probability [0,1]; -1 = not yet computed

    std::vector<unsigned>    legal;
    std::vector<std::wstring> alabel;

    std::deque<std::wstring> log;
    static constexpr std::size_t kMaxLog = 9;

    GameState last_street = GameState::PreFlop;
    Phase     phase       = Phase::HumanTurn;

    void add_log(const std::wstring &s)
    {
        log.push_back(s);
        while (log.size() > kMaxLog) log.pop_front();
    }
};

// ──────────────────── Interactive mode ────────────────────────────────────────
static void play_interactive(policy_net &net, const Blinds &blinds, int starting_chips)
{
    using namespace ftxui;

    Game g(blinds);
    g.addPlayer(static_cast<uint32_t>(starting_chips));
    g.addPlayer(static_cast<uint32_t>(starting_chips));

    GameRng rng;
    BS::thread_pool<BS::tp::none> pool(std::max(1u, std::thread::hardware_concurrency()));

    UIState ui;
    uint32_t human_before = 0;
    uint32_t bot_before   = 0;

    // ── advance: run bot turns and fast-forwards until it's human's turn
    //    or the hand ends.
    auto advance = [&]()
    {
        while (g.state() != GameState::Finished)
        {
            GameState cur = g.state();

            // Street change → log it
            if (cur != ui.last_street)
            {
                ui.last_street = cur;
                ui.add_log(L"--- " + std::wstring(street_name(cur)) + L" ---");
            }

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

            if (cur_id == kBot)
            {
                auto s    = featurize(g, kBot, blinds, pool);
                unsigned aidx = policy_greedy(net, s, leg);
                ui.add_log(L"Bot \u25b6 " + action_label(aidx, g.betData(), g.players()[kBot]));
                g.applyAction(rng, to_engine_action(aidx, g, kBot, blinds));
            }
            else
            {
                // Human's turn — populate menu and stop
                const auto &bd = g.betData();
                const auto &hp = g.players()[kHuman];
                ui.legal.clear();
                ui.alabel.clear();
                for (unsigned a : leg)
                {
                    ui.legal.push_back(a);
                    ui.alabel.push_back(action_label(a, bd, hp));
                }
                // Compute equity (once per decision point, not every render frame)
                ui.equity = static_cast<float>(probabilityOfWinning(
                    hp.hole, g.board(), 5000, g.players().size(), pool));
                ui.phase = Phase::HumanTurn;
                return;
            }
        }

        // Hand is over
        const auto &fps = g.players();
        int32_t hd = static_cast<int32_t>(fps[kHuman].chips) - static_cast<int32_t>(human_before);
        int32_t bd_d = static_cast<int32_t>(fps[kBot].chips)   - static_cast<int32_t>(bot_before);
        ui.hand_result = hd;
        ui.net_bb += hd / static_cast<int32_t>(blinds.bigBlind);

        // Classify both hands if the board ran out far enough
        if (g.board().size() >= 3)
        {
            auto classify_to_wstr = [&](const Deck &hole) -> std::wstring {
                auto r = Hand::classify(Deck::createDeck({hole, g.board()}));
                std::wostringstream os;
                os << r.getClassification();
                return os.str();
            };
            if (fps[kHuman].alive())
            {
                ui.your_class = classify_to_wstr(fps[kHuman].hole);
                ui.add_log(L"You: " + ui.your_class);
            }
            if (fps[kBot].alive())
            {
                ui.bot_class = classify_to_wstr(fps[kBot].hole);
                ui.add_log(L"Bot: " + ui.bot_class);
            }
        }

        std::wstring res = L"You " + (hd >= 0 ? std::wstring(L"+") : L"") + std::to_wstring(hd)
                         + L"  Bot " + (bd_d >= 0 ? std::wstring(L"+") : L"") + std::to_wstring(bd_d);
        ui.add_log(L">> " + res);

        ui.phase = (fps[kHuman].chips < blinds.bigBlind) ? Phase::GameOver : Phase::HandOver;
    };

    // ── start_hand: setup then call advance
    auto start_hand = [&]()
    {
        if (g.players()[kBot].chips < blinds.bigBlind)
            g.mutablePlayers()[kBot].chips = static_cast<uint32_t>(starting_chips);

        ++ui.hand_num;
        human_before   = g.players()[kHuman].chips;
        bot_before     = g.players()[kBot].chips;
        ui.last_street = GameState::PreDeal;  // force street log on first advance call
        ui.legal.clear();
        ui.alabel.clear();
        ui.your_class.clear();
        ui.bot_class.clear();
        ui.equity = -1.f;
        ui.add_log(L"=== Hand #" + std::to_wstring(ui.hand_num) + L" ===");

        g.startNewHand(rng);
        advance();
    };

    // ── render: build the FTXUI DOM from current game + ui state
    auto render_fn = [&]() -> Element
    {
        const auto &hp = g.players()[kHuman];
        const auto &bp = g.players()[kBot];
        const auto &bd = g.betData();

        uint32_t to_call = (bd.currentBet > hp.committed)
                         ? (bd.currentBet - hp.committed) : 0;

        bool reveal_bot = (ui.phase == Phase::HandOver || ui.phase == Phase::GameOver);

        // ── Left panel: hands ─────────────────────────────────────────────
        std::vector<Element> your_items = {text(L"YOUR HAND") | bold, hand_widget(hp.hole)};
        if (g.board().size() >= 3 && hp.alive())
        {
            auto r = Hand::classify(Deck::createDeck({hp.hole, g.board()}));
            std::wostringstream wos;
            wos << r.getClassification();
            your_items.push_back(text(wos.str()) | bold | color(Color::Yellow));
        }
        else if (!ui.your_class.empty())
            your_items.push_back(text(ui.your_class) | bold | color(Color::Yellow));
        if (ui.equity >= 0.f)
        {
            int pct = static_cast<int>(std::round(ui.equity * 100.f));
            auto eq_col = (pct >= 60) ? Color::Green
                        : (pct <= 40) ? Color::Red
                                      : Color::Yellow;
            your_items.push_back(
                text(L"Win: " + std::to_wstring(pct) + L"%") | color(eq_col));
        }
        auto your_hand_panel = vbox(std::move(your_items));

        std::vector<Element> bot_items = {
            text("BOT") | bold,
            reveal_bot ? hand_widget(bp.hole) : hidden_hand_widget(2),
        };
        if (reveal_bot && !ui.bot_class.empty())
            bot_items.push_back(text(ui.bot_class) | bold | color(Color::Yellow));
        auto bot_panel = vbox(std::move(bot_items));

        auto left_col = vbox({
            your_hand_panel,
            separator(),
            bot_panel,
        }) | border | flex;

        // ── Right panel: board + info ─────────────────────────────────────
        GameState disp_state = (g.state() == GameState::Finished)
                             ? ui.last_street : g.state();
        auto board_label = text(std::wstring(L"BOARD  [") + street_name(disp_state) + L"]") | bold;

        auto board_row = g.board().size() > 0
            ? hand_widget(g.board())
            : (text(L"[Pre-Flop, no board]") | dim);

        std::wstring nb_sign = ui.net_bb >= 0 ? L"+" : L"";
        auto nb_text = text(nb_sign + std::to_wstring(ui.net_bb) + L" BB")
                     | (ui.net_bb >= 0 ? color(Color::Green) : color(Color::Red));

        auto info_col = vbox({
            hbox({text(L"Pot: ") | bold, text(std::to_wstring(bd.pot))}),
            to_call > 0
                ? hbox({text(L"To call: ") | bold, text(std::to_wstring(to_call))})
                : (text(L"No bet facing") | dim),
            separator(),
            hbox({text(L"You:  ") | bold, text(std::to_wstring(hp.chips))}),
            hbox({text(L"Bot:  ") | bold, text(std::to_wstring(bp.chips))}),
            separator(),
            hbox({text(L"Net:  "), nb_text}),
            text(L"Hand #" + std::to_wstring(ui.hand_num)) | dim,
        });

        auto right_col = vbox({
            board_label,
            board_row,
            separator(),
            info_col,
        }) | border | flex;

        auto cards_section = hbox({left_col, right_col});

        // ── Action log ────────────────────────────────────────────────────
        std::vector<Element> log_elems;
        for (const auto &line : ui.log)
            log_elems.push_back(text(L"  " + line));
        if (log_elems.empty())
            log_elems.push_back(text(L"  ...") | dim);
        auto log_section = vbox(std::move(log_elems)) | border;

        // ── Action bar ────────────────────────────────────────────────────
        Element action_bar;
        if (ui.phase == Phase::HumanTurn)
        {
            std::vector<Element> btns;
            for (std::size_t i = 0; i < ui.legal.size(); ++i)
                btns.push_back(text(L" [" + std::to_wstring(i + 1) + L"] " + ui.alabel[i] + L" ") | border);
            btns.push_back(text(L" [q] Quit ") | border | dim);
            action_bar = hbox(std::move(btns)) | hcenter;
        }
        else if (ui.phase == Phase::HandOver)
        {
            auto col   = ui.hand_result >= 0 ? color(Color::Green) : color(Color::Red);
            std::wstring rsign = ui.hand_result >= 0 ? L"+" : L"";
            action_bar = hbox({
                text(L" Result: " + rsign + std::to_wstring(ui.hand_result) + L" chips ") | col | bold,
                text(L" \u2502 ") | dim,
                text(L" [any key] Next hand   [q] Quit ") | bold,
            }) | border | hcenter;
        }
        else
        {
            action_bar = text(L"  \u2620  You're busted!  [q] to exit  \u2620  ")
                       | border | hcenter | bold | color(Color::Red);
        }

        // ── Title ─────────────────────────────────────────────────────────
        auto title = text(L" \u2660\u2665 Texas Hold'em: You (seat 0) vs Neural Bot (seat 1) \u2666\u2663 ")
                   | bold | hcenter;

        return vbox({title, cards_section, log_section, action_bar}) | border;
    };

    // ── FTXUI event loop ──────────────────────────────────────────────────
    auto screen = ScreenInteractive::Fullscreen();

    auto component = Renderer(render_fn)
        | CatchEvent([&](Event event) -> bool
        {
            if (!event.is_character()) return false;
            const std::string &ch = event.character();

            if (ch == "q" || ch == "Q")
            {
                screen.Exit();
                return true;
            }

            if (ui.phase == Phase::HumanTurn && ch.size() == 1)
            {
                int i = ch[0] - '1';
                if (i >= 0 && static_cast<std::size_t>(i) < ui.legal.size())
                {
                    unsigned chosen = ui.legal[static_cast<std::size_t>(i)];
                    ui.add_log(L"You \u25b6 " + ui.alabel[static_cast<std::size_t>(i)]);
                    g.applyAction(rng, to_engine_action(chosen, g, kHuman, blinds));
                    advance();
                    return true;
                }
                return false;
            }

            if (ui.phase == Phase::HandOver)
            {
                start_hand();
                return true;
            }

            if (ui.phase == Phase::GameOver)
            {
                screen.Exit();
                return true;
            }

            return false;
        });

    start_hand();
    screen.Loop(component);
}

// ──────────────────── Benchmark mode (Neural vs Random) ───────────────────────
static void run_benchmark(policy_net &net, const Blinds &blinds,
                          int n_players, int chips, int hands)
{
    Game g(blinds);
    for (int i = 0; i < n_players; ++i)
        g.addPlayer(static_cast<uint32_t>(chips));

    GameRng rng(42);
    BS::thread_pool<BS::tp::none> pool(std::max(1u, std::thread::hardware_concurrency()));

    const std::size_t bot_id = g.players()[0].id;

    struct Stats { int64_t profit_bb = 0; int won = 0; int played = 0; };
    std::vector<Stats> stats(static_cast<std::size_t>(n_players));
    uint64_t action_hist[kNumActions] = {};

    std::cout << "\n=== BENCHMARK: Neural (seat 0) vs " << (n_players - 1) << " Random ===\n"
              << "Hands: " << hands << "  Players: " << n_players << "\n"
              << std::string(50, '-') << "\n";

    auto t0 = std::chrono::steady_clock::now();

    for (int h = 0; h < hands; ++h)
    {
        bool needs_reset = false;
        for (const auto &p : g.players())
            if (p.chips < blinds.bigBlind * 3) { needs_reset = true; break; }
        if (needs_reset) g.resetPlayerChips(static_cast<uint32_t>(chips));

        std::vector<uint32_t> before(static_cast<std::size_t>(n_players));
        for (const auto &p : g.players()) before[p.id] = p.chips;

        g.startNewHand(rng);

        while (g.state() != GameState::Finished)
        {
            if (!g.hasCurrentActor())
            {
                g.applyAction(rng, ActionStruct{ActionType::Check, 0});
                continue;
            }
            const std::size_t cur = g.currentPlayer().id;
            auto leg = legal_actions(g, cur, blinds);
            if (leg.empty())
            {
                g.applyAction(rng, ActionStruct{ActionType::Fold, 0});
                continue;
            }
            unsigned aidx;
            if (cur == bot_id)
            {
                auto s = featurize(g, cur, blinds, pool);
                aidx = policy_greedy(net, s, leg);
                action_hist[aidx]++;
            }
            else
            {
                static thread_local omp::XoroShiro128Plus xrng{std::random_device{}()};
                omp::FastUniformIntDistribution<std::size_t> D(0, leg.size() - 1);
                aidx = leg[D(xrng)];
            }
            g.applyAction(rng, to_engine_action(aidx, g, cur, blinds));
        }

        for (int i = 0; i < n_players; ++i)
        {
            int32_t d = static_cast<int32_t>(g.players()[static_cast<std::size_t>(i)].chips)
                      - static_cast<int32_t>(before[static_cast<std::size_t>(i)]);
            stats[static_cast<std::size_t>(i)].profit_bb += d / static_cast<int32_t>(blinds.bigBlind);
            stats[static_cast<std::size_t>(i)].played++;
            if (d > 0) stats[static_cast<std::size_t>(i)].won++;
        }
        if (hands >= 10 && (h + 1) % (hands / 10) == 0)
            std::cout << "." << std::flush;
    }

    double elapsed = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - t0).count();

    std::cout << "\n\nTime: " << std::fixed << std::setprecision(2) << elapsed
              << "s  (" << static_cast<int>(hands / elapsed) << " hands/s)\n\n";

    std::cout << std::left
              << std::setw(10) << "Seat"
              << std::setw(12) << "BB/100"
              << std::setw(12) << "Win%"
              << "Total BB\n"
              << std::string(46, '-') << "\n";

    for (int i = 0; i < n_players; ++i)
    {
        auto &st  = stats[static_cast<std::size_t>(i)];
        double bb100 = 100.0 * static_cast<double>(st.profit_bb) / st.played;
        double wr    = 100.0 * static_cast<double>(st.won)        / st.played;
        std::cout << std::left  << std::setw(10) << ("Seat " + std::to_string(i))
                  << std::fixed << std::setprecision(1)
                  << std::setw(12) << bb100
                  << std::setw(12) << wr
                  << st.profit_bb << "\n";
    }

    std::cout << "\nNeural bot action distribution:\n";
    static const char *kName[kNumActions] = {"Fold", "Check/Call", "Bet\u00bdpot", "BetPot", "All-in"};
    uint64_t total = 0;
    for (auto v : action_hist) total += v;
    for (int i = 0; i < kNumActions; ++i)
    {
        double pct = (total > 0) ? 100.0 * static_cast<double>(action_hist[i]) / static_cast<double>(total) : 0.0;
        std::cout << "  " << std::left << std::setw(12) << kName[i]
                  << std::fixed << std::setprecision(1) << pct << "%\n";
    }
}

// ─────────────────────────────── main ─────────────────────────────────────────
int main(int argc, char **argv)
{
    Blinds blinds{50, 100};
    std::string model_file  = "policy_best.dat";
    bool bench_mode         = false;
    int  bench_hands        = 10'000;
    int  bench_players      = 3;
    int  bench_chips        = 10'000;

    for (int i = 1; i < argc; ++i)
    {
        std::string arg = argv[i];
        if (arg == "--bench" && i + 1 < argc)
        {
            bench_mode  = true;
            bench_hands = std::max(1, std::atoi(argv[++i]));
        }
        else if (arg == "--players" && i + 1 < argc)
            bench_players = std::max(2, std::atoi(argv[++i]));
        else if (arg == "--chips" && i + 1 < argc)
            bench_chips = std::max(100, std::atoi(argv[++i]));
        else
            model_file = arg;
    }

    policy_net net;
    try
    {
        dlib::deserialize(model_file) >> net;
        std::cout << "Model: " << model_file << "\n";
    }
    catch (const std::exception &e)
    {
        std::cerr << "ERROR loading " << model_file << ": " << e.what() << "\n"
                  << "Train a model first: ./Poker_Trainer\n";
        return 1;
    }

    if (bench_mode)
        run_benchmark(net, blinds, bench_players, bench_chips, bench_hands);
    else
        play_interactive(net, blinds, 10'000);

    return 0;
}
