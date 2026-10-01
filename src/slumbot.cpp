// Plays heads-up against Slumbot (slumbot.com) through its public API: the 200bb blueprint preflop,
// resolving after the flop. One hand at a time; results go to a CSV, one line per hand.
#include "../include/cfr/gpu_subgame_solver.hpp"
#include "../include/cfr/tools.hpp"
#include <chrono>
#include <cmath>
#include <drogon/HttpClient.h>
#include <format>
#include <fstream>
#include <glaze/glaze.hpp>
#include <iostream>
#include <map>
#include <mutex>
#include <memory>
#include <random>
#include <string>
#include <thread>
#include <trantor/net/EventLoopThread.h>

struct SlumbotRequest
{
    std::optional<std::string> token;
    std::optional<std::string> incr;
};

struct SlumbotReply
{
    std::string action;
    std::string token;
    int client_pos = -1;
    std::vector<std::string> hole_cards;
    std::vector<std::string> board;
    std::optional<double> winnings;
    std::optional<std::vector<std::string>> bot_hole_cards; // at showdown
    std::optional<std::string> error_msg;
};

static std::string join(const std::vector<std::string> &cards)
{
    std::string out;
    for (const auto &c : cards)
    {
        out += c;
    }
    return out;
}

// Synchronous JSON POSTs to Slumbot, retried on network errors.
class SlumbotClient
{
public:
    SlumbotClient()
    {
        m_loop.run();
        m_client = drogon::HttpClient::newHttpClient("https://slumbot.com", m_loop.getLoop());
    }

    std::optional<SlumbotReply> post(const std::string &endpoint, const SlumbotRequest &request)
    {
        std::string body;
        if (glz::write_json(request, body))
        {
            return std::nullopt;
        }
        for (int attempt = 0; attempt < 5; ++attempt)
        {
            auto req = drogon::HttpRequest::newHttpRequest();
            req->setMethod(drogon::Post);
            req->setPath("/slumbot/api/" + endpoint);
            req->setContentTypeCode(drogon::CT_APPLICATION_JSON);
            req->setBody(body);
            const auto [result, response] = m_client->sendRequest(req, 30.0);
            if (result != drogon::ReqResult::Ok || response == nullptr)
            {
                std::cerr << "request failed (" << drogon::to_string_view(result) << "), retrying\n";
                std::this_thread::sleep_for(std::chrono::seconds(1 << attempt));
                continue;
            }
            SlumbotReply reply;
            if (const auto error = glz::read<glz::opts{.error_on_unknown_keys = false}>(reply, response->body()))
            {
                std::cerr << "bad reply: " << glz::format_error(error, response->body()) << '\n';
                return std::nullopt;
            }
            return reply;
        }
        return std::nullopt;
    }

private:
    trantor::EventLoopThread m_loop;
    drogon::HttpClientPtr m_client;
};

static std::uint64_t card(const std::string &text) { return Deck::parseHand(text).getMask(); }

// Plays --hands hands over --sessions concurrent sessions (each with its own connection, token and bot;
// they share the preflop strategy, the GPU and the results file).
template <template <typename> class Resolver>
static int play(const PreflopStrategy<Blueprint200> &preflop, std::map<std::string, double> &options, const std::string &log)
{
    const SlumbotResolves resolves{static_cast<std::size_t>(options["--flop-iterations"]), static_cast<std::size_t>(options["--turn-iterations"]),
                                   static_cast<std::size_t>(options["--river-iterations"]), static_cast<std::size_t>(options["--flop-samples"])};
    const auto hands = static_cast<std::uint64_t>(options["--hands"]);
    std::mutex mutex; // guards everything below
    std::ofstream out(log, std::ios::app);
    double sum = 0.0, squares = 0.0;
    std::uint64_t started = 0, played = 0;
    bool failed = false;
    const auto start = std::chrono::steady_clock::now();
    const auto fail = [&](const std::string &message)
    {
        const std::lock_guard lock(mutex);
        failed = true;
        std::cerr << message << '\n';
    };
    const auto session = [&](std::uint64_t seed)
    {
        SlumbotBot<Blueprint200Config, Resolver> bot(preflop, resolves, seed);
        SlumbotClient client;
        std::optional<std::string> token;
        while (true)
        {
            {
                const std::lock_guard lock(mutex);
                if (failed || started == hands)
                {
                    return;
                }
                ++started;
            }
            auto reply = client.post("new_hand", {token, std::nullopt});
            bool dealt = false;
            std::string decisions; // ours this hand: options as incr:probability, the one taken starred
            while (reply.has_value() && !reply->error_msg.has_value() && !reply->winnings.has_value())
            {
                if (!reply->token.empty())
                {
                    token = reply->token;
                }
                if (!dealt)
                {
                    const std::size_t seat = reply->client_pos == 0 ? 1 : 0; // Slumbot's 0 is the big blind
                    bot.newHand(seat, card(reply->hole_cards.at(0) + " " + reply->hole_cards.at(1)));
                    dealt = true;
                }
                const auto actions = parseSlumbotActions(reply->action);
                if (!actions.has_value())
                {
                    return fail("cannot parse action " + reply->action);
                }
                std::vector<std::uint64_t> board;
                for (const auto &c : reply->board)
                {
                    board.push_back(card(c));
                }
                const auto incr = bot.update(*actions, board);
                if (!incr.has_value())
                {
                    return fail("not our turn after " + reply->action);
                }
                decisions += decisions.empty() ? "" : ";";
                for (std::size_t a = 0; a < bot.decision().options.size(); ++a)
                {
                    const auto &option = bot.decision().options[a];
                    decisions += std::format("{}{}:{:.4f}{}", a > 0 ? "|" : "", option.incr, option.probability, a == bot.decision().chosen ? "*" : "");
                }
                reply = client.post("act", {token, incr});
            }
            if (!reply.has_value() || reply->error_msg.has_value())
            {
                return fail("hand failed: " + (reply.has_value() ? *reply->error_msg : std::string("no reply")));
            }
            if (!reply->token.empty())
            {
                token = reply->token;
            }
            const std::lock_guard lock(mutex);
            const double mbb = *reply->winnings / 100.0 * 1000.0;
            sum += mbb;
            squares += mbb * mbb;
            ++played;
            out << reply->client_pos << ',' << reply->action << ',' << *reply->winnings << ',' << join(reply->hole_cards) << ','
                << join(reply->board) << ',' << join(reply->bot_hole_cards.value_or(std::vector<std::string>{})) << ',' << bot.nudges() << ','
                << (dealt ? std::to_string(bot.handSeed()) : std::string()) << ',' << decisions << '\n'; // not dealt: Slumbot folded first
            out.flush();
            if (played % 10 == 0)
            {
                const double n = static_cast<double>(played), mean = sum / n;
                const double error = 1.96 * std::sqrt((std::max)(squares / n - mean * mean, 0.0) / n);
                const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
                std::cout << played << " hands: " << mean << " +/- " << error << " mbb/hand (" << seconds / n << " s/hand)" << std::endl;
            }
        }
    };
    {
        std::random_device seeds;
        std::vector<std::jthread> threads;
        for (int i = 0; i < static_cast<int>(options["--sessions"]); ++i)
        {
            threads.emplace_back(session, (std::uint64_t{seeds()} << 32) | seeds());
        }
    }
    return failed ? 1 : 0;
}

int main(int argc, char **argv)
{
    std::map<std::string, double> options{{"--hands", 100.0},       {"--flop-iterations", 60.0}, {"--turn-iterations", 60.0},
                                          {"--river-iterations", 100.0}, {"--flop-samples", 6.0},    {"--gpu", 1.0},
                                          {"--sessions", 1.0}};
    if (!parseOptions(argc, argv, 4, options))
    {
        std::cerr << "usage: " << argv[0] << " <abstraction.bin> <blueprint200.bin> <results.csv>" << optionsUsage(options)
                  << "\n  plays --hands hands against Slumbot over --sessions concurrent sessions (keep it small: it is a free\n"
                     "  service), appending client_pos,action,winnings,hole,board,slumbot_hole,nudges,seed,decisions per hand to\n"
                     "  results.csv (decisions: ours in order, ';' apart, each option as incr:probability, the one taken starred)\n";
        return 1;
    }
    const auto abstraction = loadAbstraction(argv[1]);
    const auto preflop = abstraction != nullptr ? loadPreflop(*abstraction, argv[2]) : nullptr;
    if (preflop == nullptr)
    {
        return 1;
    }
    if (options["--gpu"] != 0.0)
    {
        if (Gpu::instance() == nullptr)
        {
            std::cerr << "no OpenCL GPU (run with --gpu 0)\n";
            return 1;
        }
        return play<GpuSubgameSolver>(*preflop, options, argv[3]);
    }
    return play<SubgameSolver>(*preflop, options, argv[3]);
}
