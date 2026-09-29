#include <expected>
#include <ranges>
#include "../include/game.hpp"
#include <BS_thread_pool.hpp>
#include <glaze/glaze.hpp>
#include <drogon/drogon.h>
template <>
struct glz::from<glz::JSON, Deck>
{
    template <auto Opts>
    static void op(Deck &value, glz::is_context auto &&ctx, auto &&it, auto &&end)
    {
        std::string_view str_value;
        glz::parse<JSON>::op<Opts>(str_value, ctx, it, end);
        if (bool(ctx.error))
            return;
        value = Deck::parseHand(str_value);
        const auto tokens = std::ranges::count_if(str_value | std::views::split(' '), [](auto &&token)
                                                  { return !std::ranges::empty(token); });
        if (static_cast<std::size_t>(tokens) != value.size())
        {
            ctx.error = glz::error_code::constraint_violated;
            ctx.custom_error_message = "cards must be distinct and written like \"Ah Td\"";
        }
    }
};
struct ProbabilitiesRequest
{
    Deck hand;
    Deck table;
    std::size_t numPlayers = 2;
    std::size_t numSimulations = 1'000'000;
    struct glaze
    {
        using T = ProbabilitiesRequest;
        static constexpr auto limitNumSimulations = [](const T &, std::size_t numSimulations) -> bool
        {
            return numSimulations > 0 && numSimulations <= 100'000'000;
        };
        static constexpr auto limitNumPlayers = [](const T &, std::size_t numPlayers) -> bool
        {
            return numPlayers >= 2 && numPlayers <= 10;
        };
        static constexpr auto value = glz::object(
            "hand", &T::hand,
            "table", &T::table,
            "numPlayers", glz::read_constraint<&T::numPlayers, limitNumPlayers, "numPlayers must be between 2 and 10">,
            "numSimulations", glz::read_constraint<&T::numSimulations, limitNumSimulations, "numSimulations must be between 1 and 100,000,000">);
    };
};

struct ErrorResponse
{
    std::string_view error;
};

struct ProbabilityResult
{
    double probability; 
    double equity;      
};

struct StatisticsResult
{
    std::size_t wins;
    std::size_t losses;
    std::size_t ties;
    double equity;
};

using Callback = std::function<void(const drogon::HttpResponsePtr &)>;

static void respond(const Callback &callback, drogon::HttpStatusCode status, const auto &body)
{
    auto json = glz::write_json(body);
    auto resp = drogon::HttpResponse::newHttpResponse();
    if (!json)
    {
        resp->setStatusCode(drogon::HttpStatusCode::k500InternalServerError);
        callback(resp);
        return;
    }
    resp->setStatusCode(status);
    resp->setContentTypeCode(drogon::CT_APPLICATION_JSON);
    resp->setBody(std::move(*json));
    callback(resp);
}

struct Pools
{
    BS::thread_pool<BS::tp::none> compute{std::thread::hardware_concurrency()};
    BS::thread_pool<BS::tp::none> requests{4};
};

template <typename TToResponse>
static void handleRequest(const drogon::HttpRequestPtr &req, Callback &&callback, Pools &pools, TToResponse toResponse)
{
    std::string_view body = req->getBody();
    ProbabilitiesRequest r;
    auto error = glz::read_json(r, body);
    if (error)
    {
        respond(callback, drogon::HttpStatusCode::k400BadRequest, ErrorResponse{glz::format_error(error, body)});
        return;
    }
    if (r.hand.size() != 2 || r.table.size() > 5 || (r.hand.getMask() & r.table.getMask()) != 0)
    {
        respond(callback, drogon::HttpStatusCode::k400BadRequest, ErrorResponse{"hand must have 2 cards, table at most 5, and no card may appear in both"});
        return;
    }
    if (const auto quick = quickGameStatistics(r.hand, r.table, r.numSimulations, r.numPlayers))
    {
        respond(callback, drogon::HttpStatusCode::k200OK, toResponse(*quick));
        return;
    }
    pools.requests.detach_task([callback = std::move(callback), r, toResponse, &pools]()
                               {
        const GameStatistics stats = computeRandomGameStatistics(r.hand, r.table, r.numSimulations, r.numPlayers, pools.compute);
        respond(callback, drogon::HttpStatusCode::k200OK, toResponse(stats)); });
}

int main()
{
    auto &app = drogon::app();
    Pools pools;
    app.registerHandler("/statistics", [&](const drogon::HttpRequestPtr &req, Callback &&callback)
                        { handleRequest(req, std::move(callback), pools, [](const GameStatistics &s)
                                        { return StatisticsResult{s.wins, s.losses, s.ties, s.equity()}; }); }, {drogon::Post});
    app.registerHandler("/probabilities", [&](const drogon::HttpRequestPtr &req, Callback &&callback)
                        { handleRequest(req, std::move(callback), pools, [](const GameStatistics &s)
                                        { return ProbabilityResult{s.notLosing(), s.equity()}; }); }, {drogon::Post});
    app.enableGzip(true);
    app.addListener("0.0.0.0", 8080);
    app.setThreadNum(std::thread::hardware_concurrency());
    app.run();
}
