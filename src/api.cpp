#include <expected>
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
    }
};
struct ProbabilitiesRequest
{
    Deck hand;
    Deck table;
    std::size_t numPlayers;
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
};

template<typename TFunc>
static void handleRequest(const drogon::HttpRequestPtr &req, std::function<void(const drogon::HttpResponsePtr &)> &&callback, TFunc &&logicFunc)
{
    std::string_view body = req->getBody();
    ProbabilitiesRequest probReq;
    auto error = glz::read_json(probReq, body);
    if (error)
    {
        auto formatted = glz::format_error(error, body);
        ErrorResponse errorResponse{formatted};
        auto objectJson = glz::write_json(errorResponse);
        if (!objectJson)
        {
            auto resp = drogon::HttpResponse::newHttpResponse();
            resp->setStatusCode(drogon::HttpStatusCode::k500InternalServerError);
            callback(resp);
            return;
        }
        auto resp = drogon::HttpResponse::newHttpResponse();
        resp->setContentTypeCode(drogon::CT_APPLICATION_JSON);
        resp->setBody(std::move(*objectJson));
        resp->setStatusCode(drogon::HttpStatusCode::k400BadRequest);
        callback(resp);
        return;
    }
    auto result = logicFunc(probReq);
    auto jsonResponse = glz::write_json(result);
    auto resp = drogon::HttpResponse::newHttpResponse();
    if (!jsonResponse)
    {
        resp->setStatusCode(drogon::HttpStatusCode::k500InternalServerError);
        callback(resp);
        return;
    }
    resp->setStatusCode(drogon::HttpStatusCode::k200OK);
    resp->setContentTypeCode(drogon::CT_APPLICATION_JSON);
    resp->setBody(std::move(*jsonResponse));
    callback(resp);
}

int main()
{
    auto &app = drogon::app();
    std::size_t numThreads = std::thread::hardware_concurrency();
    BS::thread_pool threadPool(numThreads);
    app.registerHandler("/statistics", [&threadPool](const drogon::HttpRequestPtr &req, std::function<void(const drogon::HttpResponsePtr &)> &&callback)
    { 
        handleRequest(req, std::move(callback), [&threadPool](const ProbabilitiesRequest& probReq) { 
            return computeRandomGameStatistics(probReq.hand, probReq.table, probReq.numSimulations, probReq.numPlayers, threadPool); 
        }); 
    }, {drogon::Post});
    app.registerHandler("/probabilities", [&threadPool](const drogon::HttpRequestPtr &req, std::function<void(const drogon::HttpResponsePtr &)> &&callback)
    { 
        handleRequest(req, std::move(callback), [&threadPool](const ProbabilitiesRequest& probReq) { 
            return ProbabilityResult{ probabilityOfWinning(probReq.hand, probReq.table, probReq.numSimulations, probReq.numPlayers, threadPool) }; 
        }); 
    }, {drogon::Post});
    app.enableGzip(true);
    app.addListener("0.0.0.0", 8080);
    app.setThreadNum(numThreads);
    app.run();
}