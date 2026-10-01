#ifndef __POKER_CFR_GPU_SUBGAME_SOLVER_HPP__
#define __POKER_CFR_GPU_SUBGAME_SOLVER_HPP__
#include "subgame_kernels.hpp"
#include "subgame_tree.hpp"
#include <cmath>
#include <iostream>
#include <memory>
#include <vector>
#define CL_HPP_TARGET_OPENCL_VERSION 300
#define CL_HPP_MIN_REQUIRED_OPENCL_VERSION 120
#define CL_HPP_ENABLE_EXCEPTIONS
#pragma warning(push, 0)
#include <CL/opencl.hpp>
#pragma warning(pop)

// The first OpenCL GPU with the subgame kernels built for it, shared by every solver.
struct Gpu
{
    cl::Device device;
    cl::Context context;
    cl::Program program;

    // nullptr without an OpenCL GPU. A kernel build failure throws, with the log on stderr.
    static const Gpu *instance()
    {
        static const std::unique_ptr<Gpu> gpu = []() -> std::unique_ptr<Gpu>
        {
            std::vector<cl::Device> devices;
            try
            {
                std::vector<cl::Platform> platforms;
                cl::Platform::get(&platforms);
                for (std::size_t i = 0; i < platforms.size() && devices.empty(); ++i)
                {
                    try
                    {
                        platforms[i].getDevices(CL_DEVICE_TYPE_GPU, &devices);
                    }
                    catch (const cl::Error &)
                    {
                        devices.clear(); // no GPU on this platform
                    }
                }
            }
            catch (const cl::Error &)
            {
                return nullptr; // no OpenCL runtime
            }
            if (devices.empty())
            {
                return nullptr;
            }
            auto result = std::make_unique<Gpu>();
            result->device = devices.front();
            result->context = cl::Context(result->device);
            result->program = cl::Program(result->context, subgameKernels);
            try
            {
                result->program.build({result->device}, "-cl-std=CL1.2");
            }
            catch (const cl::BuildError &error)
            {
                for (const auto &[device, log] : error.getBuildLog())
                {
                    std::cerr << log << '\n';
                }
                throw;
            }
            return result;
        }();
        return gpu.get();
    }
};

// Seconds a profiled GpuSubgameSolver's kernels ran, by kind, and how many were launched.
struct GpuProfile
{
    double forward = 0.0, folds = 0.0, showdowns = 0.0, allIns = 0.0, backward = 0.0;
    std::size_t launches = 0;
};

// SubgameSolver on the GPU: the same DCFR over the same SubgameTree, with everything the iterations
// touch kept in device memory. Nodes are renumbered breadth first so each node's children are
// contiguous; one traversal launches a kernel per level down (reach), one for the folds, one for the
// showdowns, one for the all-ins, and one per level up (values and regrets), each over a level's nodes
// times their hands.
// Going down skips the traverser's own decisions: they pass the opponent's reach on unchanged, so the
// nodes below read it from the nearest ancestor holding it. Regrets are stored in half precision; only
// the average strategy comes back.
// Requires Gpu::instance().
template <typename C>
class GpuSubgameSolver
{
public:
    using Tree = SubgameTree<C>;
    using G = Hunl<C>;
    using Hands = typename Tree::Hands;
    using Strategy = typename Tree::Strategy;

    // See SubgameTree for the arguments; `profiled` times every kernel (see profile()).
    GpuSubgameSolver(const typename G::State &root, const std::array<Hands, 2> &ranges, const SubgameOptions &options = {}, bool profiled = false)
        : m_tree(root, ranges, options), m_gpu(*Gpu::instance()), m_profiled(profiled),
          m_queue(m_gpu.context, m_gpu.device, profiled ? CL_QUEUE_PROFILING_ENABLE : 0),
          m_forward(m_gpu.program, "forward"), m_folds(m_gpu.program, "folds"), m_showdowns(m_gpu.program, "showdowns"),
          m_allIns(m_gpu.program, "allIns"), m_backward(m_gpu.program, "backward"),
          m_averages(m_tree.averages, 0.0f)
    {
        upload();
    }

    void solve(std::size_t iterations)
    {
        const std::size_t width = (m_tree.hands + 63) / 64 * 64;
        for (std::size_t it = 0; it < iterations; ++it)
        {
            ++m_iteration;
            const double t = static_cast<double>(m_iteration - 1); // discounts apply to sums through t
            const auto positiveDiscount = static_cast<float>(std::pow(t, 1.5) / (std::pow(t, 1.5) + 1.0));
            const auto strategyDiscount = static_cast<float>(std::pow(t / (t + 1.0), 2.0));
            for (std::uint32_t p = 0; p < 2; ++p)
            {
                m_queue.enqueueCopyBuffer(m_ranges[1 - p], m_vec, 0, 0, m_tree.hands * sizeof(float));
                m_forward.setArg(1, m_forwardIds[p]);
                m_forward.setArg(3, p);
                m_forward.setArg(7, strategyDiscount);
                for (const auto &[first, count] : m_forwardLevels[p])
                {
                    if (count > 0)
                    {
                        m_forward.setArg(2, first);
                        launch(m_forward, cl::NDRange(count, width), cl::NDRange(1, 64), &GpuProfile::forward);
                    }
                }
                if (m_numFolds > 0)
                {
                    m_folds.setArg(2, p);
                    launch(m_folds, cl::NDRange(m_numFolds * foldGroup), cl::NDRange(foldGroup), &GpuProfile::folds);
                }
                if (m_numShowdowns > 0)
                {
                    m_showdowns.setArg(2, p);
                    launch(m_showdowns, cl::NDRange(m_numShowdowns * group), cl::NDRange(group), &GpuProfile::showdowns);
                }
                if (m_numAllIns > 0)
                {
                    m_allIns.setArg(2, p);
                    launch(m_allIns, cl::NDRange(m_numAllIns * allInGroup), cl::NDRange(allInGroup), &GpuProfile::allIns);
                }
                m_backward.setArg(3, p);
                m_backward.setArg(6, positiveDiscount);
                m_backward.setArg(7, m_scale[p]);
                for (std::size_t level = m_internal.size(); level-- > 0;)
                {
                    if (m_internal[level].second > 0)
                    {
                        m_backward.setArg(2, m_internal[level].first);
                        launch(m_backward, cl::NDRange(m_internal[level].second, width), cl::NDRange(1, 64), &GpuProfile::backward);
                    }
                }
            }
        }
        if (!m_averages.empty())
        {
            m_queue.enqueueReadBuffer(m_averageSums, CL_TRUE, 0, m_averages.size() * sizeof(float), m_averages.data());
        }
        m_queue.finish();
        for (const auto &[event, kind] : m_events)
        {
            m_profile.*kind += 1e-9 * static_cast<double>(event.template getProfilingInfo<CL_PROFILING_COMMAND_END>() -
                                                          event.template getProfilingInfo<CL_PROFILING_COMMAND_START>());
            ++m_profile.launches;
        }
        m_events.clear();
    }

    template <typename S>
    Strategy strategy(const S &s, std::size_t hand) const { return m_tree.strategy(m_averages, s, hand); }
    template <typename S>
    inline bool contains(const S &s) const { return m_tree.byHistory.contains(s.history); }
    inline std::size_t numNodes() const noexcept { return m_tree.nodes.size(); }
    inline const Tree &tree() const noexcept { return m_tree; }
    inline const std::vector<float> &averageSums() const noexcept { return m_averages; }
    inline const GpuProfile &profile() const noexcept { return m_profile; }

private:
    static constexpr std::uint32_t group = 256, foldGroup = 64, allInGroup = 1024; // the kernels' GROUP, FOLD_GROUP, ALL_IN_GROUP
    static constexpr std::uint32_t none = 0xFFFFFFFFu;
    static constexpr std::uint16_t noHand = 0xFFFFu;

    // Node in the kernels.
    struct GpuNode
    {
        std::uint32_t vec, regret, average, space, hands, firstChild, invested;
        std::array<std::uint32_t, 2> reach;
        float weight;
        std::uint8_t kind, toAct, children, folder;
    };
    static_assert(sizeof(GpuNode) == 44);
    static_assert(Tree::maxChildren <= 8); // MAX_ACTIONS in the kernels

    Tree m_tree;
    const Gpu &m_gpu;
    bool m_profiled;
    GpuProfile m_profile;
    std::vector<std::pair<cl::Event, double GpuProfile::*>> m_events; // of the solve under way
    cl::CommandQueue m_queue;
    cl::Kernel m_forward, m_folds, m_showdowns, m_allIns, m_backward;
    std::vector<float> m_averages;
    std::size_t m_iteration = 0;
    std::vector<std::pair<std::uint32_t, std::uint32_t>> m_internal; // range of m_internalIds per level
    std::array<std::vector<std::pair<std::uint32_t, std::uint32_t>>, 2> m_forwardLevels; // of m_forwardIds[p]
    std::array<cl::Buffer, 2> m_forwardIds; // m_internalIds without player p's decisions
    std::uint32_t m_numFolds = 0, m_numShowdowns = 0, m_numAllIns = 0;
    cl::Buffer m_nodes, m_internalIds, m_foldIds, m_showdownIds, m_allInIds, m_vec, m_regrets, m_averageSums;
    std::array<cl::Buffer, 2> m_ranges;
    // Regrets are stored in half precision times m_scale[p] for player p, which bounds them by a few
    // units: counterfactual values scale with the stack and the opponent's reach mass. Regret matching
    // only uses ratios, so the scale never needs undoing.
    std::array<float, 2> m_scale{};
    std::vector<cl::Buffer> m_spaceBuffers; // kept alive for the kernels

    void launch(cl::Kernel &kernel, const cl::NDRange &global, const cl::NDRange &local, double GpuProfile::*kind)
    {
        if (m_profiled)
        {
            cl::Event event;
            m_queue.enqueueNDRangeKernel(kernel, cl::NullRange, global, local, nullptr, &event);
            m_events.emplace_back(event, kind);
        }
        else
        {
            m_queue.enqueueNDRangeKernel(kernel, cl::NullRange, global, local);
        }
    }

    template <typename T>
    cl::Buffer buffer(const std::vector<T> &data)
    {
        const std::size_t bytes = std::max<std::size_t>(data.size(), 1) * sizeof(T);
        cl::Buffer result(m_gpu.context, CL_MEM_READ_WRITE, bytes);
        if (!data.empty())
        {
            m_queue.enqueueWriteBuffer(result, CL_TRUE, 0, data.size() * sizeof(T), data.data());
        }
        return result;
    }
    template <typename T = float>
    cl::Buffer zeros(std::size_t count)
    {
        const std::size_t bytes = std::max<std::size_t>(count, 1) * sizeof(T);
        cl::Buffer result(m_gpu.context, CL_MEM_READ_WRITE, bytes);
        m_queue.enqueueFillBuffer(result, T{}, 0, bytes);
        return result;
    }

    // SubgameTree::allInBalance computed on the device (the CPU takes ~70 ms for a new flop).
    // `low` and `high` hold the spaces' cards, the root's first.
    cl::Buffer allInBalance(const cl::Buffer &low, const cl::Buffer &high)
    {
        const std::size_t n = m_tree.hands;
        const std::uint64_t flop = m_tree.nodes.front().state.board;
        std::vector<std::uint16_t> hole(n), ranks;
        for (std::size_t i = 0; i < n; ++i)
        {
            hole[i] = static_cast<std::uint16_t>(m_tree.holeOf(i));
        }
        ranks.reserve(1176 * n); // per runout
        forEachPair(((1ull << 52) - 1) & ~flop, [&](std::uint64_t runout)
                    {
            const auto &board = boardRanks(flop | runout);
            for (std::size_t i = 0; i < n; ++i)
            {
                ranks.push_back((holes[hole[i]] & runout) == 0 ? static_cast<std::uint16_t>(board[hole[i]] + 1) : std::uint16_t{0});
            } });
        cl::Buffer balance(m_gpu.context, CL_MEM_READ_WRITE, n * n * sizeof(float));
        cl::Kernel kernel(m_gpu.program, "allInBalance");
        const cl::Buffer runoutRanks = buffer(ranks);
        kernel.setArg(0, runoutRanks);
        kernel.setArg(1, static_cast<cl_uint>(n));
        kernel.setArg(2, static_cast<cl_uint>(ranks.size() / n));
        kernel.setArg(3, low);
        kernel.setArg(4, high);
        kernel.setArg(5, balance);
        m_queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(n, n));
        return balance;
    }

    void upload()
    {
        using Kind = typename Tree::Kind;
        const auto &nodes = m_tree.nodes;
        const auto &spaces = m_tree.spaces;

        // Breadth-first order.
        std::vector<std::size_t> order;
        std::vector<std::uint32_t> gpuOf(nodes.size());
        std::vector<std::pair<std::uint32_t, std::uint32_t>> levels; // breadth-first node range per level
        std::vector<std::size_t> level{0}, next;
        while (!level.empty())
        {
            levels.emplace_back(static_cast<std::uint32_t>(order.size()), static_cast<std::uint32_t>(level.size()));
            next.clear();
            for (const std::size_t id : level)
            {
                gpuOf[id] = static_cast<std::uint32_t>(order.size());
                order.push_back(id);
                next.insert(next.end(), nodes[id].children.begin(), nodes[id].children.end());
            }
            level.swap(next);
        }

        std::vector<GpuNode> gpuNodes(order.size());
        std::vector<std::uint32_t> internalIds, foldIds, showdownIds, allInIds;
        std::array<std::vector<std::uint32_t>, 2> forwardIds;
        std::uint32_t vec = 0;
        for (const auto &[first, count] : levels)
        {
            m_internal.emplace_back(static_cast<std::uint32_t>(internalIds.size()), 0);
            for (std::size_t p = 0; p < 2; ++p)
            {
                m_forwardLevels[p].emplace_back(static_cast<std::uint32_t>(forwardIds[p].size()), 0);
            }
            for (std::uint32_t g = first; g < first + count; ++g)
            {
                const auto &node = nodes[order[g]];
                const auto &s = node.state;
                GpuNode &out = gpuNodes[g];
                out.vec = vec;
                out.hands = static_cast<std::uint32_t>(spaces[node.space].size());
                vec += out.hands;
                out.regret = static_cast<std::uint32_t>(node.offset);
                out.average = node.average == Tree::none ? none : static_cast<std::uint32_t>(node.average);
                out.space = static_cast<std::uint32_t>(node.space);
                out.firstChild = node.children.empty() ? 0 : gpuOf[node.children.front()];
                out.invested = s.invested[0] | (s.invested[1] << 16);
                out.weight = node.weight;
                out.kind = static_cast<std::uint8_t>(node.kind);
                out.toAct = s.toAct;
                out.children = static_cast<std::uint8_t>(node.children.size());
                out.folder = s.folder;
                if (node.kind == Kind::fold)
                {
                    foldIds.push_back(g);
                }
                else if (node.kind == Kind::showdown)
                {
                    showdownIds.push_back(g);
                }
                else if (node.kind == Kind::allIn)
                {
                    allInIds.push_back(g);
                }
                else
                {
                    internalIds.push_back(g);
                    ++m_internal.back().second;
                    for (std::size_t p = 0; p < 2; ++p)
                    {
                        if (node.kind == Kind::chance || s.toAct != p)
                        {
                            forwardIds[p].push_back(g);
                            ++m_forwardLevels[p].back().second;
                        }
                    }
                }
            }
        }
        // Where each node finds the opponent's reach when p traverses: in its own vector, unless its
        // parent is a decision of p's, which hands down its own source.
        gpuNodes[0].reach = {gpuNodes[0].vec, gpuNodes[0].vec};
        for (std::size_t g = 0; g < order.size(); ++g)
        {
            const auto &node = nodes[order[g]];
            for (const std::size_t child : node.children)
            {
                GpuNode &out = gpuNodes[gpuOf[child]];
                for (std::size_t p = 0; p < 2; ++p)
                {
                    out.reach[p] = node.kind == Kind::decision && node.state.toAct == p ? gpuNodes[g].reach[p] : out.vec;
                }
            }
        }
        m_numFolds = static_cast<std::uint32_t>(foldIds.size());
        m_numShowdowns = static_cast<std::uint32_t>(showdownIds.size());
        m_numAllIns = static_cast<std::uint32_t>(allInIds.size());

        // Spaces: per-hand arrays, per-card hand lists and inverse maps into parent spaces.
        std::vector<std::size_t> parentSpace(spaces.size(), 0);
        for (const auto &node : nodes)
        {
            if (node.kind == Kind::chance)
            {
                for (const std::size_t child : node.children)
                {
                    parentSpace[nodes[child].space] = node.space;
                }
            }
        }
        std::vector<std::uint32_t> spaceHand, spaceList, spaceCards, spaceInverse;
        std::vector<std::uint8_t> low, high;
        std::vector<std::uint16_t> groupStart, groupEnd, lowBefore, lowUpto, highBefore, highUpto, cardHands, inverse;
        std::size_t totalHands = 0, totalInverse = 0;
        for (std::size_t sIndex = 0; sIndex < spaces.size(); ++sIndex)
        {
            totalHands += spaces[sIndex].size();
            totalInverse += sIndex > 0 ? spaces[parentSpace[sIndex]].size() : 0;
        }
        low.reserve(totalHands);
        high.reserve(totalHands);
        for (auto *array : {&groupStart, &groupEnd, &lowBefore, &lowUpto, &highBefore, &highUpto})
        {
            array->reserve(totalHands);
        }
        cardHands.reserve(2 * totalHands);
        inverse.reserve(totalInverse);
        spaceCards.reserve(53 * spaces.size());
        for (std::size_t sIndex = 0; sIndex < spaces.size(); ++sIndex)
        {
            const auto &space = spaces[sIndex];
            const std::size_t n = space.size(), handBase = low.size(), listBase = cardHands.size();
            spaceHand.push_back(static_cast<std::uint32_t>(handBase));
            spaceList.push_back(static_cast<std::uint32_t>(listBase));
            low.insert(low.end(), space.low.begin(), space.low.end());
            high.insert(high.end(), space.high.begin(), space.high.end());
            // Each card's list of the hands holding it, in hand order, one after the other.
            std::array<std::uint32_t, 53> bounds{};
            for (std::size_t i = 0; i < n; ++i)
            {
                ++bounds[space.low[i] + 1];
                ++bounds[space.high[i] + 1];
            }
            for (std::size_t c = 0; c < 52; ++c)
            {
                bounds[c + 1] += bounds[c];
            }
            spaceCards.insert(spaceCards.end(), bounds.begin(), bounds.end());
            cardHands.resize(listBase + 2 * n);
            std::array<std::uint32_t, 52> listed{};
            for (std::size_t i = 0; i < n; ++i)
            {
                for (const std::size_t c : {space.low[i], space.high[i]})
                {
                    cardHands[listBase + bounds[c] + listed[c]++] = static_cast<std::uint16_t>(i);
                }
            }
            // Per hand, its equal-strength run and where each of its cards' lists crosses the run's ends
            // (as indices into the space's lists); left 0 in spaces without showdowns.
            for (auto *array : {&groupStart, &groupEnd, &lowBefore, &lowUpto, &highBefore, &highUpto})
            {
                array->resize(handBase + n);
            }
            std::array<std::uint32_t, 52> seen{}; // hands holding each card before the current run
            for (std::size_t g = 0; g + 1 < space.groups.size(); ++g)
            {
                const std::size_t begin = space.groups[g], end = space.groups[g + 1];
                for (std::size_t i = begin; i < end; ++i)
                {
                    groupStart[handBase + i] = static_cast<std::uint16_t>(begin);
                    groupEnd[handBase + i] = static_cast<std::uint16_t>(end);
                    lowBefore[handBase + i] = static_cast<std::uint16_t>(bounds[space.low[i]] + seen[space.low[i]]);
                    highBefore[handBase + i] = static_cast<std::uint16_t>(bounds[space.high[i]] + seen[space.high[i]]);
                }
                for (std::size_t i = begin; i < end; ++i)
                {
                    ++seen[space.low[i]];
                    ++seen[space.high[i]];
                }
                for (std::size_t i = begin; i < end; ++i)
                {
                    lowUpto[handBase + i] = static_cast<std::uint16_t>(bounds[space.low[i]] + seen[space.low[i]]);
                    highUpto[handBase + i] = static_cast<std::uint16_t>(bounds[space.high[i]] + seen[space.high[i]]);
                }
            }
            spaceInverse.push_back(static_cast<std::uint32_t>(inverse.size()));
            if (sIndex > 0)
            {
                const std::size_t base = inverse.size();
                inverse.resize(base + spaces[parentSpace[sIndex]].size(), noHand);
                for (std::size_t j = 0; j < n; ++j)
                {
                    inverse[base + space.parent[j]] = static_cast<std::uint16_t>(j);
                }
            }
        }

        m_nodes = buffer(gpuNodes);
        m_internalIds = buffer(internalIds);
        for (std::size_t p = 0; p < 2; ++p)
        {
            m_forwardIds[p] = buffer(forwardIds[p]);
        }
        m_foldIds = buffer(foldIds);
        m_showdownIds = buffer(showdownIds);
        m_allInIds = buffer(allInIds);
        m_vec = zeros(vec);
        m_regrets = zeros<std::uint16_t>(m_tree.regrets); // half precision, scaled (see m_scale)
        for (std::size_t p = 0; p < 2; ++p)
        {
            double mass = 0.0;
            for (const float r : m_tree.ranges[1 - p])
            {
                mass += r;
            }
            m_scale[p] = static_cast<float>(1.0 / std::max(mass * C::stack, 1e-6));
        }
        m_averageSums = zeros(m_tree.averages);
        for (std::size_t p = 0; p < 2; ++p)
        {
            m_ranges[p] = buffer(m_tree.ranges[p]);
        }
        const auto keep = [&](cl::Buffer b)
        {
            m_spaceBuffers.push_back(b);
            return b;
        };
        const cl::Buffer inverseBase = keep(buffer(spaceInverse)), inverseHands = keep(buffer(inverse));
        m_forward.setArg(0, m_nodes);
        m_forward.setArg(4, m_vec);
        m_forward.setArg(5, m_regrets);
        m_forward.setArg(6, m_averageSums);
        m_forward.setArg(8, inverseBase);
        m_forward.setArg(9, inverseHands);
        const cl::Buffer hand = keep(buffer(spaceHand)), list = keep(buffer(spaceList)), cards = keep(buffer(spaceCards)),
                         lows = keep(buffer(low)), highs = keep(buffer(high)), lists = keep(buffer(cardHands));
        m_folds.setArg(0, m_nodes);
        m_folds.setArg(1, m_foldIds);
        m_folds.setArg(3, m_vec);
        m_folds.setArg(4, hand);
        m_folds.setArg(5, list);
        m_folds.setArg(6, cards);
        m_folds.setArg(7, lows);
        m_folds.setArg(8, highs);
        m_folds.setArg(9, lists);
        m_showdowns.setArg(0, m_nodes);
        m_showdowns.setArg(1, m_showdownIds);
        m_showdowns.setArg(3, m_vec);
        m_showdowns.setArg(4, hand);
        m_showdowns.setArg(5, list);
        m_showdowns.setArg(6, cards);
        m_showdowns.setArg(7, lows);
        m_showdowns.setArg(8, highs);
        m_showdowns.setArg(9, keep(buffer(groupStart)));
        m_showdowns.setArg(10, keep(buffer(groupEnd)));
        m_showdowns.setArg(11, keep(buffer(lowBefore)));
        m_showdowns.setArg(12, keep(buffer(lowUpto)));
        m_showdowns.setArg(13, keep(buffer(highBefore)));
        m_showdowns.setArg(14, keep(buffer(highUpto)));
        m_showdowns.setArg(15, lists);
        m_allIns.setArg(0, m_nodes);
        m_allIns.setArg(1, m_allInIds);
        m_allIns.setArg(3, m_vec);
        if (m_numAllIns > 0)
        {
            m_allIns.setArg(4, keep(allInBalance(lows, highs)));
        }
        m_backward.setArg(0, m_nodes);
        m_backward.setArg(1, m_internalIds);
        m_backward.setArg(4, m_vec);
        m_backward.setArg(5, m_regrets);
        m_backward.setArg(8, inverseBase);
        m_backward.setArg(9, inverseHands);
    }
};
#endif // __POKER_CFR_GPU_SUBGAME_SOLVER_HPP__
