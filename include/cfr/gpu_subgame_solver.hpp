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

// SubgameSolver on the GPU: the same DCFR over the same SubgameTree, with everything the iterations
// touch kept in device memory. Nodes are renumbered breadth first so each node's children are
// contiguous; one traversal launches a kernel per level down (reach), one for all terminals, and one
// per level up (values and regrets), each over a level's non-terminal nodes times their hands. Regrets
// are stored in half precision; only the average strategy comes back.
// Requires Gpu::instance().
template <typename C>
class GpuSubgameSolver
{
public:
    using Tree = SubgameTree<C>;
    using G = Hunl<C>;
    using Hands = typename Tree::Hands;
    using Strategy = typename Tree::Strategy;

    // See SubgameTree for the arguments.
    GpuSubgameSolver(const typename G::State &root, const std::array<Hands, 2> &ranges, bool averageLaterStreets = true, double minReach = 0.0,
                     std::size_t chanceSamples = 0, std::size_t allInSamples = 0)
        : m_tree(root, ranges, averageLaterStreets, minReach, chanceSamples, allInSamples), m_gpu(*Gpu::instance()), m_queue(m_gpu.context, m_gpu.device),
          m_forward(m_gpu.program, "forward"), m_terminals(m_gpu.program, "terminals"), m_backward(m_gpu.program, "backward"),
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
                m_forward.setArg(3, p);
                m_forward.setArg(7, strategyDiscount);
                for (const auto &[first, count] : m_internal)
                {
                    if (count > 0)
                    {
                        m_forward.setArg(2, first);
                        m_queue.enqueueNDRangeKernel(m_forward, cl::NullRange, cl::NDRange(count, width), cl::NDRange(1, 64));
                    }
                }
                if (m_numTerminals > 0)
                {
                    m_terminals.setArg(2, p);
                    m_queue.enqueueNDRangeKernel(m_terminals, cl::NullRange, cl::NDRange(m_numTerminals * group), cl::NDRange(group));
                }
                m_backward.setArg(3, p);
                m_backward.setArg(6, positiveDiscount);
                m_backward.setArg(7, m_scale[p]);
                for (std::size_t level = m_internal.size(); level-- > 0;)
                {
                    if (m_internal[level].second > 0)
                    {
                        m_backward.setArg(2, m_internal[level].first);
                        m_queue.enqueueNDRangeKernel(m_backward, cl::NullRange, cl::NDRange(m_internal[level].second, width), cl::NDRange(1, 64));
                    }
                }
            }
        }
        if (!m_averages.empty())
        {
            m_queue.enqueueReadBuffer(m_averageSums, CL_TRUE, 0, m_averages.size() * sizeof(float), m_averages.data());
        }
        m_queue.finish();
    }

    template <typename S>
    Strategy strategy(const S &s, std::size_t hand) const { return m_tree.strategy(m_averages, s, hand); }
    template <typename S>
    inline bool contains(const S &s) const { return m_tree.byHistory.contains(s.history); }
    inline std::size_t numNodes() const noexcept { return m_tree.nodes.size(); }
    inline const Tree &tree() const noexcept { return m_tree; }
    inline const std::vector<float> &averageSums() const noexcept { return m_averages; }

private:
    static constexpr std::uint32_t group = 256; // GROUP in the kernels
    static constexpr std::uint32_t none = 0xFFFFFFFFu;
    static constexpr std::uint16_t noHand = 0xFFFFu;

    // Node in the kernels.
    struct GpuNode
    {
        std::uint32_t vec, regret, average, space, hands, firstChild, invested;
        float weight;
        std::uint8_t kind, toAct, children, folder;
    };
    static_assert(sizeof(GpuNode) == 36);
    static_assert(G::maxActions <= 8); // MAX_ACTIONS in the kernels

    Tree m_tree;
    const Gpu &m_gpu;
    cl::CommandQueue m_queue;
    cl::Kernel m_forward, m_terminals, m_backward;
    std::vector<float> m_averages;
    std::size_t m_iteration = 0;
    std::vector<std::pair<std::uint32_t, std::uint32_t>> m_internal; // range of m_internalIds per level
    std::uint32_t m_numTerminals = 0;
    cl::Buffer m_nodes, m_internalIds, m_terminalIds, m_vec, m_regrets, m_averageSums;
    std::array<cl::Buffer, 2> m_ranges;
    // Regrets are stored in half precision times m_scale[p] for player p, which bounds them by a few
    // units: counterfactual values scale with the stack and the opponent's reach mass. Regret matching
    // only uses ratios, so the scale never needs undoing.
    std::array<float, 2> m_scale{};
    std::vector<cl::Buffer> m_spaceBuffers; // kept alive for the kernels

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
        std::vector<std::uint32_t> internalIds, terminalIds;
        std::uint32_t vec = 0;
        for (const auto &[first, count] : levels)
        {
            m_internal.emplace_back(static_cast<std::uint32_t>(internalIds.size()), 0);
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
                if (node.kind == Kind::fold || node.kind == Kind::showdown)
                {
                    terminalIds.push_back(g);
                }
                else
                {
                    internalIds.push_back(g);
                    ++m_internal.back().second;
                }
            }
        }
        m_numTerminals = static_cast<std::uint32_t>(terminalIds.size());

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
        for (std::size_t sIndex = 0; sIndex < spaces.size(); ++sIndex)
        {
            const auto &space = spaces[sIndex];
            const std::size_t n = space.size();
            spaceHand.push_back(static_cast<std::uint32_t>(low.size()));
            spaceList.push_back(static_cast<std::uint32_t>(cardHands.size()));
            low.insert(low.end(), space.low.begin(), space.low.end());
            high.insert(high.end(), space.high.begin(), space.high.end());
            std::array<std::vector<std::uint16_t>, 52> lists;
            for (std::size_t i = 0; i < n; ++i)
            {
                lists[space.low[i]].push_back(static_cast<std::uint16_t>(i));
                lists[space.high[i]].push_back(static_cast<std::uint16_t>(i));
            }
            std::array<std::uint32_t, 53> bounds{};
            for (std::size_t c = 0; c < 52; ++c)
            {
                bounds[c + 1] = bounds[c] + static_cast<std::uint32_t>(lists[c].size());
                cardHands.insert(cardHands.end(), lists[c].begin(), lists[c].end());
            }
            spaceCards.insert(spaceCards.end(), bounds.begin(), bounds.end());
            // Entries of card c's list before position `at`, as an index into the space's lists.
            const auto before = [&](std::size_t c, std::size_t at)
            {
                const auto &list = lists[c];
                return static_cast<std::uint16_t>(bounds[c] + (std::lower_bound(list.begin(), list.end(), at) - list.begin()));
            };
            for (std::size_t g = 0; g + 1 < space.groups.size(); ++g)
            {
                for (std::size_t i = space.groups[g]; i < space.groups[g + 1]; ++i)
                {
                    groupStart.push_back(static_cast<std::uint16_t>(space.groups[g]));
                    groupEnd.push_back(static_cast<std::uint16_t>(space.groups[g + 1]));
                    lowBefore.push_back(before(space.low[i], space.groups[g]));
                    lowUpto.push_back(before(space.low[i], space.groups[g + 1]));
                    highBefore.push_back(before(space.high[i], space.groups[g]));
                    highUpto.push_back(before(space.high[i], space.groups[g + 1]));
                }
            }
            for (std::size_t i = groupStart.size(); i < low.size(); ++i) // spaces without showdowns
            {
                groupStart.push_back(0);
                groupEnd.push_back(0);
                lowBefore.push_back(0);
                lowUpto.push_back(0);
                highBefore.push_back(0);
                highUpto.push_back(0);
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
        m_terminalIds = buffer(terminalIds);
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
        m_forward.setArg(1, m_internalIds);
        m_forward.setArg(4, m_vec);
        m_forward.setArg(5, m_regrets);
        m_forward.setArg(6, m_averageSums);
        m_forward.setArg(8, inverseBase);
        m_forward.setArg(9, inverseHands);
        m_terminals.setArg(0, m_nodes);
        m_terminals.setArg(1, m_terminalIds);
        m_terminals.setArg(3, m_vec);
        m_terminals.setArg(4, keep(buffer(spaceHand)));
        m_terminals.setArg(5, keep(buffer(spaceList)));
        m_terminals.setArg(6, keep(buffer(spaceCards)));
        m_terminals.setArg(7, keep(buffer(low)));
        m_terminals.setArg(8, keep(buffer(high)));
        m_terminals.setArg(9, keep(buffer(groupStart)));
        m_terminals.setArg(10, keep(buffer(groupEnd)));
        m_terminals.setArg(11, keep(buffer(lowBefore)));
        m_terminals.setArg(12, keep(buffer(lowUpto)));
        m_terminals.setArg(13, keep(buffer(highBefore)));
        m_terminals.setArg(14, keep(buffer(highUpto)));
        m_terminals.setArg(15, keep(buffer(cardHands)));
        m_backward.setArg(0, m_nodes);
        m_backward.setArg(1, m_internalIds);
        m_backward.setArg(4, m_vec);
        m_backward.setArg(5, m_regrets);
        m_backward.setArg(8, inverseBase);
        m_backward.setArg(9, inverseHands);
    }
};
#endif // __POKER_CFR_GPU_SUBGAME_SOLVER_HPP__
