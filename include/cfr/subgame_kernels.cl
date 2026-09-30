// OpenCL C for GpuSubgameSolver: one traversal of vector-form DCFR for player `p`, as
// forward (reach, level by level from the root) -> terminals -> backward (values and regrets, level by
// level to the root). `vec` holds each node's opponent reach on the way down and is overwritten with
// the traverser's values on the way up. The current strategy is recomputed from regrets where needed.
// CMake embeds this file as the string `subgameKernels` (see cmake/embed.cmake).
#define MAX_HANDS 1326
#define GROUP 256
#define MAX_ACTIONS 8
#define NONE 0xFFFFFFFFu
#define NO_HAND 0xFFFFu
#define DECISION 0
#define FOLD 1
#define SHOWDOWN 2
#define CHANCE 3

typedef struct
{
    uint vec;        // into vec: one float per hand
    uint regret;     // decision: into regrets, [action * hands + hand]
    uint average;    // decision: into averages, same layout, or NONE
    uint space;
    uint hands;
    uint firstChild; // children are contiguous
    uint invested;   // player 0 in the low half, player 1 in the high half
    float weight;    // chance: each child's probability for a hand pair
    uchar kind, toAct, children, folder;
} Node;

inline float positive(float r) { return r > 0.0f ? r : 0.0f; }

// Reach of the children of the non-terminal nodes ids[first, first + count), from theirs: one work-item
// per node and hand computes the current strategy once for all its children.
__kernel void forward(__global const Node *nodes, __global const uint *ids, uint first, uint p, __global float *vec,
                      __global const half *regrets, __global float *averages, float discount, __global const uint *spaceInverse,
                      __global const ushort *inverse)
{
    const Node node = nodes[ids[first + get_global_id(0)]];
    const uint i = get_global_id(1);
    if (i >= node.hands)
    {
        return;
    }
    const float reach = vec[node.vec + i];
    const uint n = node.children;
    if (node.kind == CHANCE)
    {
        for (uint k = 0; k < n; ++k)
        {
            const Node child = nodes[node.firstChild + k];
            const uint j = inverse[spaceInverse[child.space] + i];
            if (j != NO_HAND)
            {
                vec[child.vec + j] = reach;
            }
        }
        return;
    }
    if (node.toAct == p)
    {
        for (uint a = 0; a < n; ++a)
        {
            vec[nodes[node.firstChild + a].vec + i] = reach;
        }
        return;
    }
    __global const half *regret = regrets + node.regret;
    const uint hands = node.hands;
    float r[MAX_ACTIONS];
    float total = 0.0f;
    for (uint a = 0; a < n; ++a)
    {
        r[a] = positive(vload_half(a * hands + i, regret));
        total += r[a];
    }
    const float inverseTotal = total > 0.0f ? 1.0f / total : 0.0f; // rounded as on the CPU
    const float uniform = total > 0.0f ? 0.0f : 1.0f / (float)n;
    for (uint a = 0; a < n; ++a)
    {
        const float x = reach * (r[a] * inverseTotal + uniform);
        vec[nodes[node.firstChild + a].vec + i] = x;
        if (node.average != NONE)
        {
            const uint at = node.average + a * hands + i;
            averages[at] = averages[at] * discount + x;
        }
    }
}

// Terminal values for player p, one work-group per terminal, in place of its reach.
__kernel __attribute__((reqd_work_group_size(GROUP, 1, 1))) void terminals(
    __global const Node *nodes, __global const uint *ids, uint p, __global float *vec, __global const uint *spaceHand,
    __global const uint *spaceList, __global const uint *spaceCards, __global const uchar *low, __global const uchar *high,
    __global const ushort *groupStart, __global const ushort *groupEnd, __global const ushort *lowBefore,
    __global const ushort *lowUpto, __global const ushort *highBefore, __global const ushort *highUpto,
    __global const ushort *cardHands)
{
    __local float reach[MAX_HANDS];
    __local float prefix[MAX_HANDS + 1];      // reach of the hands before each position
    __local float cardPrefix[2 * MAX_HANDS + 52]; // per card list entry k of card c at k + c; the card's total after its list
    __local float part[GROUP], cardPart[GROUP];
    const Node node = nodes[ids[get_group_id(0)]];
    const uint n = node.hands, lid = get_local_id(0);
    __global float *v = vec + node.vec;
    for (uint i = lid; i < n; i += GROUP)
    {
        reach[i] = v[i];
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    const uint h0 = spaceHand[node.space];
    __global const uint *cards = spaceCards + 53 * node.space;
    __global const ushort *list = cardHands + spaceList[node.space];
    // Per-card prefixes, four threads per card: each sums a quarter of the card's list, and once all
    // quarter sums are in, writes its prefixes starting from the quarters before it.
    const uint card = lid / 4, quarter = lid % 4;
    uint from = 0, to = 0;
    float cardSum = 0.0f;
    if (card < 52)
    {
        const uint cardBegin = cards[card], cardEnd = cards[card + 1];
        const uint per = (cardEnd - cardBegin + 3) / 4;
        from = min(cardBegin + quarter * per, cardEnd);
        to = min(from + per, cardEnd);
        for (uint k = from; k < to; ++k)
        {
            cardSum += reach[list[k]];
        }
    }
    cardPart[lid] = cardSum;
    // Exclusive prefix over positions: per-thread chunks, then a scan of the chunk sums.
    const uint chunk = (n + GROUP - 1) / GROUP;
    const uint begin = min(lid * chunk, n), stop = min(begin + chunk, n);
    float sum = 0.0f;
    for (uint i = begin; i < stop; ++i)
    {
        sum += reach[i];
    }
    part[lid] = sum;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (uint offset = 1; offset < GROUP; offset <<= 1)
    {
        const float add = lid >= offset ? part[lid - offset] : 0.0f;
        barrier(CLK_LOCAL_MEM_FENCE);
        part[lid] += add;
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    float acc = part[lid] - sum;
    for (uint i = begin; i < stop; ++i)
    {
        prefix[i] = acc;
        acc += reach[i];
    }
    if (lid == GROUP - 1)
    {
        prefix[n] = part[lid];
    }
    if (card < 52)
    {
        float cardAcc = 0.0f;
        for (uint q = 0; q < quarter; ++q)
        {
            cardAcc += cardPart[4 * card + q];
        }
        for (uint k = from; k < to; ++k)
        {
            cardPrefix[k + card] = cardAcc;
            cardAcc += reach[list[k]];
        }
        if (quarter == 3)
        {
            cardPrefix[to + card] = cardAcc; // the last quarter ends at the list's end
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    const float total = prefix[n];
    const uint mine = (node.invested >> (16 * p)) & 0xFFFFu;
    if (node.kind == FOLD)
    {
        const uint folderInvested = (node.invested >> (16 * node.folder)) & 0xFFFFu;
        const float payoff = node.folder == p ? -(float)mine : (float)folderInvested;
        for (uint i = lid; i < n; i += GROUP)
        {
            const uint l = low[h0 + i], h = high[h0 + i];
            v[i] = payoff * (total - cardPrefix[cards[l + 1] + l] - cardPrefix[cards[h + 1] + h] + reach[i]);
        }
        return;
    }
    // Showdown, as on the CPU: win - lose = W + W' - D over disjoint opponent mass strictly weaker (W),
    // weaker or tied (W') and all of it (D); a hand counts itself in both card sums, so W' and D add it back.
    const float stake = (float)mine;
    for (uint i = lid; i < n; i += GROUP)
    {
        const uint l = low[h0 + i], h = high[h0 + i];
        const float weaker = prefix[groupStart[h0 + i]] - cardPrefix[lowBefore[h0 + i] + l] - cardPrefix[highBefore[h0 + i] + h];
        const float upto = prefix[groupEnd[h0 + i]] - cardPrefix[lowUpto[h0 + i] + l] - cardPrefix[highUpto[h0 + i] + h] + reach[i];
        const float all = total - cardPrefix[cards[l + 1] + l] - cardPrefix[cards[h + 1] + h] + reach[i];
        v[i] = stake * (weaker + upto - all);
    }
}

// Values of the non-terminal nodes ids[first, first + count) from their children's, updating the
// traverser's regrets.
__kernel void backward(__global const Node *nodes, __global const uint *ids, uint first, uint p, __global float *vec,
                       __global half *regrets, float positiveDiscount, float scale, __global const uint *spaceInverse,
                       __global const ushort *inverse)
{
    const Node node = nodes[ids[first + get_global_id(0)]];
    const uint i = get_global_id(1);
    if (i >= node.hands)
    {
        return;
    }
    const uint n = node.children;
    float out = 0.0f;
    if (node.kind == CHANCE)
    {
        for (uint k = 0; k < n; ++k)
        {
            const Node child = nodes[node.firstChild + k];
            const uint j = inverse[spaceInverse[child.space] + i];
            out += j != NO_HAND ? vec[child.vec + j] : 0.0f;
        }
        vec[node.vec + i] = out * node.weight;
        return;
    }
    if (node.toAct != p)
    {
        for (uint a = 0; a < n; ++a)
        {
            out += vec[nodes[node.firstChild + a].vec + i];
        }
        vec[node.vec + i] = out;
        return;
    }
    __global half *regret = regrets + node.regret;
    const uint hands = node.hands;
    float r[MAX_ACTIONS], v[MAX_ACTIONS];
    float total = 0.0f;
    for (uint a = 0; a < n; ++a)
    {
        r[a] = vload_half(a * hands + i, regret);
        v[a] = vec[nodes[node.firstChild + a].vec + i];
        total += positive(r[a]);
    }
    const float inverseTotal = total > 0.0f ? 1.0f / total : 0.0f;
    const float uniform = total > 0.0f ? 0.0f : 1.0f / (float)n;
    for (uint a = 0; a < n; ++a)
    {
        out += (positive(r[a]) * inverseTotal + uniform) * v[a];
    }
    for (uint a = 0; a < n; ++a)
    {
        vstore_half(r[a] * (r[a] > 0.0f ? positiveDiscount : 0.5f) + (v[a] - out) * scale, a * hands + i, regret);
    }
    vec[node.vec + i] = out;
}
