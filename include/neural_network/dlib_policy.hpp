#ifndef __POKER_DLIB_POLICY_HPP__
#define __POKER_DLIB_POLICY_HPP__

#include <dlib/dnn.h>
#include <vector>
#include <algorithm>
#include <span>

constexpr int kInputDims = 24;
constexpr int kNumActions = 5; // Fold, Check/Call, 1/2 pot, pot, all-in

// 24 -> 64 -> 64 -> 5
using policy_net = dlib::loss_multiclass_log<
    dlib::fc<kNumActions,
             dlib::relu<dlib::fc<64,
                                 dlib::relu<dlib::fc<64,
                                                     dlib::input<dlib::matrix<float>>>>>>>>;

// Simple greedy policy with legality fallback.
// net(s) returns output_label_type = unsigned long (predicted class).
inline unsigned policy_greedy(policy_net &net,
                              const dlib::matrix<float> &s,
                              const std::span<const unsigned> legal)
{
  const unsigned long pred = net(s); // class index [0..kNumActions-1]

  // If the predicted action is legal, use it.
  if (std::find(legal.begin(), legal.end(), static_cast<unsigned>(pred)) != legal.end())
  {
    return static_cast<unsigned>(pred);
  }

  // Otherwise, pick a reasonable fallback from a preference list:
  // 1) Check/Call if allowed, 2) Fold if allowed, 3) first legal.
  auto has = [&](unsigned a)
  { return std::find(legal.begin(), legal.end(), a) != legal.end(); };

  constexpr unsigned A_Fold = 0;
  constexpr unsigned A_CheckCall = 1;
  // constexpr unsigned A_BetHalfPot = 2;
  // constexpr unsigned A_BetPot     = 3;
  // constexpr unsigned A_AllIn      = 4;

  if (has(A_CheckCall))
  {
    return A_CheckCall;
  }
  if (has(A_Fold))
  {
    return A_Fold;
  }

  return legal.empty() ? 0u : legal.front();
}

#endif // __POKER_DLIB_POLICY_HPP__
