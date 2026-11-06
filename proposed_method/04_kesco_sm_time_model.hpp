#pragma once

#include <algorithm>
#include <array>
#include <optional>
#include <stdexcept>

/*
 * Per-kernel median execution times measured by
 * benchmarks/kesco_sm_scaling_benchmark/run.sh.
 *
 * Every curve is ordered by SM count: 8, 16, ..., 64. Repeated instances of
 * the same kernel (Black-Scholes tiles and HITS iterations) are represented by
 * the median across those task IDs. Values are milliseconds and correspond to
 * the fixed benchmark inputs and launch configurations in
 * benchmarks/common/kesco_benchmark_kernels.cuh.
 */
namespace kesco_sm_time_model {

using Curve = std::array<double, 8>;

inline constexpr Curve kVectorSquare[] = {
    {32.0314884, 32.0274715, 32.1832638, 32.1849747,
     32.6512470, 32.6467667, 32.6424484, 32.6399345},
    {34.7885742, 34.7899857, 34.9335670, 34.9178886,
     35.3140793, 35.3351669, 35.3176651, 35.3284492},
};

inline constexpr Curve kBlackScholes = {
    2.04760003, 1.03276801, 0.778239996, 0.525904000,
    0.525072009, 0.524495989, 0.524880022, 0.276927992,
};

inline constexpr Curve kMachineLearning[] = {
    {385.348755, 385.908722, 385.418213, 387.769226,
     385.397919, 387.686798, 385.378571, 385.341217},
    {112.910591, 71.987999, 58.6229134, 42.6519051,
     42.6819191, 42.4061127, 41.9016953, 35.3137932},
    {112.580803, 64.9129944, 55.8679695, 41.4912949,
     41.6698570, 41.4431343, 41.1469421, 33.4367027},
    {0.496544003, 0.491167992, 0.492736012, 0.492639989,
     0.494239986, 0.494688004, 0.494367987, 0.493600011},
    {0.682079971, 0.666463971, 0.668320000, 0.667295992,
     0.669856012, 0.670943975, 0.670047998, 0.669920027},
    {1.95382404, 1.69148803, 1.72316801, 1.65110397,
     1.67539203, 1.65727997, 1.64678395, 1.64844799},
    {1.76886404, 1.49008000, 1.47913599, 1.40780795,
     1.42208004, 1.41094399, 1.40480006, 1.40537596},
    {1.68899202, 1.29711997, 1.26064003, 1.25110400,
     1.25132799, 1.25212801, 1.25091195, 1.25174403},
    {1.68934405, 1.29488003, 1.25932801, 1.24870396,
     1.24995196, 1.25017595, 1.25001597, 1.24953604},
    {0.807392001, 0.796447992, 0.798976004, 0.801151991,
     0.803744018, 0.801984012, 0.799264014, 0.799583972},
};

inline constexpr Curve kHits[] = {
    {21.2108154, 21.1405754, 21.2611523, 21.1881924,
     21.3549442, 21.3264008, 21.4671688, 21.3239689},
    {21.2421112, 21.1257591, 21.2343998, 21.1719360,
     21.3400002, 21.3120956, 21.4573441, 21.2935047},
    {4.95852804, 4.96028805, 4.97743988, 4.98054409,
     5.04403210, 5.04169607, 5.04460812, 5.04073620},
    {4.96003199, 4.96051216, 4.97545624, 4.98127985,
     5.03945589, 5.04329586, 5.03683186, 5.03984022},
    {5.33807993, 5.33292818, 5.35283184, 5.35004807,
     5.41728020, 5.41692781, 5.42080021, 5.41296005},
    {5.34793615, 5.34284782, 5.36323214, 5.36345577,
     5.42947197, 5.43107224, 5.42748785, 5.42700815},
    {0.005344, 0.005376, 0.005440, 0.005344,
     0.005344, 0.005312, 0.005344, 0.005440},
};

inline constexpr Curve kImageProcessing[] = {
    {18.7137280, 9.28787231, 7.81670380, 4.73350382,
     6.97523212, 7.00051212, 6.59001589, 4.28480005},
    {41.6382408, 21.5595207, 17.2307205, 11.9008961,
     12.3946877, 13.9250879, 12.8088636, 8.52278423},
    {18.7327995, 9.28585625, 7.71161604, 4.73583984,
     6.94499207, 6.84966421, 6.45193577, 4.28812790},
    {15.0478401, 7.94630384, 6.04537582, 4.14121580,
     4.22748804, 4.31903982, 4.39113617, 3.07609606},
    {15.0600643, 7.94025612, 6.03452778, 4.13769579,
     4.24150419, 4.31190395, 4.39881611, 3.07491207},
    {8.10937595, 8.10697555, 8.13625622, 8.14988804,
     8.25155163, 8.25980759, 8.25260830, 8.25443172},
    {8.10614395, 8.10486412, 8.13478375, 8.13884830,
     8.24172783, 8.25129604, 8.24479961, 8.23983955},
    {9.69574356, 9.67715168, 9.70934391, 9.70764828,
     9.83651161, 9.83657646, 9.83244801, 9.83385563},
    {9.10102367, 9.09478378, 9.12297630, 9.12211227,
     9.21615982, 9.21587181, 9.21235180, 9.21417618},
    {9.47907162, 9.47539234, 9.50207996, 9.49667168,
     9.59440041, 9.59427166, 9.59062386, 9.59132767},
    {9.59331226, 9.58819199, 9.61475182, 9.61065579,
     9.70278359, 9.70662403, 9.70793629, 9.70454407},
};

inline constexpr Curve kDeepLearning[] = {
    {90.1472321, 82.0704346, 80.3971558, 78.6731186,
     78.7451172, 78.6065292, 78.4036484, 78.3773727},
    {90.1784668, 82.0724487, 80.4044495, 78.6755829,
     78.7462082, 78.6083527, 78.4059830, 78.2465897},
    {32.1786575, 31.7999039, 31.2411518, 29.9734077,
     31.0960636, 31.1550713, 31.1086082, 31.0160637},
    {32.1582069, 31.7565441, 31.2125111, 29.9617271,
     31.0677433, 31.1264000, 31.0716801, 30.9839363},
    {17.8698883, 9.00831985, 6.93062401, 5.27539206,
     5.26854420, 5.26867199, 5.27504015, 4.60336018},
    {17.8926392, 9.02166367, 6.93142414, 5.27587223,
     5.27222395, 5.27238417, 5.26748800, 4.60486412},
    {2.27728009, 2.27654409, 2.28451204, 2.28451204,
     2.31337595, 2.31344008, 2.31171203, 2.31420803},
    {2.53996801, 2.53823996, 2.54662394, 2.54559994,
     2.57382393, 2.57283211, 2.57334399, 2.57484794},
};

inline const Curve* find_curve(const TaskSpec& task) {
  switch (task.benchmark_kind) {
    case stg::BenchmarkKind::SYNTHETIC:
      return nullptr;
    case stg::BenchmarkKind::VECTOR_SQUARE:
      if (task.id >= 0 && task.id <= 2) {
        return &kVectorSquare[task.id == 2 ? 1 : 0];
      }
      break;
    case stg::BenchmarkKind::BLACK_SCHOLES:
      if (task.id >= 0 && task.id < 10) return &kBlackScholes;
      break;
    case stg::BenchmarkKind::MACHINE_LEARNING:
      if (task.id >= 0 && task.id < 10) return &kMachineLearning[task.id];
      break;
    case stg::BenchmarkKind::HITS:
      if (task.id >= 0 && task.id < 35) return &kHits[task.id % 7];
      break;
    case stg::BenchmarkKind::IMAGE_PROCESSING:
      if (task.id >= 0 && task.id < 11) return &kImageProcessing[task.id];
      break;
    case stg::BenchmarkKind::DEEP_LEARNING:
      if (task.id >= 0 && task.id < 8) return &kDeepLearning[task.id];
      break;
  }

  throw std::out_of_range("no measured SM-time curve for KESCO task");
}

inline std::optional<double> estimate_ms(
    const TaskSpec& task,
    int stream_sm_count
) {
  const Curve* curve = find_curve(task);
  if (curve == nullptr) return std::nullopt;

  const int effective_sm = std::clamp(
      std::min(stream_sm_count, std::max(1, task.parallel_sm_limit)),
      8,
      64
  );
  const int lower_index = (effective_sm - 8) / 8;
  const int lower_sm = 8 + lower_index * 8;
  if (effective_sm == lower_sm || lower_index == 7) {
    return curve->at(static_cast<std::size_t>(lower_index));
  }

  const int upper_index = lower_index + 1;
  const double fraction =
      static_cast<double>(effective_sm - lower_sm) / 8.0;
  const double lower_time = curve->at(static_cast<std::size_t>(lower_index));
  const double upper_time = curve->at(static_cast<std::size_t>(upper_index));
  return lower_time + fraction * (upper_time - lower_time);
}

}  // namespace kesco_sm_time_model
