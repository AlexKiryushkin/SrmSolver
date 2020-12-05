#pragma once

#include <vector>

namespace kae
{

enum class ETimeDiscretizationOrder { eOne, eTwo, eThree };

template <class FloatT>
void reinitializeStep(std::vector<FloatT> & prevState,
                      std::vector<FloatT> & firstState,
                      std::vector<FloatT> & currState,
                      const std::vector<FloatT> & roots,
                      FloatT h,
                      ETimeDiscretizationOrder timeOrder);

} // namespace kae
