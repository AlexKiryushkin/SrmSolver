#pragma once

#include <vector>

namespace kae
{

enum class ETimeDiscretizationOrder { eOne, eTwo, eThree };

template <class FloatT>
void reinitializeStep2d(const std::vector<FloatT> & initState,
                        std::vector<FloatT> & prevState,
                        std::vector<FloatT> & firstState,
                        std::vector<FloatT> & currState,
                        const std::vector<FloatT> & xRoots,
                        const std::vector<FloatT> & yRoots,
                        std::size_t nx,
                        std::size_t ny,
                        FloatT hx,
                        FloatT hy,
                        ETimeDiscretizationOrder timeOrder);

} // namespace kae
