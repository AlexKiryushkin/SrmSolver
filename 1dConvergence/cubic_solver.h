#pragma once

#include <vector>

#include "types.h"

namespace kae {

ElemT cubicSolve(const std::vector<ElemT> & data, std::size_t idx, ElemT h);

} // namespace kae