#pragma once

#include <vector>

namespace kae {

template <class ElemT>
ElemT quarticSolve(const std::vector<ElemT>& data, std::size_t idx, ElemT h);

} // namespace kae
