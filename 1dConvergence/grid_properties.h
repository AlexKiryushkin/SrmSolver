#pragma once

#include <cstddef>

#include "types.h"

constexpr std::size_t lastMassFlowIdx = 80U;
constexpr std::size_t arraySize = 25U * lastMassFlowIdx + 1U;
constexpr std::size_t startIdx = lastMassFlowIdx + 1U;
constexpr std::size_t endIdx = arraySize - 10U;
constexpr std::size_t iterationCount = arraySize;

constexpr ElemT length = 1.0;
constexpr ElemT h = length / (arraySize - 1);
constexpr ElemT xBoundary = (startIdx - 1) * h;
