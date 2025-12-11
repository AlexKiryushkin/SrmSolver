#pragma once

#include "cuda_includes.h"
#include "to_float.h"

namespace kae
{

template <class ElemT>
struct GpuGridT
{
    GpuGridT(unsigned nx_, unsigned ny_, ElemT lx_, ElemT ly_, unsigned smExtension_,
        unsigned blockSizeX_ = 32U * sizeof(float) / sizeof(ElemT),
        unsigned blockSizeY_ = 8U * sizeof(float) / sizeof(ElemT))
        : nx{ nx_ }, ny{ ny_ }, n{ nx * ny }, lx{ lx_ }, ly{ ly_ },
          hx{ lx / (nx - 1) }, hy{ ly / (ny - 1) }, hxReciprocal{ 1 / hx }, hyReciprocal{ 1 / hy },
          blockSize{ blockSizeX_, blockSizeY_ }, gridSize{ (nx + blockSize.x - 1) / blockSize.x, (ny + blockSize.y - 1) / blockSize.y },
        smExtension{smExtension_}, sharedMemory{ blockSize.x + 2 * smExtension, blockSize.y + 2 * smExtension },
        smSize{ sharedMemory.x * sharedMemory.y }, smSizeBytes{static_cast<unsigned>(smSize * sizeof(ElemT))}
    {
    }

    unsigned nx;
    unsigned ny;
    unsigned n;
    ElemT lx;
    ElemT ly;
    ElemT hx;
    ElemT hy;
    ElemT hxReciprocal;
    ElemT hyReciprocal;
    dim3 blockSize;
    dim3 gridSize;
    unsigned smExtension;
    dim3 sharedMemory;
    unsigned smSize;
    unsigned smSizeBytes;
};

template <class ElemT>
std::ostream& operator<<(std::ostream& os, GpuGridT<ElemT> grid)
{
    os << "******************************************\n";
    os << "Grid values\n";
    os << "h = " << grid.hx << "\n";
    os << "(nx, ny) = (" << grid.nx << ", " << grid.ny << ")\n";
    os << "(lx, ly) = (" << grid.lx << ", " << grid.ly << ")\n";
    os << "******************************************\n";

    return os;
}

} // namespace kae
