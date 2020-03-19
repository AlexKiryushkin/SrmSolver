#pragma once

#include <cuda_runtime_api.h>

inline __host__ __device__ float4 operator+(float4 lhs, float4 rhs)
{
  return float4{ lhs.x + rhs.x, lhs.y + rhs.y, lhs.z + rhs.z, lhs.w + rhs.w };
}

inline __host__ __device__ float4 operator-(float4 lhs, float4 rhs)
{
  return float4{ lhs.x - rhs.x, lhs.y - rhs.y, lhs.z - rhs.z, lhs.w - rhs.w };
}

inline __host__ __device__ float4 operator*(float lhs, float4 rhs)
{
  return float4{ lhs * rhs.x, lhs * rhs.y, lhs * rhs.z, lhs * rhs.w };
}
