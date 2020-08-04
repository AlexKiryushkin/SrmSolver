#pragma once

#pragma warning(push, 0)
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>

#include <thrust/logical.h>
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/extrema.h>
#include <thrust/host_vector.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/reduce.h>
#include <thrust/remove.h>
#include <thrust/sequence.h>
#pragma warning(pop)

#ifdef __CUDACC__

#define HOST_DEVICE __host__  __device__

#else

#define HOST_DEVICE

#endif
