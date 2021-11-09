#include "cublas_real.h"

namespace cccc
{

#ifndef _NO_CUDA
#ifdef STATIC_BLAS
cublasHandle_t Cublas::handle_;
#endif
#endif

}    // namespace cccc