#include "cublas_real.h"

namespace will
{

#ifndef _NO_CUDA
#ifdef STATIC_BLAS
cublasHandle_t Cublas::handle_;
#endif
#endif

}    // namespace will