#include "cublas_real.h"

namespace woco
{

#ifndef _NO_CUDA
#ifdef STATIC_BLAS
cublasHandle_t Cublas::handle_;
#endif
#endif

}    // namespace woco