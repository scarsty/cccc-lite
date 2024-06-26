IMPORT(cublasCreate_v2)
IMPORT(cublasDestroy_v2)
IMPORT(cublasGetVersion)
IMPORT(cublasIsamax_v2)
IMPORT(cublasSasum_v2)
IMPORT(cublasSdot_v2)
IMPORT(cublasSetMathMode)
IMPORT(cublasSgemm_v2)
IMPORT(cublasSgemv_v2)
IMPORT(cublasSscal_v2)
IMPORT(cublasScalEx	)
IMPORT(cublasIdamax_v2)
IMPORT(cublasHgemm)
IMPORT(cublasDscal_v2)
IMPORT(cublasDotEx)
IMPORT(cublasDgemv_v2)
IMPORT(cublasDgemm_v2)
IMPORT(cublasDdot_v2)
IMPORT(cublasDasum_v2)

IMPORT(cudnnActivationBackward)
IMPORT(cudnnActivationForward)
IMPORT(cudnnAddTensor)
IMPORT(cudnnBatchNormalizationBackward)
IMPORT(cudnnBatchNormalizationForwardInference)
IMPORT(cudnnBatchNormalizationForwardTraining)
IMPORT(cudnnConvolutionBackwardBias)
IMPORT(cudnnConvolutionBackwardData)
IMPORT(cudnnConvolutionBackwardFilter)
IMPORT(cudnnConvolutionForward)
IMPORT(cudnnCreate)
IMPORT(cudnnCreateActivationDescriptor)
IMPORT(cudnnCreateConvolutionDescriptor)
IMPORT(cudnnCreateDropoutDescriptor)
IMPORT(cudnnCreateFilterDescriptor)
IMPORT(cudnnCreateLRNDescriptor)
IMPORT(cudnnCreateOpTensorDescriptor)
IMPORT(cudnnCreatePoolingDescriptor)
IMPORT(cudnnCreateRNNDescriptor)
IMPORT(cudnnCreateSpatialTransformerDescriptor)
IMPORT(cudnnCreateTensorDescriptor)
IMPORT(cudnnDeriveBNTensorDescriptor)
IMPORT(cudnnDestroy)
IMPORT(cudnnDestroyActivationDescriptor)
IMPORT(cudnnDestroyConvolutionDescriptor)
IMPORT(cudnnDestroyDropoutDescriptor)
IMPORT(cudnnDestroyFilterDescriptor)
IMPORT(cudnnDestroyLRNDescriptor)
IMPORT(cudnnDestroyOpTensorDescriptor)
IMPORT(cudnnDestroyPoolingDescriptor)
IMPORT(cudnnDestroyRNNDescriptor)
IMPORT(cudnnDestroySpatialTransformerDescriptor)
IMPORT(cudnnDestroyTensorDescriptor)
IMPORT(cudnnDropoutBackward)
IMPORT(cudnnDropoutForward)
IMPORT(cudnnDropoutGetReserveSpaceSize)
IMPORT(cudnnDropoutGetStatesSize)
IMPORT(cudnnFindConvolutionBackwardDataAlgorithm)
IMPORT(cudnnFindConvolutionBackwardFilterAlgorithm)
IMPORT(cudnnFindConvolutionForwardAlgorithm)
IMPORT(cudnnGetConvolutionBackwardDataWorkspaceSize)
IMPORT(cudnnGetConvolutionBackwardFilterWorkspaceSize)
IMPORT(cudnnGetConvolutionForwardWorkspaceSize)
IMPORT(cudnnGetErrorString)
IMPORT(cudnnGetTensor4dDescriptor)
IMPORT(cudnnGetTensorNdDescriptor)
IMPORT(cudnnGetVersion)
IMPORT(cudnnLRNCrossChannelBackward)
IMPORT(cudnnLRNCrossChannelForward)
IMPORT(cudnnOpTensor)
IMPORT(cudnnPoolingBackward)
IMPORT(cudnnPoolingForward)
IMPORT(cudnnSetActivationDescriptor)
IMPORT(cudnnSetConvolution2dDescriptor)
IMPORT(cudnnSetConvolutionMathType)
IMPORT(cudnnSetConvolutionNdDescriptor)
IMPORT(cudnnSetDropoutDescriptor)
IMPORT(cudnnSetFilter4dDescriptor)
IMPORT(cudnnSetFilterNdDescriptor)
IMPORT(cudnnSetLRNDescriptor)
IMPORT(cudnnSetOpTensorDescriptor)
IMPORT(cudnnSetPooling2dDescriptor)
IMPORT(cudnnSetPoolingNdDescriptor)
IMPORT(cudnnSetTensor)
IMPORT(cudnnSetTensor4dDescriptor)
IMPORT(cudnnSetTensorNdDescriptor)
IMPORT(cudnnSoftmaxBackward)
IMPORT(cudnnSoftmaxForward)
IMPORT(cudnnGetLastErrorString)

IMPORT(cudaMemset)
IMPORT(cudaGetDeviceProperties_v2, "cudaGetDeviceProperties")
IMPORT(cudaGetErrorString)
IMPORT(cudaMalloc)
IMPORT(cudaSetDevice)
IMPORT(cudaDeviceGetPCIBusId)
IMPORT(cudaMemGetInfo)
IMPORT(cudaFree)
IMPORT(cudaRuntimeGetVersion)
IMPORT(cudaGetDeviceCount)
IMPORT(cudaDriverGetVersion)
IMPORT(cudaMemcpy)
IMPORT(cudaMemcpyPeer)

UNIMPORT_OR_BLANKTEMP(cuda_reciprocal)
UNIMPORT_OR_BLANKTEMP(cuda_addnumber)
UNIMPORT_OR_BLANKTEMP(cuda_pow)
UNIMPORT_OR_BLANKTEMP(cuda_sparse)
UNIMPORT_OR_BLANKTEMP(cuda_sign)
UNIMPORT_OR_BLANKTEMP(cuda_cross_entropy)
UNIMPORT_OR_BLANKTEMP(cuda_cross_entropy2)
UNIMPORT_OR_BLANKTEMP(cuda_add)
UNIMPORT_OR_BLANKTEMP(cuda_mul)
UNIMPORT_OR_BLANKTEMP(cuda_div)
UNIMPORT_OR_BLANKTEMP(cuda_sectionlimit)
UNIMPORT_OR_BLANKTEMP(cuda_ada_update)
UNIMPORT_OR_BLANKTEMP(cuda_ada_delta_update)
UNIMPORT_OR_BLANKTEMP(cuda_adam_update)
UNIMPORT_OR_BLANKTEMP(cuda_rms_prop_update)
UNIMPORT_OR_BLANKTEMP(cuda_sin)
UNIMPORT_OR_BLANKTEMP(cuda_cos)
UNIMPORT_OR_BLANKTEMP(cuda_zigzag)
UNIMPORT_OR_BLANKTEMP(cuda_zigzagb)
UNIMPORT_OR_BLANKTEMP(cuda_step)
UNIMPORT_OR_BLANKTEMP(cuda_leaky_relu)
UNIMPORT_OR_BLANKTEMP(cuda_leaky_relub)
UNIMPORT_OR_BLANKTEMP(cuda_max)
UNIMPORT_OR_BLANKTEMP(cuda_maxb)