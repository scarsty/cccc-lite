#pragma once

struct cudaDeviceProp
{
    char name[256];                               /**< ASCII string identifying device */
    uint16_t uuid;                              /**< 16-byte unique identifier */
    char luid[8];                                 /**< 8-byte locally unique identifier. Value is undefined on TCC and non-Windows platforms */
    unsigned int luidDeviceNodeMask;              /**< LUID device node mask. Value is undefined on TCC and non-Windows platforms */
    size_t totalGlobalMem;                        /**< Global memory available on device in bytes */
    size_t sharedMemPerBlock;                     /**< Shared memory available per block in bytes */
    int regsPerBlock;                             /**< 32-bit registers available per block */
    int warpSize;                                 /**< Warp size in threads */
    size_t memPitch;                              /**< Maximum pitch in bytes allowed by memory copies */
    int maxThreadsPerBlock;                       /**< Maximum number of threads per block */
    int maxThreadsDim[3];                         /**< Maximum size of each dimension of a block */
    int maxGridSize[3];                           /**< Maximum size of each dimension of a grid */
    int clockRate;                                /**< Deprecated, Clock frequency in kilohertz */
    size_t totalConstMem;                         /**< Constant memory available on device in bytes */
    int major;                                    /**< Major compute capability */
    int minor;                                    /**< Minor compute capability */
    size_t textureAlignment;                      /**< Alignment requirement for textures */
    size_t texturePitchAlignment;                 /**< Pitch alignment requirement for texture references bound to pitched memory */
    int deviceOverlap;                            /**< Device can concurrently copy memory and execute a kernel. Deprecated. Use instead asyncEngineCount. */
    int multiProcessorCount;                      /**< Number of multiprocessors on device */
    int kernelExecTimeoutEnabled;                 /**< Deprecated, Specified whether there is a run time limit on kernels */
    int integrated;                               /**< Device is integrated as opposed to discrete */
    int canMapHostMemory;                         /**< Device can map host memory with cudaHostAlloc/cudaHostGetDevicePointer */
    int computeMode;                              /**< Deprecated, Compute mode (See ::cudaComputeMode) */
    int maxTexture1D;                             /**< Maximum 1D texture size */
    int maxTexture1DMipmap;                       /**< Maximum 1D mipmapped texture size */
    int maxTexture1DLinear;                       /**< Deprecated, do not use. Use cudaDeviceGetTexture1DLinearMaxWidth() or cuDeviceGetTexture1DLinearMaxWidth() instead. */
    int maxTexture2D[2];                          /**< Maximum 2D texture dimensions */
    int maxTexture2DMipmap[2];                    /**< Maximum 2D mipmapped texture dimensions */
    int maxTexture2DLinear[3];                    /**< Maximum dimensions (width, height, pitch) for 2D textures bound to pitched memory */
    int maxTexture2DGather[2];                    /**< Maximum 2D texture dimensions if texture gather operations have to be performed */
    int maxTexture3D[3];                          /**< Maximum 3D texture dimensions */
    int maxTexture3DAlt[3];                       /**< Maximum alternate 3D texture dimensions */
    int maxTextureCubemap;                        /**< Maximum Cubemap texture dimensions */
    int maxTexture1DLayered[2];                   /**< Maximum 1D layered texture dimensions */
    int maxTexture2DLayered[3];                   /**< Maximum 2D layered texture dimensions */
    int maxTextureCubemapLayered[2];              /**< Maximum Cubemap layered texture dimensions */
    int maxSurface1D;                             /**< Maximum 1D surface size */
    int maxSurface2D[2];                          /**< Maximum 2D surface dimensions */
    int maxSurface3D[3];                          /**< Maximum 3D surface dimensions */
    int maxSurface1DLayered[2];                   /**< Maximum 1D layered surface dimensions */
    int maxSurface2DLayered[3];                   /**< Maximum 2D layered surface dimensions */
    int maxSurfaceCubemap;                        /**< Maximum Cubemap surface dimensions */
    int maxSurfaceCubemapLayered[2];              /**< Maximum Cubemap layered surface dimensions */
    size_t surfaceAlignment;                      /**< Alignment requirements for surfaces */
    int concurrentKernels;                        /**< Device can possibly execute multiple kernels concurrently */
    int ECCEnabled;                               /**< Device has ECC support enabled */
    int pciBusID;                                 /**< PCI bus ID of the device */
    int pciDeviceID;                              /**< PCI device ID of the device */
    int pciDomainID;                              /**< PCI domain ID of the device */
    int tccDriver;                                /**< 1 if device is a Tesla device using TCC driver, 0 otherwise */
    int asyncEngineCount;                         /**< Number of asynchronous engines */
    int unifiedAddressing;                        /**< Device shares a unified address space with the host */
    int memoryClockRate;                          /**< Deprecated, Peak memory clock frequency in kilohertz */
    int memoryBusWidth;                           /**< Global memory bus width in bits */
    int l2CacheSize;                              /**< Size of L2 cache in bytes */
    int persistingL2CacheMaxSize;                 /**< Device's maximum l2 persisting lines capacity setting in bytes */
    int maxThreadsPerMultiProcessor;              /**< Maximum resident threads per multiprocessor */
    int streamPrioritiesSupported;                /**< Device supports stream priorities */
    int globalL1CacheSupported;                   /**< Device supports caching globals in L1 */
    int localL1CacheSupported;                    /**< Device supports caching locals in L1 */
    size_t sharedMemPerMultiprocessor;            /**< Shared memory available per multiprocessor in bytes */
    int regsPerMultiprocessor;                    /**< 32-bit registers available per multiprocessor */
    int managedMemory;                            /**< Device supports allocating managed memory on this system */
    int isMultiGpuBoard;                          /**< Device is on a multi-GPU board */
    int multiGpuBoardGroupID;                     /**< Unique identifier for a group of devices on the same multi-GPU board */
    int hostNativeAtomicSupported;                /**< Link between the device and the host supports native atomic operations */
    int singleToDoublePrecisionPerfRatio;         /**< Deprecated, Ratio of single precision performance (in floating-point operations per second) to double precision performance */
    int pageableMemoryAccess;                     /**< Device supports coherently accessing pageable memory without calling cudaHostRegister on it */
    int concurrentManagedAccess;                  /**< Device can coherently access managed memory concurrently with the CPU */
    int computePreemptionSupported;               /**< Device supports Compute Preemption */
    int canUseHostPointerForRegisteredMem;        /**< Device can access host registered memory at the same virtual address as the CPU */
    int cooperativeLaunch;                        /**< Device supports launching cooperative kernels via ::cudaLaunchCooperativeKernel */
    int cooperativeMultiDeviceLaunch;             /**< Deprecated, cudaLaunchCooperativeKernelMultiDevice is deprecated. */
    size_t sharedMemPerBlockOptin;                /**< Per device maximum shared memory per block usable by special opt in */
    int pageableMemoryAccessUsesHostPageTables;   /**< Device accesses pageable memory via the host's page tables */
    int directManagedMemAccessFromHost;           /**< Host can directly access managed memory on the device without migration. */
    int maxBlocksPerMultiProcessor;               /**< Maximum number of resident blocks per multiprocessor */
    int accessPolicyMaxWindowSize;                /**< The maximum value of ::cudaAccessPolicyWindow::num_bytes. */
    size_t reservedSharedMemPerBlock;             /**< Shared memory reserved by CUDA driver per block in bytes */
    int hostRegisterSupported;                    /**< Device supports host memory registration via ::cudaHostRegister. */
    int sparseCudaArraySupported;                 /**< 1 if the device supports sparse CUDA arrays and sparse CUDA mipmapped arrays, 0 otherwise */
    int hostRegisterReadOnlySupported;            /**< Device supports using the ::cudaHostRegister flag cudaHostRegisterReadOnly to register memory that must be mapped as read-only to the GPU */
    int timelineSemaphoreInteropSupported;        /**< External timeline semaphore interop is supported on the device */
    int memoryPoolsSupported;                     /**< 1 if the device supports using the cudaMallocAsync and cudaMemPool family of APIs, 0 otherwise */
    int gpuDirectRDMASupported;                   /**< 1 if the device supports GPUDirect RDMA APIs, 0 otherwise */
    unsigned int gpuDirectRDMAFlushWritesOptions; /**< Bitmask to be interpreted according to the ::cudaFlushGPUDirectRDMAWritesOptions enum */
    int gpuDirectRDMAWritesOrdering;              /**< See the ::cudaGPUDirectRDMAWritesOrdering enum for numerical values */
    unsigned int memoryPoolSupportedHandleTypes;  /**< Bitmask of handle types supported with mempool-based IPC */
    int deferredMappingCudaArraySupported;        /**< 1 if the device supports deferred mapping CUDA arrays and CUDA mipmapped arrays */
    int ipcEventSupported;                        /**< Device supports IPC Events. */
    int clusterLaunch;                            /**< Indicates device supports cluster launch */
    int unifiedFunctionPointers;                  /**< Indicates device supports unified pointers */
    int reserved2[2];
    int reserved[61]; /**< Reserved for future use */
};


enum
{
    cudaSuccess = 0,
    CUDNN_DIM_MAX = 8,

    CUBLAS_STATUS_SUCCESS,

    CUDNN_ACTIVATION_SIGMOID,
    CUDNN_ACTIVATION_TANH,
    CUDNN_ACTIVATION_RELU,
    CUDNN_ACTIVATION_CLIPPED_RELU,
    CUDNN_ACTIVATION_ELU,

    CUDNN_BATCHNORM_PER_ACTIVATION,
    CUDNN_BATCHNORM_SPATIAL,
    CUDNN_BIDIRECTIONAL,
    CUDNN_CROSS_CORRELATION,
    CUDNN_DATA_FLOAT,
    CUDNN_GRU,
    CUDNN_LINEAR_INPUT,
    CUDNN_LRN_CROSS_CHANNEL_DIM1,
    CUDNN_LSTM,
    CUDNN_NOT_PROPAGATE_NAN,
    CUDNN_OP_TENSOR_ADD,
    CUDNN_OP_TENSOR_MUL,
    CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING,
    CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING,
    CUDNN_POOLING_MAX,
    CUDNN_RNN_ALGO_PERSIST_DYNAMIC,
    CUDNN_RNN_ALGO_PERSIST_STATIC,
    CUDNN_RNN_ALGO_STANDARD,
    CUDNN_RNN_RELU,
    CUDNN_RNN_TANH,
    CUDNN_SKIP_INPUT,
    CUDNN_SOFTMAX_ACCURATE,
    CUDNN_SOFTMAX_FAST,
    CUDNN_SOFTMAX_LOG,
    CUDNN_SOFTMAX_MODE_INSTANCE,
    CUDNN_STATUS_SUCCESS,
    CUDNN_TENSOR_NCHW,
    CUDNN_TENSOR_OP_MATH,
    CUDNN_UNIDIRECTIONAL,
};

enum cudaMemcpyKind
{
    cudaMemcpyHostToHost = 0,     /**< Host   -> Host */
    cudaMemcpyHostToDevice = 1,   /**< Host   -> Device */
    cudaMemcpyDeviceToHost = 2,   /**< Device -> Host */
    cudaMemcpyDeviceToDevice = 3, /**< Device -> Device */
    cudaMemcpyDefault = 4         /**< Direction of the transfer is inferred from the pointer values. Requires unified virtual addressing */
};

#define CUDNN_BN_MIN_EPSILON 1e-5

enum cudaComputeMode
{
    cudaComputeModeDefault = 0,         /**< Default compute mode (Multiple threads can use ::cudaSetDevice() with this device) */
    cudaComputeModeExclusive = 1,       /**< Compute-exclusive-thread mode (Only one thread in one process will be able to use ::cudaSetDevice() with this device) */
    cudaComputeModeProhibited = 2,      /**< Compute-prohibited mode (No threads can use ::cudaSetDevice() with this device) */
    cudaComputeModeExclusiveProcess = 3 /**< Compute-exclusive-process mode (Many threads in one process will be able to use ::cudaSetDevice() with this device) */
};

struct Cheat
{
    size_t memory;
    int algo, mathType;
};

using cudaError = int;
using cudnnActivationDescriptor_t = void*;
using cudnnActivationMode_t = int;
using cudnnBatchNormMode_t = int;
using cudnnConvolutionBwdDataAlgo_t = int;
using cudnnConvolutionBwdDataAlgoPerf_t = Cheat;
using cudnnConvolutionBwdFilterAlgo_t = int;
using cudnnConvolutionBwdFilterAlgoPerf_t = Cheat;
using cudnnConvolutionDescriptor_t = void*;
using cudnnConvolutionFwdAlgo_t = int;
using cudnnConvolutionFwdAlgo_t = int;
using cudnnConvolutionFwdAlgoPerf_t = Cheat;
using cudnnConvolutionFwdAlgoPerf_t = Cheat;
using cudnnDataType_t = int;
using cudnnDropoutDescriptor_t = void*;
using cudnnFilterDescriptor_t = void*;
using cudnnHandle_t = void*;
using cudnnLRNDescriptor_t = void*;
using cudnnMathType_t = int;
using cudnnOpTensorDescriptor_t = void*;
using cudnnPoolingDescriptor_t = void*;
using cudnnPoolingMode_t = int;
using cudnnRNNDescriptor_t = void*;
using cudnnSpatialTransformerDescriptor_t = void*;
using cudnnStatus_t = int;

struct cudnnTensorStruct{};
using cudnnTensorDescriptor_t = cudnnTensorStruct*;