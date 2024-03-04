#if !defined(_COMMON_H_)
#define _COMMON_H_

#include <map>
#include <time.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <iostream>
#include <switch.h>
#include <utility>
#include <vector>
#include <stack>
#include <list>
#include <forward_list>
#include <cuda.h>
#include <cudnn.h>
#include <assert.h>
#include <cublas_v2.h>
#include <pthread.h>
#include <cuda_runtime.h>
#include <glog/logging.h>
#include <util/ATP_math.h>

// #define SEED_LOCK
// #define GRAD_ACC            // Gradient accumulation
// #define BIG_BATCHSIZE 64    // Gradient accumulation batchsize

// #define BASELINE_TRAINING               // run the training without recomputation and swapping
// #define TRAINING_CONFIGURATION_SEARCH  // run the best training configuration search (search the batch size, recomputing tensors and swapping tensors)
#define RECOPUTING_SWAPPING_TRAINING    // run the training with the strategy of ATP

#define ATP_SOLUTION

#ifdef BASELINE_TRAINING
    #define POOL_MALLOC_MODE
#endif


#ifdef TRAINING_CONFIGURATION_SEARCH
    #define BEST_BATCHSIZE
    #define SWAP_MALLOC_MODE
    #define REAL_BANDWIDTH 
    #define HEAD_END_SWAP
#endif

#ifdef RECOPUTING_SWAPPING_TRAINING
    #define RECOMPUTE_ON
    #define SWAP_ON
#endif

#ifndef TRAINING_CONFIGURATION_SEARCH
    #define POOL_MALLOC_MODE
#endif

#ifdef ATP_SOLUTION
    #define GA_ON
    #define TENSOR_GRANULARITY
    // #define TORCH_DEFAULT_ALGO
    // #define FASTEST_ALGO
    #define NO_WORKSPACE_ALGO
#endif

#define B_DATA_POOL_NUM 3

#define MULTI_SWAP_BLOCK
#ifdef MULTI_SWAP_BLOCK
#define SWAP_BLOCK_NUM 3
#endif

#ifdef SWAP_ONLY
#define RECOMPUTE_POOL_NUM 0
#else
#define RECOMPUTE_POOL_NUM 3
#endif

#define SIMULATE
#define FAKE_TRAIN             // uncomment this definition and use the randomly generated dataset which can be used for the throughput test


#define BLASX_GPU_MEM_SIZE  (1024L*1024L*1000L*3L)

typedef std::pair<int, int> d_key_t;

typedef enum MODEL_TYPE {
	CNN_NETWORK		= 0,
	RNN_NETWORK		= 1
} MODEL_TYPE;

typedef enum net_comp {
    FORWARD  = 0,
    BACKWARD = 1,
	RECOMPUTE = 2
} net_comp;

typedef enum network_stage {
    NET_TRAIN   = 0,
    NET_INFER   = 1
} network_stage;

typedef enum data_mode {
    DATA_TRAIN   = 0,
    DATA_TEST    = 1
} data_mode;

typedef enum join_mode {
    ELEWISE_SUM  = 0,
    ELEWISE_MAX  = 1
} join_mode;

typedef enum structure_type {
    FORK = 0,
    JOIN = 1,
} structure_type;

typedef enum mem_mode {
    VOID      = 0,
    GPU_NIL   = 1,      // gpu with invalid data
    GPU_FUL   = 2,      // gpu with valid data
    CPU       = 3,
    CPU2GPU   = 4,
    GPU2CPU   = 5
    // RECOMPUTE = 6
} mem_mode;

typedef enum LAYER {
    /*---network layers---*/
    CONV    = 0,
    POOL    = 1,
    ACT     = 2,
    BN      = 3,
    FC      = 4,
    LRN     = 5,
    PADDING = 6,
    DATA_L  = 7,
    DROPOUT = 8,
    SOFTMAX = 9,
	CTCLOSS = 10,
	RNN		= 11,
    /*--structure layers--*/
    CONCAT  = 12,
    FORK_L  = 13,
    JOIN_L  = 14,
    SATTN   = 15
} LAYER;

typedef enum RNN_TYPE {
    RNN_RELU       	= 0,
	RNN_TANH		= 1,
    LSTM       		= 2,
    GRU       		= 3
} RNN_TYPE;

typedef enum RNN_BIAS {
    NO	       		= 0,
	SINGLE			= 1,
    DOUBLE     		= 2,
} RNN_BIAS;

typedef enum RNN_DIRECTION {
	UNIDIRECTIONAL	= 0,
	BIDIRECTIONAL	= 1
} RNN_DIRECTION;

typedef enum LOSS_MODE {
    MEAN        	= 0,
    SUM        		= 1,
    NONE       		= 2
} LOSS_MODE;

typedef enum COMM_DIRECTION {
	COMM_GPU2CPU = 0,
	COMM_CPU2GPU = 1
} COMM_DIRECTION;

#define INSTANTIATE_CLASS(classname) \
char gInstantiationGuard##classname; \
template class classname<float>; \
template class classname<double>

#endif // _COMMON_H_
