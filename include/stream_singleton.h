//
// Created by ay27 on 8/4/17.
//

#ifndef ATP_STREAM_CONTROL_H
#define ATP_STREAM_CONTROL_H

#include <cuda.h>
#include <cuda_runtime_api.h>

namespace ATP {

#define MAX_STREAM 

class stream_singleton {

    cudaStream_t compute_stream;
    cudaStream_t cpu2gpu_stream;
    cudaStream_t gpu2cpu_stream;
	// cudaStream_t cpu2gpu_stream

    stream_singleton() {
        // cudaSetDevice(0);
        cudaStreamCreate(&compute_stream);
        cudaStreamCreate(&cpu2gpu_stream);
        cudaStreamCreate(&gpu2cpu_stream);
    }

    // we must call the destructor explicitly
    ~stream_singleton() {
        cudaStreamDestroy(compute_stream);
        cudaStreamDestroy(cpu2gpu_stream);
        cudaStreamDestroy(gpu2cpu_stream);
    }

    static stream_singleton* instance;

public:

    static cudaStream_t get_compute_stream() {
        if (instance != NULL) {
            return instance->compute_stream;
        } else {
            instance = new stream_singleton();
            return instance->compute_stream;
        }
    }

    static cudaStream_t get_cpu2gpu_stream() {
        if (instance != NULL) {
            return instance->cpu2gpu_stream;
        } else {
            instance = new stream_singleton();
            return instance->cpu2gpu_stream;
        }
    }

    static cudaStream_t get_gpu2cpu_stream() {
        if (instance != NULL) {
            return instance->gpu2cpu_stream;
        } else {
            instance = new stream_singleton();
            return instance->gpu2cpu_stream;
        }
    }

    // we must call the destructor explicitly
    static void destory_stream() {
        delete instance;
    }

};

}

#endif //ATP_STREAM_CONTROL_H
