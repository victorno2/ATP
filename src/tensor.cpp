#include <util/common.h>
#include <tensor.h>
#include <cublas_alias.h>
#include <util/mem_util.h>
#include <thread>


namespace ATP{

//PRIVATE METHODS

template <class value_type>
inline void tensor_t<value_type>::check_state(mem_mode target) {
#ifdef DEBUG
    mem_mode curt = this->get_state();
    if (curt != target) {
        printf("err state: tensor %p current state is : %d, target state is : %d\n", this, curt, target);
    }
#endif
}

template <class value_type>
inline void tensor_t<value_type>::atomic_set_state(int new_val) {
    int old_val = this->state.load();
#ifdef DEBUG
    printf("^^change state: layer %d tensor %p, %d -> %d\n", layer_id, this, old_val, new_val);
#endif
    while( !(this->state.compare_exchange_strong( old_val, new_val ) ) ) {
        old_val = this->state.load();
    };
}

template <class value_type>
inline mem_mode tensor_t<value_type>::get_state() {
    return (mem_mode) this->state.load();
}

//PUBLIC METHODS

template <class value_type>
void tensor_t<value_type>::sync_cpu_to_gpu() {
    /**
     * sync the async data transfer
     * state: CPU2GPU -> GPU_FUL
     */

    if (is_cpu_to_gpu_ready()) {
        this->atomic_set_state(GPU_FUL);
        return;
    }

    check_state(CPU2GPU);

    checkCudaErrors( cudaEventSynchronize(this->cpu2gpu_event) );
    while (!is_cpu_to_gpu_ready()) { }
    this->atomic_set_state(GPU_FUL);

#ifdef LRU_ON
    if (this->get_type() == DATA) {
        lru->update(this);
    }
#endif
}

template <class value_type>
void tensor_t<value_type>::sync_gpu_to_cpu() {
    /**
     * sync the async data transfer
     * state: GPU2CPU -> GPU_FUL
     */

    if (is_gpu_to_cpu_ready()) {
        return;
    }

    check_state(GPU2CPU);

    checkCudaErrors( cudaEventSynchronize(this->gpu2cpu_event) );
    while (!is_gpu_to_cpu_ready()) { }
    this->atomic_set_state(GPU_FUL);
}

template <class value_type>
void tensor_t<value_type>::init_tensor_data(value_type data, bool to_gpu) {
    size_t total = N*C*H*W;
    for (size_t i = 0; i < total; i++) {
        *(cpu_ptr + i) = data;
    }
    if (to_gpu) {
        stash_gpu_space();
        CPUtoGPU();
    }
}

template <class value_type>
void tensor_t<value_type>::GPUtoCPU() {
    /**
     * Sync GPU to CPU
     * state : GPU_FUL
     */
    check_state(GPU_FUL);

    assert(this->cpu_ptr != NULL);
    assert(this->gpu_ptr != NULL);
    // long total = this->N*this->C*this->H*this->W;

	// printf("GPUtoCPU : %p layer %d type %d, cpu_ptr=%x, gpu_ptr=%x\n", this, this->get_layer_id(), this->get_type(), this->cpu_ptr, this->gpu_ptr);

    // checkCudaErrors( cudaMemcpy((void*) this->cpu_ptr, (void*) this->gpu_ptr, total*sizeof(value_type), cudaMemcpyDeviceToHost) );
    checkCudaErrors( cudaMemcpy((void*) this->cpu_ptr, (void*) this->gpu_ptr, get_mem_size(), cudaMemcpyDeviceToHost) );
	
#ifdef DEBUG
    printf("GPUtoCPU : %p layer %d type %d\n", this, this->get_layer_id(), this->get_type());
#endif
}

template <class value_type>
void tensor_t<value_type>::CPUtoGPU() {
    /**
     * Sync CPU to GPU
     * state : VOID, CPU, GPU_NIL, RECOMPUTE -> GPU_FUL
     */
    assert(this->cpu_ptr != NULL);
    assert(this->gpu_ptr != NULL);
    long total = this->N*this->C*this->H*this->W;
    checkCudaErrors( cudaMemcpy((void*) this->gpu_ptr, (void*) this->cpu_ptr, total*sizeof(value_type), cudaMemcpyHostToDevice) );

//     if (data_t == DATA) {
// 		// 如果是数据
//         into_cnt += 1; 
//     }

//     if (this->get_state() == GPU_FUL) {
// 		// 如果GPU有数据
//         if (data_t == DATA) {
//             hit_cnt += 1;
//         }
//         // we do nothing because GPU has valid data
//         return;
//     }

//     bool is_void_pre = this->get_state() == VOID;
//     bool is_recompute_pre = this->get_state() == RECOMPUTE;

//     if (this->gpu_ptr == NULL) {
// 		// gpu内存指针为空
//         stash_gpu_space();  // 申请显存空间
//     }

// 	// 如果该tensor是重计算
//     if (is_void_pre || is_recompute_pre) {
//         this->atomic_set_state(GPU_FUL); // 设置gpu中有该tensor
//         if (data_t == DATA) {
//             hit_cnt += 1;
//         }

// // 使用LRU策略
// #ifdef LRU_ON
//         if (this->get_type() == DATA) {
//             lru->update(this); // 如果tensor是data，更新lru列表
//         }
// #endif

//         return;
//     }

//     check_state(GPU_NIL);  // 该tensor的显存是否有无效数据

//     if (data_t == DATA) {
//         miss_cnt += 1;
//     }

	// // 申请显存
    // long total = this->N*this->C*this->H*this->W;
    // checkCudaErrors( cudaMemcpy((void*) this->gpu_ptr, (void*) this->cpu_ptr, total*sizeof(value_type), cudaMemcpyHostToDevice) );

#ifdef DEBUG
    printf("CPUtoGPU : %p layer %d type %d\n", this, this->get_layer_id(), this->get_type());
#endif

    this->atomic_set_state(GPU_FUL);  // 设置GPU中有有效数据 

#ifdef LRU_ON
    if (this->get_type() == DATA) {
        lru->update(this);
    }
#endif
}

template <class value_type>
void tensor_t<value_type>::async_cpu_to_gpu() {
    /**
     * Async CPU to GPU
     * state : GPU_NIL -> CPU2GPU
     */
    assert(this->cpu_ptr != NULL);

    if (this->get_state() == GPU_FUL) {
        // we do nothing because GPU has valid data
        return;
    }

    if (this->gpu_ptr == NULL) {
        stash_gpu_space();
    }

    check_state(GPU_NIL);

    // this->atomic_set_state(CPU2GPU);
    long total = this->N*this->C*this->H*this->W;
    checkCudaErrors(cudaMemcpyAsync((void*) this->gpu_ptr, (void*)this->cpu_ptr,
                                    total* sizeof(value_type), cudaMemcpyHostToDevice, stream_singleton::get_cpu2gpu_stream()));
    // checkCudaErrors(cudaEventRecord(this->cpu2gpu_event, stream_singleton::get_cpu2gpu_stream()));
}

template <class value_type>
void tensor_t<value_type>::async_gpu_to_cpu() {
    /**
     * Async CPU to GPU
     * state : GPU_FUL -> GPU2CPU
     */
    assert(this->cpu_ptr != NULL);
    assert(this->gpu_ptr != NULL);

    check_state(GPU_FUL);
    // cudaEvent_t event_s;
    // cudaEvent_t event_e;
    // checkCudaErrors(cudaEventCreate(&event_s));
    // checkCudaErrors(cudaEventCreate(&event_e));
    // this->atomic_set_state(GPU2CPU);
    long total = this->N*this->C*this->H*this->W;
    // double start = get_cur_time();
    // checkCudaErrors(cudaEventRecord(event_s, stream_singleton::get_gpu2cpu_stream()));
    checkCudaErrors(cudaMemcpyAsync((void*) this->cpu_ptr, (void*)this->gpu_ptr,
                                    total* sizeof(value_type), cudaMemcpyDeviceToHost, stream_singleton::get_gpu2cpu_stream()));
    // checkCudaErrors(cudaEventRecord(event_e, stream_singleton::get_gpu2cpu_stream()));
    // checkCudaErrors(cudaEventSynchronize(event_s));
    // checkCudaErrors(cudaEventSynchronize(event_e));
    // double end = get_cur_time();
    // float time;
    // checkCudaErrors(cudaEventElapsedTime(&time, event_s, event_e));
    // printf("in tensor%d size = %zd, time = %lf\n", this->get_tensor_id(), total* sizeof(value_type), time/1000.0);
}    

template <class value_type>
inline bool tensor_t<value_type>::is_cpu_to_gpu_ready() {
    /**
     * check if the async cpu 2 gpu finish.
     * state : CPU2GPU -> GPU
     */
    if (cpu2gpu_event_not_happen.load()) {
        return true;
    }

    check_state(CPU2GPU);

    cudaError_t r = cudaEventQuery(this->cpu2gpu_event);
    if (r == cudaSuccess) {
        cpu2gpu_event_not_happen = true;

        this->atomic_set_state(GPU_FUL);

#ifdef LRU_ON
#undef LRU_ON
#endif

#ifdef LRU_ON
        if (this->get_type() == DATA) {
            lru->update(this);
        }
#endif

        return true;
    } else if (r == cudaErrorNotReady) {
        return false;
    } else {
        fprintf(stderr, "error when checking cpu2gpu_event, error message : %s\n", cudaGetErrorString(r));
        return false;
    }
}

template <class value_type>
inline bool tensor_t<value_type>::is_gpu_to_cpu_ready() {
    /**
     * check if async gpu 2 cpu finish.
     * state : GPU2CPU -> GPU
     */
    if (gpu2cpu_event_not_happen.load()) {
        return true;
    }

    check_state(GPU2CPU);

    cudaError_t r = cudaEventQuery(this->gpu2cpu_event);
    if (r == cudaSuccess) {
        gpu2cpu_event_not_happen = true;

        this->atomic_set_state(GPU_FUL);

        return true;
    } else if (r == cudaErrorNotReady) {
        return false;
    } else {
        fprintf(stderr, "error when checking cpu2gpu_event, error message : %s\n", cudaGetErrorString(r));
        return false;
    }
}

//------GPU functions-----//
template <class value_type> //ptr1 = ptr1 + ptr2
void tensor_sum(value_type* ptr1, value_type* ptr2, int size);

template <class value_type> //copy ptr1 to ptr2
void tensor_copy(value_type* ptr1, value_type* ptr2, int size);

template <class value_type> //ptr1 = ptr1 * s
void tensor_scale(value_type* ptr1, value_type s, int size);
//-----------------------//

template <class value_type>
void tensor_t<value_type>::sum(tensor_t<value_type>* t) {
    size_t len = this->N*this->C*this->H*this->W;
    value_type one = 1.0;
    tensor_sum(this->get_gpu_ptr(), t->get_gpu_ptr(), len);
}

template <class value_type>
value_type tensor_t<value_type>::squared_sum(cublasHandle_t *handle) {
    size_t len = this->N*this->C*this->H*this->W;
    value_type squared_sum = 0;
    value_type result = 0;
    cublas_dot(handle, this->get_scalar_count(), this->get_gpu_ptr(), 1, this->get_gpu_ptr(), 1, &result);
    return result;
}

template <class value_type>
void tensor_t<value_type>::copy(tensor_t<value_type>* t,
                                int src_start_idx, int src_end_idx,
                                int dst_start_idx, int dst_end_idx) {
    size_t len = 0, offset_dst = 0, offset_src = 0;
    if ((src_start_idx == -1) && (src_end_idx == -1) && ( dst_start_idx == -1) && (dst_end_idx == -1)) {
        len = this->N * this->C * this->H * this->W;
    }
    if ((src_start_idx >= 0) && (src_end_idx >= 0)) {
        len = (size_t) (src_end_idx - src_start_idx);
        offset_src = (size_t) src_start_idx;
    }
    if ((dst_start_idx >= 0) && (dst_end_idx >= 0)) {
        if (len != 0) {
            if (len != (size_t)(dst_end_idx - dst_start_idx)) {
                fprintf(stderr, "tensor copy size does not match, src len: %zu, dst len: %d\n", len, dst_end_idx - dst_start_idx);
            }
        } else {
            len = (size_t) (dst_end_idx - dst_start_idx);
        }
        offset_dst = (size_t) dst_start_idx;
    }
    // TODO : this memcpy is with error in loss decrease
//    cudaMemcpy(this->get_gpu_ptr()+offset_dst, t->get_gpu_ptr()+offset_src, len, cudaMemcpyDeviceToDevice);
    tensor_copy(t->get_gpu_ptr()+offset_src, this->get_gpu_ptr()+offset_dst, len);
}

template <class value_type>
void tensor_t<value_type>::scale(value_type s) {
    size_t len = this->N*this->C*this->H*this->W;
    tensor_scale(this->get_gpu_ptr(), s, len);
}

template <class value_type>
void tensor_t<value_type>::hostRegister() {
    if (this->gpu_ptr != NULL) {
        long total = this->N * this->C * this->H * this->W;
        checkCudaErrors( cudaHostRegister(this->cpu_ptr, total*sizeof(value_type), cudaHostRegisterPortable) );
    }
}

#define PRINT_TENSOR
template <class value_type>
void tensor_t<value_type>::printTensorInt(const char* str) {
#ifdef PRINT_TENSOR
    printf("\n@:%s Tensor%d: N:%zu C:%zu H:%zu W:%zu\n", str, this->get_tensor_id(), this->N, this->C, this->H, this->W);
    // printf("PRINT OUT TENSOR %p N:%zu C%zu H:%zu W:%zu@:%s\n", this, this->N, this->C, this->H, this->W, str);
    // GPUtoCPU();
    size_t total = this->N*this->C*this->H*this->W;
    // this->cpu_ptr_int = (int*)malloc(sizeof(int)*total);
    // checkCudaErrors( cudaMemcpy((void*) this->cpu_ptr_int, (void*) this->gpu_ptr_int, total*sizeof(int), cudaMemcpyDeviceToHost) );
    for (size_t d0 = 0; d0 < Dims[0]; d0++) {
        printf("TimeStep[%d]:\n", d0);
        for (size_t d1 = 0; d1 < Dims[1]; d1++) {
            printf("Batch[%d]: ", d1);
            for (size_t d2 = 0; d2 < Dims[2]; d2++) {
                printf("%d ", this->cpu_ptr_int[(d0*Dims[1]+d1)*Dims[2]+d2]);
            }
            printf("\n");
        }
        // printf("\n");
    }
    // for(size_t n = 0; n < this->N; n++) {
    //     printf("#################### CPU n:%zu ####################\n", n);
    //     for (size_t c = 0; c < this->C; c++) {
    //         printf("--------c:%zu--------\n", c);
    //         for (size_t h = 0; h < this->H; h++) {
    //             for (size_t w = 0; w < this->W; w++) {
    //                 //float and double
    //                 printf(" %d, ", this->cpu_ptr_int[((n*C+c)*H+h)*W+w]);
    //             }
    //             printf("\n");
    //         }
    //     }
    // }
    printf("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n");
#endif
#undef PRINT_TENSOR
}

#define PRINT_TENSOR
template <class value_type>
void tensor_t<value_type>::printTensor(const char* str) {
#ifdef PRINT_TENSOR
    printf("\n@:%s Tensor%d: N:%zu C:%zu H:%zu W:%zu\n", str, this->get_tensor_id(), this->N, this->C, this->H, this->W);
    // printf("PRINT OUT TENSOR %p N:%zu C%zu H:%zu W:%zu@:%s\n", this, this->N, this->C, this->H, this->W, str);
    // value_type* temp = this->cpu_ptr;
    // this->cpu_ptr = (value_type*)malloc(this->N * this->C * this->H * this->W * sizeof(value_type) * 10000);
    GPUtoCPU();
    for(size_t n = 0; n < this->N; n++) {
        printf("#################### CPU n:%zu ####################\n", n);
        for (size_t c = 0; c < this->C; c++) {
            printf("--------c:%zu--------\n", c);
            for (size_t h = 0; h < this->H; h++) {
                for (size_t w = 0; w < this->W; w++) {
                    //float and double
                    printf(" %3.4f, ", this->cpu_ptr[((n*C+c)*H+h)*W+w]);
                }
                printf("\n");
            }
        }
    }
    // free(this->cpu_ptr);
    // this->cpu_ptr = temp;
    printf("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n");
#endif
#undef PRINT_TENSOR
}

template <class value_type>
void tensor_t<value_type>::printTensorData(const char* str, int m) {
    printf("\n%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n");
    printf("@:%s Tensor%d: N:%zu C:%zu H:%zu W:%zu gpu_ptr = %x ", str, this->get_tensor_id(), this->N, this->C, this->H, this->W, get_gpu_ptr());
    // this->cpu_ptr = (value_type*)this->cpu_ptr;
    GPUtoCPU();
    size_t total = this->N * this->C * this->H * this->W;
    if (m == 1) {
        for (size_t i = 0; i < total; i++) {
            printf("%3.3f ", this->cpu_ptr[i]);
        }
        printf("total = %zd\n", total);
    }
    else {
        value_type sum = 0;
        for (size_t i = 0; i < total; i++) {
            sum += this->cpu_ptr[i];
        }
        printf("normal_v2 = %f, total = %zd\n", sum, total);
    }
    
}

template <class value_type>
void tensor_t<value_type>::printRNNTensor(const char* str) {
    printf("\n%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n");
    printf("PRINT OUT TENSOR D1:%zu D2:%zu D3:%zu @:%s\n", Dims[0], Dims[1], Dims[2], str);
    GPUtoCPU();
    for (size_t d0 = 0; d0 < Dims[0]; d0++) {
        printf("TimeStep[%d]:\n", d0);
        for (size_t d1 = 0; d1 < Dims[1]; d1++) {
            printf("Batch[%d]: ", d1);
            for (size_t d2 = 0; d2 < Dims[2]; d2++) {
                printf("%f ", this->cpu_ptr[(d0*Dims[1]+d1)*Dims[2]+d2]);
            }
            printf("\n");
        }
        // printf("\n");
    }
    printf("\n%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n");
}

template <class value_type>
void tensor_t<value_type>::printTensorNoDebug(const char* str) {
    printf("\n\n%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n");
    printf("PRINT OUT TENSOR N:%zu C:%zu H:%zu W:%zu@:%s\n", this->N, this->C, this->H, this->W, str);
    // GPUtoCPU();
    for(size_t n = 0; n < this->N; n++) {
        printf("#################### CPU n:%zu ####################\n", n);
        for (size_t c = 0; c < this->C; c++) {
            printf("--------c:%zu--------\n", c);
            for (size_t h = 0; h < this->H; h++) {
                for (size_t w = 0; w < this->W; w++) {
                    //float and double
                    printf(" %3.5f ", this->cpu_ptr[((n*C+c)*H+h)*W+w]);
                }
                printf("\n");
            }
        }
    }
    printf("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n");
}
    
template <class value_type>
void tensor_t<value_type>::writeToFile(const char* str) {
    printf("\n\n%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n");
    printf("PRINT OUT TENSOR N:%zu C:%zu H:%zu W:%zu@:%s\n", this->N, this->C, this->H, this->W, str);
    FILE *fp;
    fp = fopen(str, "a");
    GPUtoCPU();
    for(size_t n = 0; n < this->N; n++) {
        //fprintf(fp, "#################### CPU n:%zu ####################\n", n);
        for (size_t c = 0; c < this->C; c++) {
            //fprintf(fp, "--------c:%zu--------\n", c);
            for (size_t h = 0; h < this->H; h++) {
                for (size_t w = 0; w < this->W; w++) {
                    //float and double
                    fprintf(fp, "%f ", this->cpu_ptr[((n*C+c)*H+h)*W+w]);
                }
                //fprintf(fp, "\n");
            }
        }
    }
    fclose(fp);
}


template <class value_type>
void tensor_t<value_type>::printTensorFirst(const char* str) {
    printf("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n");
	this->cpu_ptr = (value_type*)malloc(sizeof(value_type) * this->N * this->C * this->H * this->W);
    printf("PRINT OUT TENSOR N:%zu C:%zu H:%zu W:%zu@:%s\n", this->N, this->C, this->H, this->W, str);
//        size_t total = this->N*this->C*this->H*this->W;
        GPUtoCPU();
        for(size_t n = 0; n < 1; n++) {
            printf("#################### CPU n:%zu ####################\n", n);
            for (size_t c = 0; c < this->C; c++) {
                printf("--------c:%zu--------\n", c);
                for (size_t h = 0; h < this->H; h++) {
                    for (size_t w = 0; w < this->W; w++) {
                        //float and double
                        //printf(" %2.0f ", this->cpu_ptr[((n*C+c)*H+h)*W+w]);
                    }
                    // printf("\n");
                }
            }
        }
        printf("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n");
		free(this->cpu_ptr);
}

template <class value_type>
void tensor_t<value_type>::printTensorState(const char* str) {
    printf("%s tensor%d-layer%d:", str, this->get_tensor_id(), this->get_layer_id());
    switch(this->data_p) {
        case REMAIN_IN_GPU: printf(" REMAIN_IN_GPU "); break;
        case SHARED_GPU_POOL: printf(" SHARED_GPU_POOL "); break;
        case RECOMPUTE_IN_BACKWARD: printf(" RECOMPUTE_IN_BACKWARD "); break;
        default: printf(" **** ");
    }
    printf(",");
    switch(this->data_ds) {
        case NO_COMPUTED: printf(" NO_COMPUTED "); break;
        case STILL_USED: printf(" STILL_USED "); break;
        case FORWARD_DELETE_OK: printf(" FORWARD_DELETE_OK "); break;
        default: printf(" **** ");
    }
    printf(",");
    switch(this->data_dp) {
        case IN_GPU: printf(" IN_GPU "); break;
        case IN_CPU: printf(" IN_CPU "); break;
        case DELETED: printf(" DELETED "); break;
        case IN_CPU_GPU: printf(" IN_CPU_GPU "); break;
        case NO_DATA: printf(" NO_DATA "); break;
        default: printf(" **** ");
    }
    printf(", type%d, fc = %d/%d, bc = %d/%d ", this->data_t, forward_cur_use_counter, forward_use_counter, backward_cur_use_counter, backward_use_counter);
    printf("\n");
}

template <class value_type>
void tensor_t<value_type>::resizeTensor(size_t n, size_t c, size_t h, size_t w) {
    /**
     * state : not change
     */
    assert(n >= 1);
    assert(c >= 1);
    assert(h >= 1);
    assert(w >= 1);

//    bool flag = this->gpu_ptr != NULL;
    freeSpaceGPU();

//    if (flag) {
        acquireSpaceGPU(n * c * h * w);
//    }

    freeSpaceCPU();

#ifdef LIVENESS
    if (this->data_t != CONV_BUFF) {
        acquireSpaceCPU(n * c * h * w);
    }
#else
    acquireSpaceCPU(n * c * h * w);
#endif

    this->N = n;
    this->C = c;
    this->H = h;
    this->W = w;
    
    CHECK_GT( (int) n, 0);
    CHECK_GT( (int) c, 0);
    CHECK_GT( (int) h, 0);
    CHECK_GT( (int) w, 0);
    
    checkCUDNN( cudnnDestroyTensorDescriptor(cudnn_tensor_desc) );
    checkCUDNN( cudnnCreateTensorDescriptor(&cudnn_tensor_desc) );
    checkCUDNN( cudnnSetTensor4dDescriptor(this->cudnn_tensor_desc,
                                           this->cudnn_tensor_format,
                                           this->cudnn_data_type,
                                           n, c, h, w) );
}

template <class value_type>
value_type tensor_t<value_type>::get_scalar(const size_t n, const size_t c, const size_t h, const size_t w)
{
    assert( n < N );
    assert( c < C );
    assert( h < H );
    assert( w < W );
    GPUtoCPU();
    return (this->cpu_ptr[((n*C+c)*H+h)*W+w]);
}

template <class value_type>
void tensor_t<value_type>::set_scalar(const size_t n, const size_t c, const size_t h, const size_t w, value_type t)
{
    assert( n < N );
    assert( c < C );
    assert( h < H );
    assert( w < W );
    GPUtoCPU();
    this->cpu_ptr[((n*C+c)*H+h)*W+w] = t;
    CPUtoGPU();
}

template <class value_type>
void tensor_t<value_type>::init(initializer_t<value_type> *initializer) {
    initializer->call(this->cpu_ptr, this->N, this->C, this->H, this->W);
#ifndef POOL_MALLOC_MODE
    // CPUtoGPU();
    // GPUtoCPU();
    // printf("tensor%d from layer%d init param\n", this->get_layer_id());
    // this->printTensor("s");
    // while(1);
#endif

    // TODO : the initializer should be used only once !!!
//#pragma GCC diagnostic ignored "-Wdelete-non-virtual-dtor"
//    delete initializer;
}

template <class value_type>
void tensor_t<value_type>::acquireSpecifiedSpaceGPU(size_t total, value_type *specified_ptr, size_t offset) {
	// if ( gpu_ptr != NULL) {
    //     return;
    // }
    assert( total > 0 );
	this->gpu_ptr = specified_ptr + offset + sizeof(value_type)*total;
	// gmalloc_specified_space(gpu_malloc, &(this->gpu_ptr), sizeof(value_type)*total, (specified_ptr + _offset));
}

template <class value_type>
void tensor_t<value_type>::acquireSpaceGPU(size_t total) {
    // if ( gpu_ptr != NULL) {
    //     return;
    // }
    assert( total > 0 );

//    printf("before malloc %zu byte\n", query_free_mem());
    gmalloc(gpu_malloc, &(this->gpu_ptr), sizeof(value_type)*total);
//    printf("after malloc %zu byte\n", query_free_mem());

    if (this->gpu_ptr != NULL) {
        this->atomic_set_state(GPU_NIL);
    }

    if (data_t != DATA && data_t != CONV_BUFF) {
        return;
    }

//    if (data_t != CONV_BUFF) {
//        into_cnt += 1;
//    }
//
//    if (data_t != CONV_BUFF && this->gpu_ptr != NULL) {
//        hit_cnt += 1;
//    }
#ifdef LRU_ON
#undef LRU_ON
#endif

#ifdef LRU_ON
//    if (this->gpu_ptr == NULL && data_t != CONV_BUFF) {
//        miss_cnt += 1;
//    }

    while (this->gpu_ptr == NULL) {
        printf("LRU start!! tensor %p, current free memory %zu\n", this, query_free_mem());
        lru->print_list();
        int x = 0;
        while (lru->get_item(x) != NULL) {
            tensor_t<value_type>* t = (tensor_t<value_type>*)lru->get_item(x)->item;
            printf("tensor %p layer %d\n", t, t->get_layer_id());
            x += 1;
        }

        // kick out some tensors in LRU
        tensor_t<value_type> *t = (tensor_t<value_type> *) (lru->remove_oldest());
        if (t == NULL) {
            lru->print_list();
            fprintf(stderr, "LRU NULL !!! Can not alloc GPU memory !!!! tensor %p need %zu free %zu\n", this, total, query_free_mem());
            exit(-1);
        }
        printf("kick out tensor %p layer %d\n", t, t->get_layer_id());
        if (t->get_state() == GPU_FUL) {
            t->GPUtoCPU();
            t->free_gpu_space(CPU);
        } else {
            fprintf(stderr, "Found a not GPU tensor in LRU %p\n", t);
            fprintf(stderr, "tensor: %p, layer %d type: %d, state: %d\n", t, t->get_layer_id(), t->get_type(), t->get_state());
            lru->print_list();
            exit(-1);
        }

//#ifdef DEBUG
        printf("kick out oldest? \n");
        lru->print_list();
//#endif

        gmalloc(gpu_malloc, &(this->gpu_ptr), sizeof(value_type) * total);
    }

    if (this->gpu_ptr != NULL) {
        if (this->get_state() != GPU_NIL) {
            this->atomic_set_state(GPU_NIL);
        }
        if (this->get_type() == DATA) {
            lru->update(this);
        }
    }

#endif

}

template <class value_type>
void tensor_t<value_type>::freeSpaceGPU(mem_mode target) {
    if (this->cpu_ptr == NULL) {
        this->atomic_set_state(VOID);
    } else {
        this->atomic_set_state(target);
    }

    if (gpu_ptr == NULL) {
        return;
    }
#ifdef DEBUG
    printf("free tensor %p layer %d gpu %p  curt: %d target: %d\n", this, this->get_layer_id(), gpu_ptr, get_state(), target);
#endif

    gfree(gpu_malloc, this->gpu_ptr);
    this->gpu_ptr = NULL;

#ifdef LRU_ON
    if (this->get_type() == DATA) {
        lru->remove_item(lru->find(this));
    }
#endif
}

template <class value_type>
void tensor_t<value_type>::acquireSpaceCPU(size_t total) {
    assert( cpu_ptr == NULL );
        assert( total > 0 );
        checkCudaErrors( cudaMallocHost(&(this->cpu_ptr), total*sizeof(value_type) ) );
}

template <class value_type>
void tensor_t<value_type>::freeSpaceCPU() {
        if (cpu_ptr == NULL) {
            return;
        }
        checkCudaErrors(cudaFreeHost(this->cpu_ptr));
        this->cpu_ptr = NULL;
}

template <class value_type>
void tensor_t<value_type>::freeSpecifiedSpaceGPU(mem_mode target) {
    if (this->cpu_ptr == NULL) {
        this->atomic_set_state(VOID);
    } else {
        this->atomic_set_state(target);
    }

    if (gpu_ptr == NULL) {
        return;
    }
#ifdef DEBUG
    printf("free tensor %p layer %d gpu %p  curt: %d target: %d\n", this, this->get_layer_id(), gpu_ptr, get_state(), target);
#endif

    gfree(gpu_malloc, this->gpu_ptr);
    this->gpu_ptr = NULL;

}

template <class value_type>
void tensor_t<value_type>::replace_data(value_type *new_cpu_ptr, value_type *new_gpu_ptr) {

    if (new_cpu_ptr != NULL) {
        value_type *old_cpu_ptr = this->cpu_ptr;
        this->cpu_ptr = new_cpu_ptr;
        checkCudaErrors(cudaFreeHost(old_cpu_ptr));

        if (new_gpu_ptr == NULL) {
            CPUtoGPU();
        }
    }

    if (new_gpu_ptr != NULL) {
        value_type *old_gpu_ptr = this->gpu_ptr;
        this->gpu_ptr = new_gpu_ptr;

        // remember to free the old ptr
        checkCudaErrors(cudaFree(old_gpu_ptr));
    }
}
    
/*---math functions-------*/
template <class value_type>
void tensor_t<value_type>::forward_fft() {
    CHECK_EQ(this->data_t, GRAD);
    CHECK_EQ( cufftExecR2C(fft_plan_f, (cufftReal*) this->gpu_ptr, (cufftComplex*) this->freq_ptr ), CUFFT_SUCCESS );
    const size_t total_size = this->get_scalar_count();
}
    
template <class value_type>
void tensor_t<value_type>::backward_fft() {
    CHECK_EQ(this->data_t, GRAD);
    CHECK_EQ( cufftExecC2R(fft_plan_b, (cufftComplex*) this->freq_ptr, (cufftReal*) this->gpu_ptr), CUFFT_SUCCESS );
    const value_type rescale_factor = 1.0f / (value_type) this->get_scalar_count();
    this->scale( rescale_factor );
}

template <class value_type>
void tensor_t<value_type>::reset_all_state() {
    this->swap_block_id = -1;
    this->prefetch_block_id = -1;
    this->set_position(REMAIN_IN_GPU);
	this->set_data_state(NO_COMPUTED);
	this->set_data_position(NO_DATA);
    this->cur_use_count = this->use_count;
	this->backward_cur_use_counter = 0;
	this->forward_cur_use_counter = 0;
}

template <class value_type>
void tensor_t<value_type>::reset_block_allocation() {
    this->swap_block_id = -1;
    this->prefetch_block_id = -1;
    this->recompute_pool_id = -1;
    this->backward_recompute_pool_id = -1;
}

template <class value_type>
void tensor_t<value_type>::reset_all_data_state() {
	this->set_data_state(NO_COMPUTED);
	this->set_data_position(NO_DATA);
    this->cur_use_count = this->use_count;
	this->backward_cur_use_counter = 0;
	this->forward_cur_use_counter = 0;
}

template <class value_type>
double tensor_t<value_type>::GetRealBandwidth(size_t size, COMM_DIRECTION direction) {  // direction = 0:gpu2cpu or 1:cpu2gpu
	size_t left = 0;
	size_t right = SIZE_LIST_SIZE - 1;
	size_t loc = 0;
	double bandwidth;
	for (left = 0; left <= right; left++) {
		loc = left;
		if (size < size_list[left]) {
			break;
		}
    }
	if (direction == COMM_GPU2CPU) {
		return Bs_gpu2cpu[loc];
	}
	else {  // direction == COMM_CPU2GPU
		return Bs_cpu2gpu[loc];
	}
}

template <class value_type>
double tensor_t<value_type>::bandwidth_bisearch(size_t size) {
    int left = 0;
    int right = SIZE_LIST_SIZE - 1;
    size_t flag = 0;
    double bandwidth;
    for (left = 0; left <= right; left++) {
        if (size < size_list[left]) {
            flag = 1;
            break;
        }
    }
    left = left - 1;
    bandwidth = Bs_gpu2cpu[left] < Bs_cpu2gpu[left] ? Bs_gpu2cpu[left] : Bs_cpu2gpu[left];
    // printf("bandwidth = %lf\n", bandwidth);
    return bandwidth;
}
    
template <class value_type>
size_t tensor_t<value_type>::tensor_counter = 0;

template <class value_type>
size_t tensor_t<value_type>::tensor_malloc_count = 0;

template <class value_type>
size_t tensor_t<value_type>::gpu_pool_offset = 0;
template <class value_type>
void* tensor_t<value_type>::gpu_pool_ptr = NULL;
template <class value_type>
size_t tensor_t<value_type>::gpu_pool_size = 0;



INSTANTIATE_CLASS(tensor_t);

} // ATP namespace
