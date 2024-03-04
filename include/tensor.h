#if !defined(_TENSOR_H_)
#define _TENSOR_H_
#include <vector>
#include <cassert>
#include <cudnn.h>
#include <stdio.h>
#include <switch.h>
#include <random>
#include <sys/time.h>
#include <chrono>
#include <math.h>
#include <atomic>
#include <util/error_util.h>
#include <util/common.h>
#include <initializer.h>
#include <stream_singleton.h>
#include <gpu_malloc.h>
#include <util/lru.h>
#include <cufft.h>
#include <util/mem_util.h>
#include <bandwidth.h>

//#define BLASX_MALLOC

namespace ATP{
    
typedef enum TENSOR_TYPE {
    DATA        	= 0,
    GRAD        	= 1,
    PARAM       	= 2,
    AUX         	= 3,
    BN_MEAN_VAR 	= 4,
    CONV_BUFF   	= 5,
    DATA_SOURCE 	= 6,
	B_DATA			= 7,
	RNN_BUFF    	= 8,
	RNN_DATA    	= 9,
	RNN_PARAM   	= 10,
	RNN_GRAD		= 11,
	RNN_B_DATA		= 12,
	DATA_SOURCE_INT = 13,
	DATA_INT		= 14,
	RNN_RESERVE		= 15,
	QKV_DATA		= 16,
	DQKV_DATA		= 17,
} TENSOR_TYPE;

typedef enum SWAP_DIR {
    OFFLOAD = 0,
    FETCH = 1
} SWAP_DIR;

typedef enum TENSOR_MEM_STATE {
    DATA_MEM_READY = 0,
    DATA_MEM_NO_READY = 1
} TENSOR_MEM_STATE;

typedef enum TENSOR_POSITION {
	REMAIN_IN_GPU			= 0,
	SHARED_GPU_POOL		  	= 1,
	RECOMPUTE_IN_BACKWARD 	= 2
} TENSOR_POSITION;

typedef enum TENSOR_DATA_STATE {
    NO_COMPUTED = 0,
	STILL_USED = 1,
	FORWARD_DELETE_OK = 2,
    // IN_GPU = 1,
    // IN_CPU = 2,
    // FORWARD_DELETE_OK = 3,
} TENSOR_DATA_STATE;
    
typedef enum TENSOR_DATA_POSITION {
	IN_GPU 		= 0,
	IN_CPU 		= 1,
	DELETED 	= 2,
	IN_CPU_GPU	= 3,
	NO_DATA		= 4
} TENSOR_DATA_POSITION;
	
typedef enum TENSOR_DATA_LAYOUT {
	IMAGE_NCHW 		= 0,
	SEQ2SEQ_TNBV	= 1
} TENSOR_DATA_LAYOUT;
	
template <class value_type>
class tensor_t {
private:

    std::atomic<int> state;  // std::atomic 模板的实例化和全特化定义一个原子类型
    
	TENSOR_DATA_LAYOUT data_layout = IMAGE_NCHW;
    TENSOR_TYPE    data_t;
    value_type*    gpu_ptr  = NULL;                  //gpu and cpu data are mutually exclusive
    value_type*    cpu_ptr  = NULL;
    int* 		   gpu_ptr_int  = NULL;
	int* 		   cpu_ptr_int  = NULL;
	cufftComplex*  freq_ptr = NULL;
	value_type*	   temp_gpu_ptr = NULL;

	bool use_in_backward = true;
	int swap_block_id = -1;
	int prefetch_block_id = -1;
 
    static size_t gpu_pool_offset;
    static void* gpu_pool_ptr;
    static size_t gpu_pool_size;
    size_t align_size = 512;  // 512 bytes
    size_t total_size_bytes;
    double bandwidth;
    double comm_time;
	double gpu2cpu_time;
	double cpu2gpu_time;
    
	TENSOR_POSITION data_p = REMAIN_IN_GPU;  // tensor data position
    TENSOR_MEM_STATE data_ms;
	// std::atomic<TENSOR_DATA_STATE> data_ds;// = NO_COMPUTED;
    TENSOR_DATA_STATE data_ds = NO_COMPUTED;
	// std::atomic<TENSOR_DATA_POSITION> data_dp;
	TENSOR_DATA_POSITION data_dp = NO_DATA;
    bool is_first_pool_tensor = false;
    bool is_last_pool_tensor = false;
	bool require_space = true;

	int backward_use_counter = 0;
	int backward_cur_use_counter = 0;
	int forward_use_counter = 0;
	int forward_cur_use_counter = 0;

	int use_count = 0;  // used times during forward
	int cur_use_count = 0;  // cur_use_count = use_count at the beginning, it will reduce during forward and increase during backward

	int recompute_pool_id = -1;
	int backward_recompute_pool_id = -1;

	bool backward_useful = true;

	int b_use_count = 0;   // used times in backward
	int cur_b_use_count = 0;
	int b_data_pool_id = -1;

	// for SeqData
	int seq_time, seq_batch, seq_beam, seq_vect;

    size_t GPU_id;                               //this identifies the GPU RAM
    int layer_id;                                //this identifies the affilited layer
    size_t N;
    size_t C;
    size_t H;
    size_t W;
    /*--cufft-handle--*/
    //this has to for each tensor
    cufftHandle fft_plan_f;
    cufftHandle fft_plan_b;
	size_t Dims[3];
	size_t Strides[3];
    
	int *labelLengths;
	
    /*---tensor_id----*/
	static size_t tensor_malloc_count;
    static size_t tensor_counter;
	int tensor_malloc_rank = 0;
    int tensor_id = 0;
	size_t tensor_base_id;

	/*---simulator---*/
	size_t mem_begin;
	size_t mem_end;
    
#ifdef LRU_ON
    lru_list_t* lru = lru_singleton::get_lru();  // 返回的是一个静态的lru_list_t，这个lru是静态的，全局统一
#endif
    
    //CUDNN configuration
    cudnnDataType_t cudnn_data_type;
    cudnnTensorFormat_t cudnn_tensor_format;    //default with NCHW
    cudnnTensorDescriptor_t cudnn_tensor_desc;
    
    cudaEvent_t  cpu2gpu_event, gpu2cpu_event;
    std::atomic_bool cpu2gpu_event_not_happen, gpu2cpu_event_not_happen;

    blasx_gpu_malloc_t* gpu_malloc = NULL;
    
    void acquireSpaceGPU_v2(size_t total);
	void acquireSpecifiedSpaceGPU(size_t total, value_type *ptr, size_t offset);
    void freeSpecifiedSpaceGPU(mem_mode target=CPU);

    double bandwidth_bisearch(size_t size);

    double GetRealBandwidth(size_t size, COMM_DIRECTION direction);

    // make it private
    void atomic_set_state(int m);
    inline void check_state(mem_mode target);
    
public:

	int b_live_count = 0;   // a counter used to record b_data liveness
    int hit_cnt = 0, miss_cnt = 0, into_cnt = 0;

    tensor_t(size_t n, size_t c, size_t h, size_t w, std::vector<tensor_t<value_type>* >* reg, TENSOR_TYPE dtype, int layer_id):tensor_id(tensor_counter) {
        assert(n >= 1);
        assert(c >= 1);
        assert(h >= 1);
        assert(w >= 1);
        
		if (n <= 0) n = 1;
		if (c <= 0) n = 1;
		if (h <= 0) n = 1;
		if (w <= 0) n = 1;
		
		backward_use_counter = 0;
		backward_cur_use_counter = 0;
		forward_use_counter = 0;
		forward_cur_use_counter = 0;
		b_use_count = 0;   // used times in backward
		cur_b_use_count = 0;
		b_data_pool_id = -1;
		recompute_pool_id = -1;
		backward_recompute_pool_id = -1;
		
		data_ds = NO_COMPUTED;
		data_dp = NO_DATA;
		
        // TODO : set GPU affinity
        GPU_id = 0;
#ifdef BLASX_MALLOC
        gpu_malloc = blasx_gpu_singleton::get_blasx_gpu_malloc_t(GPU_id);
#endif
        
        this->state    = VOID;
        this->data_t   = dtype;
        this->layer_id = layer_id;

        // set tensor state
        this->data_p = REMAIN_IN_GPU;
        
        switch (sizeof(value_type))
        {
            case 2 : cudnn_data_type = CUDNN_DATA_HALF; break;
            case 4 : cudnn_data_type = CUDNN_DATA_FLOAT; break;
            case 8 : cudnn_data_type = CUDNN_DATA_DOUBLE; break;
            default : FatalError("Unsupported data type");
        }
    
        cudnn_tensor_format = CUDNN_TENSOR_NCHW;
        checkCUDNN( cudnnCreateTensorDescriptor(&cudnn_tensor_desc) );
        checkCUDNN( cudnnSetTensor4dDescriptor(this->cudnn_tensor_desc,
                                               this->cudnn_tensor_format,
                                               this->cudnn_data_type,
                                               n, c, h, w) );
											   
        const size_t total_size = n * c * h * w;

		this->N = n;
        this->C = c;
        this->H = h;
        this->W = w;
		
		Dims[0] = n; Dims[1] = c; Dims[2] = h*w;
		Strides[0] = Dims[1] * Dims[2];
		Strides[1] = Dims[2];
		Strides[2] = 1;

		this->seq_time = n;
		this->seq_batch = c;
		this->seq_beam = h;
		this->seq_vect = w;

		total_size_bytes = sizeof(value_type) * n*c*h*w;
        total_size_bytes = (size_t)(total_size_bytes / align_size) * align_size + (((total_size_bytes % align_size) > 0 ? align_size : 0));  // align to 512
        // bandwidth = bandwidth_bisearch(total_size_bytes);
        // comm_time = total_size_bytes / bandwidth;
		if (require_space==false) {  // tensor without real gpu space, such as the output of fork
			total_size_bytes = 0;
		}

		if (this->data_t != CONV_BUFF && this->data_t != RNN_BUFF) 
		{
			if (this->data_t == DATA_SOURCE_INT) {
				checkCudaErrors(cudaMallocHost(&(this->cpu_ptr_int), total_size*sizeof(int)));
				printf("cpu_ptr_int = %x\n", this->cpu_ptr_int);
			} 
			else {
			//#ifndef SWAP_MALLOC_MODE
				// acquireSpaceCPU(total_size_bytes/sizeof(value_type));
				if (data_t==PARAM || data_t==BN_MEAN_VAR || data_t==RNN_PARAM || data_t==GRAD || data_t==AUX) {
					checkCudaErrors(cudaMallocHost(&(this->cpu_ptr), total_size_bytes));
					printf("acquireSpaceCPU done cpu_ptr=%x\n", this->cpu_ptr);
				}
			//#else
				//if (data_t != DATA) {
					//acquireSpaceCPU(total_size_bytes/sizeof(value_type));
				//}
			//#endif
			}
		}

		// acquireSpaceCPU(total_size_bytes/sizeof(value_type));
		
        bandwidth = bandwidth_bisearch(total_size_bytes);
        comm_time = ((double)total_size_bytes / bandwidth);
        printf("tensor%d-type%d-layer%d [%d %d %d %d] size=%f bandwidth=%.2f comm_time=%lf\n", 
            tensor_id, this->data_t, this->get_layer_id(), get_N(), get_C(), get_H(), get_W(), BYTE_TO_MB(total_size_bytes), BYTE_TO_MB(bandwidth), comm_time);

        gpu2cpu_time = (double)total_size_bytes / GetRealBandwidth(total_size_bytes, COMM_GPU2CPU);
        cpu2gpu_time = (double)total_size_bytes / GetRealBandwidth(total_size_bytes, COMM_CPU2GPU);
        printf("tensor%d size=%zd gpu2cpu_time=%lf, cpu2gpu_time=%lf\n", tensor_id, total_size_bytes, gpu2cpu_time, cpu2gpu_time);

#ifdef SWAP_MALLOC_MODE
		/*
        if (this->data_t != DATA && this->data_t != B_DATA && this->data_t != RNN_BUFF && this->data_t != CONV_BUFF && this->data_t != RNN_RESERVE)			
		{
            this->data_dp = IN_GPU;
            acquireSpaceGPU(n*c*h*w);
			tensor_malloc_rank = tensor_malloc_count;
			tensor_malloc_count++;
        }
        else {
            this->data_ds = NO_COMPUTED;
            // this->data_ms = DATA_MEM_NO_READY;
        }
		*/
#endif

#ifdef POOL_MALLOC_MODE
		if (this->data_t == DATA || this->data_t == RNN_RESERVE ) {
			this->data_ds = NO_COMPUTED;
		}
		else {
			this->data_dp = IN_GPU;
		}
#endif
     
        reg->push_back(this);  // 这里的reg是reg->get_vector()，给reg.tensor_to_free添加一个tensor作为统计
        /*---init-event-asyn-comm--*/
        checkCudaErrors(cudaEventCreate(&cpu2gpu_event));
        checkCudaErrors(cudaEventCreate(&gpu2cpu_event));
        cpu2gpu_event_not_happen = true;
        gpu2cpu_event_not_happen = true;
		
        /*---init-counter---*/
		tensor_base_id = tensor_counter;
        tensor_counter++;
 
#ifdef DEBUG
        total_size_bytes = sizeof(value_type)*n*c*h*w;

        if(this->data_t == DATA) {
            printf("create tensor:%p DATA gpu_ptr:%p size: %zu byte\n", this, gpu_ptr, total_size_bytes);
        } else if(this->data_t == PARAM) {
            printf("create tensor:%p PARAM gpu_ptr:%p size: %zu byte\n", this, gpu_ptr, total_size_bytes);
        } else if(this->data_t == GRAD) {
            printf("create tensor:%p GRAD gpu_ptr:%p size: %zu byte\n", this, gpu_ptr, total_size_bytes);
        } else if(this->data_t == AUX) {
            printf("create tensor:%p AUX gpu_ptr:%p size: %zu byte\n", this, gpu_ptr, total_size_bytes);
        } else if(this->data_t == BN_MEAN_VAR) {
            printf("create tensor:%p BN_MEAN_VAR gpu_ptr:%p size: %zu byte\n", this, gpu_ptr, total_size_bytes);
        } else if(this->data_t == CONV_BUFF) {
            printf("create tensor:%p CONV_BUFF gpu_ptr:%p size: %zu byte\n", this, gpu_ptr, total_size_bytes);
        } else if(this->data_t == DATA_SOURCE) {
            printf("create tensor:%p DATA_SOURCE gpu_ptr:%p size: %zu byte\n", this, gpu_ptr, total_size_bytes);
        } else if(this->data_t == RNN_BUFF) {
			printf("create tensor:%p RNN_BUFF gpu_ptr:%p size: %zu byte\n", this, gpu_ptr, total_size_bytes);
		} else if(this->data_t == RNN_DATA) {
			printf("create tensor:%p RNN_DATA gpu_ptr:%p size: %zu byte\n", this, gpu_ptr, total_size_bytes);
		} else if(this->data_t == RNN_PARAM) {
			printf("create tensor:%p RNN_PARAM gpu_ptr:%p size: %zu byte\n", this, gpu_ptr, total_size_bytes);
		} else if(this->data_t == RNN_GRAD) {
			printf("create tensor:%p RNN_GRAD gpu_ptr:%p size: %zu byte\n", this, gpu_ptr, total_size_bytes);
		} else if(this->data_t == B_DATA) {
			printf("create tensor:%p B_DATA gpu_ptr:%p size: %zu byte\n", this, gpu_ptr, total_size_bytes);
		} 
		else {
            printf("unsupported type@%d tensor.h line 86\n", this->data_t);
            exit(1);
        }
#endif

    } 
    
    ~tensor_t() {
        if (cpu_ptr != NULL) {
            cudaFreeHost(cpu_ptr);
        }
        cpu_ptr = NULL;

#ifndef POOL_MALLOC_MODE
        if (this->data_t != DATA) {
            // checkCudaErrors( cudaFree(gpu_ptr) );
            // printf("delete *(****** gpu_ptr = %x\n", gpu_ptr);
        }
#endif

        // 显存池模式只需要重置显存指针
	#ifdef SWAP_MALLOC_MODE
	/*
        if (this->data_t != DATA && this->data_t != B_DATA && this->data_t != RNN_BUFF && this->data_t != CONV_BUFF && this->data_t != RNN_RESERVE)			
		{
            this->data_dp = NO_DATA;
            // acquireSpaceGPU(n*c*h*w);
			cudaFree(this->gpu_ptr);
			tensor_malloc_rank = tensor_malloc_count;
			tensor_malloc_count++;
        }
        else {
            this->data_ds = NO_COMPUTED;
            // this->data_ms = DATA_MEM_NO_READY;
        }
		*/
	#endif
		if (this->data_t != CONV_BUFF && this->data_t != RNN_BUFF) 
		{
			if (this->data_t == DATA_SOURCE_INT) {
				// printf("cpu_ptr_int = %x\n", this->cpu_ptr_int);
				// checkCudaErrors(cudaFreeHost(this->cpu_ptr_int));
				this->cpu_ptr_int = NULL;
			} 
			else {
			#ifndef SWAP_MALLOC_MODE
				freeSpaceCPU();
			#else
				if (this->data_t != DATA) {
					freeSpaceCPU();
				}
			#endif
			}
		}
	
        gpu_ptr = NULL;
        checkCUDNN(cudnnDestroyTensorDescriptor(cudnn_tensor_desc));
        cufftDestroy(fft_plan_f);
        cufftDestroy(fft_plan_b);
        checkCudaErrors(cudaEventDestroy(cpu2gpu_event));
        checkCudaErrors(cudaEventDestroy(gpu2cpu_event));
		tensor_counter--;
    }
    
	void set_data_layout(TENSOR_DATA_LAYOUT data_l) {
		this->data_layout = data_l;
	}
	
	TENSOR_DATA_LAYOUT get_data_layout() {
		return this->data_layout;
	}
	
	void acquireSpaceGPU(size_t total);
	void freeSpaceGPU(mem_mode target=CPU);
	void acquireSpaceCPU(size_t total);
    void freeSpaceCPU();
	
	inline int get_seq_time() { return this->seq_time; }
	inline int get_seq_batch() { return this->seq_batch; }
	inline int get_seq_beam() { return this->seq_beam; }
	inline int get_seq_vect() { return this->seq_vect; }

	void set_rnn_dim(size_t d1, size_t d2, size_t d3) {
		Dims[0] = d1; Dims[1] = d2, Dims[2] = d3;
					Strides[0] = Dims[1] * Dims[2];
			Strides[1] = Dims[2];
			Strides[2] = 1;
	}
	
    /*----utility functions----*/

	void set_tensor_id(int id) {
		this->tensor_id = id;
	}
	
	inline size_t get_tensor_base_id() {
		return tensor_base_id;
	}
	
	int get_malloc_rank() {
		return tensor_malloc_rank;
	}
	
	int get_tensor_id() {
		return this->tensor_id;
	}

    /**
     * NCHW, layer_id, data_type, data
     */
    void gen_description(char* buff, size_t* len_in_byte) {
        value_type _n = N, _c = C, _h = H, _w = W;
        value_type _layer_id = layer_id, _type = data_t;

        size_t SIZE = sizeof(value_type);
        memcpy(buff, &_n, SIZE);
        memcpy(buff+1*SIZE, &_c, SIZE); 
        memcpy(buff+2*SIZE, &_h, SIZE);
        memcpy(buff+3*SIZE, &_w, SIZE);
        memcpy(buff+4*SIZE, &_layer_id, SIZE);
        memcpy(buff+5*SIZE, &_type, SIZE);

        this->GPUtoCPU();

        memcpy(buff+6*SIZE, this->cpu_ptr, N*C*H*W*SIZE);

        *len_in_byte = (6+N*C*H*W)*SIZE;
    }

    //those are for mem_controller
	
	void stash_specified_gpu_space(value_type *specified_ptr, size_t offset) {
		size_t total = this->N * this->C * this->H * this->W;
		this->gpu_ptr = specified_ptr + (offset);
		// acquireSpecifiedSpaceGPU(total, specified_ptr, *offset);
		// *offset = *offset + total;
	}
	
	void set_require_space(bool x) {
		this->require_space = x;
	}
	
	bool if_require_space() {
		return require_space;
	}
	
    void stash_gpu_space() {
        long total = this->N * this->C * this->H * this->W;
        acquireSpaceGPU(total);
    }
    
    inline void free_gpu_space(mem_mode target=CPU) {
        freeSpaceGPU(target);
    }

	inline void set_backward_useful(bool x) {
		this->backward_useful = x;
	}
	
	inline bool get_backward_useful() {
		return this->backward_useful;
	}

	int get_use_counter(net_comp dir) {
		if (dir == FORWARD) {
			return this->forward_use_counter;
		}
		else if (dir == BACKWARD) {
			return this->backward_use_counter;
		}
	}

	int get_cur_use_counter(net_comp dir) {
		if (dir == FORWARD) {
			return this->forward_cur_use_counter;
		}
		else if (dir == BACKWARD) {
			return this->backward_cur_use_counter;
		}
	}

	inline void increase_use_counter(net_comp dir) {
		if (dir == FORWARD) {
			this->forward_use_counter++;
		}
		else if (dir == BACKWARD) {
			this->backward_use_counter++;
		}
	}
	inline void increase_cur_use_counter(net_comp dir) {
		if (dir == FORWARD) {
			this->forward_cur_use_counter++;
		}
		else if (dir == BACKWARD) {
			this->backward_cur_use_counter++;
		}
	}
	inline void reset_cur_use_counter(net_comp dir) {
		if (dir == FORWARD) {
			this->forward_cur_use_counter = 0;
		}
		else if (dir == BACKWARD) {
			this->backward_cur_use_counter = 0;
		}
	}
	inline void reset_use_counter(net_comp dir) {
		if (dir == FORWARD) {
			this->forward_use_counter = 0;
		}
		else if (dir == BACKWARD) {
			this->backward_use_counter = 0;
		}
	}
	inline bool is_tensor_useless(net_comp dir) {
		if (dir == FORWARD) {
			if (this->forward_cur_use_counter == forward_use_counter) {
				return true;
			}
			else {
				return false;
			}
			if (forward_cur_use_counter > forward_use_counter || forward_cur_use_counter <0) {
				printf("tensor%d from layer%d forward_cur_use_counter = %d, forward_cur_use_counter = %d\n",
						get_tensor_id(), get_layer_id(), forward_cur_use_counter, forward_use_counter);
			}
		}
		else if (dir == BACKWARD) {
			if (this->backward_cur_use_counter == backward_use_counter) {
				return true;
			}
			else {
				return false;
			}
			if (backward_cur_use_counter > backward_use_counter || backward_cur_use_counter <0) {
				printf("tensor%d from layer%d backward_cur_use_counter = %d, backward_cur_use_counter = %d\n",
						get_tensor_id(), get_layer_id(), backward_cur_use_counter, backward_use_counter);
			}
		}
	}
	
	bool is_use_in_backward() {
		return use_in_backward;
	}
	
	void set_use_in_backward(bool flag) {
		this->use_in_backward = flag;
	}
	
	inline void reset_data_state() {
		this->cur_use_count = this->use_count;
		this->data_ds = NO_COMPUTED;
	}

	inline void increase_use_count_initial() {
		this->use_count += 1;
	}

	inline int get_use_count() {
		return this->cur_use_count;
	}

	inline void increase_use_count() {
		this->cur_use_count += 1;
	}

	inline void decrease_use_count() {
		this->cur_use_count -= 1;
	}

	inline bool is_cur_use_count_zero() {
		return (this->cur_use_count == 0) ? true : false;
	}

	inline bool is_cur_equal_use_count() {
		return (this->cur_use_count == this->use_count) ? true : false ;
	}

    inline int print_use_count() {
        printf("tensor %d  cur_use_count=%d, use_count=%d\n", this->tensor_id, cur_use_count, use_count);
    }
	
	inline void print_cur_use_counter(net_comp dir) {
		if (dir == FORWARD) {
			printf("tensor%d FORWARD: %d/%d\n", this->get_tensor_id(), this->forward_cur_use_counter, this->forward_use_counter);
		}
		else {
			printf("tensor%d BACKWARD: %d/%d\n", this->get_tensor_id(), this->backward_cur_use_counter, this->backward_use_counter);
		}
	}
	
	inline void set_backward_recompute_pool_id(int id) {
		this->backward_recompute_pool_id = id;
	}
	
	inline int get_backward_recompute_pool_id() {
		return this->backward_recompute_pool_id;
	}
	
	inline void set_recompute_pool_id(int id) {
		this->recompute_pool_id = id;
	}
	
	inline int get_recompute_pool_id() {
		return this->recompute_pool_id;
	}
	
	inline void set_b_data_pool_id(int id) {
		this->b_data_pool_id = id;
	}
	
	inline int get_b_data_pool_id() {
		return this->b_data_pool_id;
	}
	
	inline void decrease_cur_b_use_count() {
		cur_b_use_count--;
	}
	
	inline void increase_b_use_count() {
		b_use_count++;
		cur_b_use_count++;
	}
	
	inline bool is_cur_b_use_count_zero() {
		return (this->cur_b_use_count == 0) ? true : false;
	}
	
	inline void reset_cur_b_use_count() {
		cur_b_use_count = b_use_count;
	}
	
    inline size_t get_N() { return this->N; }
    
    inline size_t get_C() { return this->C; }
    
    inline size_t get_H() { return this->H; }
    
    inline size_t get_W() { return this->W; }
    
    inline size_t get_scalar_count() {
        return this->get_N()*this->get_C()*this->get_H()*this->get_W();
    }
    
    inline size_t get_mem_size() {
        // const size_t total_size_bytes = sizeof(value_type)*this->N*this->C*this->H*this->W;
        if (require_space) {
			return total_size_bytes;
		}
		else {
			total_size_bytes = 0;
			return total_size_bytes;
		}
    }

    inline double get_bandwidth() {
        return bandwidth;
    }

	void set_swap_time(double tc, SWAP_DIR dir) {
		if (dir == OFFLOAD) this->gpu2cpu_time = tc;
		else this->cpu2gpu_time = tc;
	}
	
	double get_swap_time(SWAP_DIR dir) {
		if (dir == OFFLOAD) return this->gpu2cpu_time;
		else return this->cpu2gpu_time;
	}

    inline double get_comm_time() {
		if (require_space) {
			return comm_time;
		}
		else {
			comm_time = 0;
			return comm_time;
		}
    }
    
    void reshape(size_t n, size_t c, size_t h, size_t w) {
        assert(N*C*H*W == n*c*h*w);
        this->N = n;
        this->C = c;
        this->H = h;
        this->W = w;
		
		Dims[0] = n; Dims[1] = c; Dims[2] = h*w;
		Strides[0] = Dims[1] * Dims[2];
		Strides[1] = Dims[2];
		Strides[2] = 1;

		this->seq_time = n;
		this->seq_batch = c;
		this->seq_beam = h;
		this->seq_vect = w;
		
        checkCUDNN( cudnnSetTensor4dDescriptor(this->cudnn_tensor_desc,
                                               this->cudnn_tensor_format,
                                               this->cudnn_data_type,
                                               n, c, h, w) );
    }

    // add it to support data reader
    void replace_data(value_type *new_cpu_ptr, value_type* new_gpu_ptr=NULL);

    void replace_gpu_ptr_without_free(value_type* new_gpu_ptr) {
        // NOTE: this is a danger action!!!! now it just used in parallel_reader!!!!
        this->gpu_ptr = new_gpu_ptr;
        this->atomic_set_state(GPU_FUL);
    }
    
    inline int get_layer_id() {
        return this->layer_id;
    }

    inline TENSOR_TYPE get_type() {
        return this->data_t;
    }

	inline TENSOR_DATA_POSITION get_data_position() {
		return this->data_dp;
	}

    inline void set_data_position(TENSOR_DATA_POSITION p) {
		// if (this->get_tensor_id() == 1139 && p == IN_GPU) {
		// 	printf("LOOK! tensor1139 is set to IN_GPU%d\n", 666);
		// }
		// else if (this->get_tensor_id() == 1139 && p != IN_GPU) {
		// 	printf("LOOK! tensor1139 is set to %d\n", p);
		// }
        this->data_dp = p;
    }
	
	inline TENSOR_POSITION get_position() {
        return this->data_p;
    }

	inline void set_position(TENSOR_POSITION p) {
        this->data_p = p;
    }

    inline void set_first_last_pool_tensor(int fl) {
        if (fl == 0) {
            this->is_first_pool_tensor = 1;
        }
        else {
            this->is_last_pool_tensor = 1;
        }
    }

    inline TENSOR_MEM_STATE get_mem_state() {
        return this->data_ms;
    }

    inline void set_data_state(TENSOR_DATA_STATE ds) {
        this->data_ds = ds;
    }

    inline TENSOR_DATA_STATE get_data_state() {
        return this->data_ds;
    }

    inline bool check_first_pool_tensor() {
        return this->is_first_pool_tensor;
    }

    inline bool check_last_pool_tensor() {
        return this->is_last_pool_tensor;
    }

    inline void set_mem_state(TENSOR_MEM_STATE ms) {
        this->data_ms = ms;
    }        

	int get_swap_block_id() {
		return swap_block_id;
	}
	
	int get_prefetch_block_id() {
		return prefetch_block_id;
	}
	
	void set_swap_block_id(int id) {
		swap_block_id = id;
	}

	void set_prefetch_block_id(int id) {
		prefetch_block_id = id;
	}

    inline value_type* get_cpu_ptr() {
        return this->cpu_ptr;
    }
    
    inline value_type* get_gpu_ptr() {
        return this->gpu_ptr;
    }
	
	inline int* get_gpu_ptr_int() {
		return (this->gpu_ptr_int);
	}
	
	inline int* get_cpu_ptr_int() {
		return (this->cpu_ptr_int);
	}

    inline value_type* get_cpu_ptr_v2(size_t offset) {
		
        return ((this->cpu_ptr) + offset);
    }
    
    inline value_type* get_gpu_ptr_v2(size_t offset) {
		// return (value_type*)((void*)gpu_ptr + offset); 
        return ((this->gpu_ptr) + offset);
    }	

	void set_gpu_ptr_int(int* ptr) {
        this->gpu_ptr_int = ptr;    
    }

    void set_gpu_ptr(value_type* ptr) {
        this->gpu_ptr = ptr;    
    }

    void set_cpu_ptr(value_type* ptr) {
        this->cpu_ptr = ptr; 
    }
	
	void set_cpu_ptr_int(int* ptr) {
        this->cpu_ptr_int = ptr; 
    }
	
	inline value_type* get_temp_gpu_ptr_v2(size_t offset) {
		return (this->temp_gpu_ptr + offset);
	}
	inline value_type* get_temp_gpu_ptr() {
		return this->temp_gpu_ptr;
	}
	void set_temp_gpu_ptr(value_type* ptr) {
		this->temp_gpu_ptr = ptr;
	}
    
    inline cudnnTensorDescriptor_t get_tensor_desc() {
        return this->cudnn_tensor_desc;
    }
    
    inline cudnnTensorFormat_t get_tensor_format() {
        return this->cudnn_tensor_format;
    }
    
	void init_tensor_data(value_type data, bool to_gpu);
	void init_tensor_data2(value_type data, size_t num, bool to_gpu);
	
    void GPUtoCPU();
    
    void CPUtoGPU();
    
    void async_cpu_to_gpu();
    
    void async_gpu_to_cpu();
    
    inline bool is_cpu_to_gpu_ready();
    
    inline bool is_gpu_to_cpu_ready();
    
    void sync_cpu_to_gpu();
    
    void sync_gpu_to_cpu();
    
    void init(initializer_t<value_type> *initializer);
    
	void printRNNTensor(const char* str);
	
    void printTensorNoDebug(const char* str);
    
	void printTensorInt(const char* str);
	
    void printTensor(const char* str);
	
	void printTensorData(const char* str, int m);
    
    void printTensorFirst(const char* str);
	
	void printTensorState(const char* str);
    
    void writeToFile(const char* str);
    
    void hostRegister();
    
    void resizeTensor(size_t n, size_t c, size_t h, size_t w);
    
    void copy(tensor_t<value_type>* t,
              int src_start_idx=-1, int src_end_idx=-1, int dst_start_idx=-1, int dst_end_idx=-1);
    
    value_type get_scalar(const size_t n, const size_t c, const size_t h, const size_t w);
    
    void set_scalar( const size_t n, const size_t c, const size_t h, const size_t w, const value_type t );
    
    mem_mode get_state();

    /*---math functions-------*/
    void scale(value_type s);
    
    void sum(tensor_t<value_type>* t);
    
    value_type squared_sum(cublasHandle_t *handle);

    void forward_fft();
    
    void backward_fft();
	
	void reset_all_state();
	
	void reset_all_data_state();
	
	void reset_block_allocation();

};


    
} // ATP namespace

#endif // _TENSOR_H_
