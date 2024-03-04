#ifndef ATP_MEM_CONTROL_H
#define ATP_MEM_CONTROL_H

#include <thread>
#include <tensor.h>
#include <registry.h>
#include <util/common.h>
#include <layer/base_layer.h>
#include <layer/base_network_layer.h>
#include <layer/base_structure_layer.h>
#include <util/thread_routine.h>
#include <util/lru.h>
#include <liveness.h>
#include <layer/rnn_layer.h>
#include <layer/self_attn_layer.h>
// #include <recompute.h>
#include <stream_singleton.h>

typedef enum STREAM {
    COMPUTE_STREAM	= 0,
    GPU2CPU_STREAM	= 1,
    CPU2GPU_STREAM	= 2
}STREAM;

typedef enum POOL_STATE {
    POOL_READY		= 0,
    POOL_OCCUPIED	= 1
}POOL_STATE;

typedef enum SWAP_STATE {
	NO_STATE	= 0,
    READY		= 1,
    SWAPPING	= 2,
	DONE		= 3
}SWAP_STATE;

namespace ATP {

template<class value_type>
class mem_controller_t {
private:

	typedef struct SWAP_BLOCK {
		int id;
		void *block_ptr;
		size_t block_size;
		bool is_occup;
		// std::atomic<SWAP_STATE> swap_state;
		SWAP_STATE swap_state;
		tensor_t<value_type>* tensor;
		int tensor_id;
		// std::atomic<int> tensor_id;
	} SWAP_BLOCK;

	// multi swapping buffer
	std::vector<SWAP_BLOCK*> swap_blocks;
	// int *swap_ready_queue;
	// swap_blocks_list is used for control blocks' sequence when allocing
	std::list<SWAP_BLOCK*> swap_ready_list;
	std::list<int> recompute_ready_list;

    std::vector<LAYER> CHECKPOINT_LAYERS;

    registry_t<value_type>* reg;
	// network_t<value_type>* net;
    
    std::map<tensor_t<value_type>*, mem_mode> regulated_tensors;

    std::vector<std::vector<std::pair<int, net_comp> > > subsequent_forward;
    std::vector<std::vector<std::pair<int, net_comp> > > subsequent_backward;

	// 模拟训练共享存储单元
	value_type *ptr = NULL;
	int device;
	size_t offset;
	size_t free_space;
	value_type *ptr2 = NULL;
	int device2;
	size_t offset2;
	size_t free_space2;
	
	// grad共享空间
	value_type* grad_shared_block_ptr = NULL;
	size_t grad_shared_block_offset;
	size_t grad_shared_block_size_by_byte;
	size_t grad_shared_block_free_space;

	// 显存池,swap
	value_type *shared_block_ptr = NULL;
	size_t shared_block_offset;
	size_t shared_block_size_by_byte;
	size_t shared_block_free_space;
	value_type *shared_block_ptr2 = NULL;
	size_t shared_block_offset2;
	size_t shared_block_size_by_byte2;
	size_t shared_block_free_space2;
	
	// 显存池2
	void* pool_ptr = NULL;
	size_t pool_size = 0;
	size_t pool_offset = 0;
	size_t pool_free_size;
	
	// b_data显存池
	void* b_data_pool_ptr[B_DATA_POOL_NUM];
	bool b_data_pool_flag[B_DATA_POOL_NUM];
	tensor_t<value_type>* b_data_pool_tensors[B_DATA_POOL_NUM];
	
	// QKV显存池
	void* QKV_buffer[3];
	void* dQKV_buffer[3];
	
	// workspace pool
	void* workspace_pool = NULL;
	
	// recomputing memory pool
	// #define RECOMPUTE_POOL_NUM 3
	value_type* recomputing_pool[RECOMPUTE_POOL_NUM];
	bool recomputing_pool_flag[RECOMPUTE_POOL_NUM];
	tensor_t<value_type>* recompute_record[RECOMPUTE_POOL_NUM];
	
	POOL_STATE pool_state = POOL_READY;
    
	void* cpu_pool;
	size_t cpu_pool_offset;
	size_t cpu_pool_free_size;
	size_t cpu_pool_size;
	
	void alloc_mem_by_cpu_pool(void* cpu_ptr, size_t mem_size);
	
	void init_cpu_pool(size_t mem_size);
	
    void print_required_tensor(int layer_id, net_comp dir);
    
    void print_layer_type(int layer_id, net_comp dir);

    int max_layer_id=0;

    liveness_analysis_t<value_type>* live_anls = NULL;
    // recompute_t<value_type>*         recomp;

	std::vector<base_layer_t<value_type>*> swapped_layers;
	std::vector<tensor_t<value_type>*>  swap_tensors;
	std::vector<tensor_t<value_type>*>  swapped_tensors;
	  
	void malloc_gpu_mem_pool(size_t pool_size);
	
	void alloc_mem_by_gpu_mem_pool(void** gpu_ptr, size_t size);
	
	void reset_gpu_mem_pool_offset();
	
	inline std::vector<base_layer_t<value_type>*> get_swapped_layers() {
		return swapped_layers;
	}

	inline void add_swap_tensors(tensor_t<value_type>* t) {
		swap_tensors.push_back(t);
	}

	inline tensor_t<value_type>* get_last_swap_tensor() {
		return swap_tensors.back();
	}

	inline void delete_last_swap_tensor() {	
		swap_tensors.pop_back();
	}

	inline void add_swapped_layer(base_layer_t<value_type>* layer) {
		swapped_layers.push_back(layer);
		// printf("add layer%d into swap_layers\n", (swapped_layers.back())->get_base_id());
	}

	inline base_layer_t<value_type>* get_last_swapped_layer() {
		// printf("get layer%d from swap_layers\n", (swapped_layers.back())->get_base_id());
		return swapped_layers.back();
	}

	inline void delete_last_swapped_layer() {
		// printf("delete layer%d from swap_layers\n", (swapped_layers.back())->get_base_id());
		swapped_layers.pop_back();
	}

	inline void add_swapped_tensors(tensor_t<value_type>* t) {
		swapped_tensors.push_back(t);
		// printf("add tensor%x from swapped_tensors\n", swapped_tensors.back());
	}

	inline tensor_t<value_type>* get_last_swapped_tensor() {
		// printf("get tensor%x from swapped_tensors\n", swapped_tensors.back());
		return swapped_tensors.back();
	}

	inline void delete_last_swapped_tensor() {	
		// printf("delete tensor%x from swapped_tensors\n", swapped_tensors.back());
		swapped_tensors.pop_back();
	}

    void set_regulated_tensors();

	

public:
    mem_controller_t() {
		// 可以作为CHECKPOINT_LAYERS的层
        CHECKPOINT_LAYERS.push_back(CONV);
        CHECKPOINT_LAYERS.push_back(FC);
        CHECKPOINT_LAYERS.push_back(JOIN_L);
        CHECKPOINT_LAYERS.push_back(FORK_L);
        CHECKPOINT_LAYERS.push_back(CONCAT);
    }
    
    ~mem_controller_t() {
//        print_regulated_tensors(true);
	#ifdef RECOMPUTE_POOL_NUM
		for (int i = 0; i < RECOMPUTE_POOL_NUM; i++) {
			recomputing_pool[i] = NULL;
		}
	#endif
	#ifdef B_DATA_POOL_NUM
		for (int i = 0; i < B_DATA_POOL_NUM; i++) {
			b_data_pool_ptr[i] = NULL;
		}
	#endif
		if (workspace_pool != NULL) {
			free(workspace_pool);
			workspace_pool = NULL;
		}
		if (pool_ptr != NULL) {
			free(pool_ptr);
			pool_ptr = NULL;
		}
		if (shared_block_ptr != NULL) {
			free(shared_block_ptr);
			shared_block_ptr = NULL;
		}
		if (shared_block_ptr2 != NULL) {
			free(shared_block_ptr2);
			shared_block_ptr2 = NULL;
		}
		if (grad_shared_block_ptr != NULL) {
            free(shared_block_ptr2);
			shared_block_ptr2 = NULL;
        }
    }
	
	void reset_all_gpu_mem_pool(); 
    /*
	void set_pool_tensors(int *layers_id, int count) {
		int *_layers_id = layers_id;
		auto net_layers = reg->get_net_layers();
		base_network_layer_t<value_type>* net_layer;
		tensor_t<value_type>* t;
		for (size_t i = 0; i < count; i++) {
			net_layer = (base_network_layer_t<value_type>*)net_layers[*(_layers_id+i)];
			t = net_layer->get_f_out();
			t->set_data_position(SHARED_GPU_POOL);
			if (i == 0) {
				t->set_first_last_pool_tensor(0);
			}
			else if (i == (count-1)) {
				t->set_first_last_pool_tensor(1);
			}
		}
	}
	*/
	void clear_related_record();
	
	void print_pool_size() {
		printf("pool_size = %zd\n", this->pool_size);
	}
	
    liveness_analysis_t<value_type>* get_liveness_analysis_t() {
        return this->live_anls;
    }

    // recompute_t<value_type>* get_recompute_t() {
       // return this->recomp;
    // }

    void init(registry_t<value_type> *r);

	void* get_OKV_buffer(net_comp dir, int j) {
		if (dir = FORWARD) {
			return this->QKV_buffer[j];
		}
		else {
			return this->dQKV_buffer[j];
		}
	}
	
	void set_OKV_buffer(net_comp dir, int j, void* ptr) {
		if (dir = FORWARD) {
			this->QKV_buffer[j] = ptr;
		}
		else {
			this->dQKV_buffer[j] = ptr;
		}
	}

	void get_pool(void** pool_ptr, size_t* pool_offset) {
		*pool_ptr = this->pool_ptr;
		*pool_offset = this->pool_offset;
	}

	void set_pool(size_t offset) {
		pool_offset = offset;
		pool_free_size = pool_size - offset;
	}

	void init_all_tensors(
		size_t gpu_pool_size, size_t swap_pool_size, size_t recompute_pool_size,
		size_t max_grad_size, size_t max_data_size, size_t b_data_pool_size, size_t QKV_buffer_size,
		size_t max_workspace_pool);

	void init_simulator_memory();

	void free_tensor_shared_memory();

	void init_swap_memory_pool(size_t pool_size = 0);
	
	void init_grad_memory_pool(size_t max_grad_size);
	
	void init_all_tensor_gpu_memory();

	void free_gpu_mem_pool();

    /***************************************/
    // for profile
    std::pair<double, double> stash_tensors_for_profile(int curt_layer_id, net_comp dir);
    void free_tensors_for_profile(int curt_layer_id, net_comp dir);
    /***************************************/

    void print_regulated_tensors(bool log=false, int layer_id=-1);
    
    void reset_tensor_state();
    
    int stash_tensor(int layer_id, base_layer_t<value_type>* layer, net_comp dir, network_stage stage, std::stack<int>* recompute_layers_stack);

	void stash_tensor_shared_memory(int layer_id, net_comp dir, network_stage stage);

    void stash_tensor_malloc_all(int layer_id, net_comp dir, network_stage stage);
    
    void update_tensor_state(int layer_id, base_layer_t<value_type>* layer, net_comp dir, network_stage stage);

	void update_tensor_state_shared_memory(int layer_id, net_comp dir, network_stage stage);

    void update_tensor_state_free_all(int layer_id, net_comp dir, network_stage stage);
	
	void synchronize(STREAM stream) {
		switch(stream) {
			case COMPUTE_STREAM: cudaStreamSynchronize(stream_singleton::get_compute_stream());
			case GPU2CPU_STREAM: cudaStreamSynchronize(stream_singleton::get_gpu2cpu_stream());
			case CPU2GPU_STREAM: cudaStreamSynchronize(stream_singleton::get_cpu2gpu_stream());
		}
	}
	
	bool find_free_recompute_block(net_comp dir, int* recompute_pool_id, tensor_t<value_type>* tensor, bool has_block_id);
	
	bool find_free_swap_block(net_comp dir, int* swap_block_id, tensor_t<value_type>* tensor, bool has_block_id);
	
	bool PreAllocateRecompteSwapBlock(
    std::vector<tensor_t<value_type>* > *swap_tensors, 
	std::vector<tensor_t<value_type>* > *prefetch_tensors);
	
	bool preallocate_swap_recompute_block(
		std::vector<tensor_t<value_type>* > *recompute_tensors, 
		std::vector<tensor_t<value_type>* > *swap_tensors, 
		std::vector<tensor_t<value_type>* > *prefetch_tensors);
	
	bool preallocate_swap_block(
		std::vector<tensor_t<value_type>* > *swap_tensors, 
		std::vector<tensor_t<value_type>* > *prefetch_tensors); 
	
	void swap_ctrl(net_comp dir, MODEL_TYPE model_type, tensor_t<value_type>* tensor, double* pur_swap_time);
	
	SWAP_BLOCK* get_swap_block(int id) {
		return swap_blocks[id];
	}
	
	tensor_t<value_type>* get_swap_block_tensor(int id) {
		return swap_blocks[id]->tensor;
	}
	
	tensor_t<value_type>* get_recompute_block_tensor(int id) {
		return recompute_record[id];
	}	
	
	bool is_swap_block_occupied(int id) {
		return swap_blocks[id]->is_occup;
	}
	
	SWAP_STATE get_swap_block_state(int id) {
		return swap_blocks[id]->swap_state;
	}
	
	void set_recompute_block(int id, tensor_t<value_type>* tensor) {
		if (recompute_record[id] != NULL) {
            recompute_record[id]->set_data_position(DELETED);
		}
		recomputing_pool_flag[id] = true;
        recompute_record[id] = tensor;
        tensor->set_data_position(IN_GPU);
	}
	
	void set_swap_block(int id, tensor_t<value_type>* tensor);
	
	void set_swap_block_state(int id, SWAP_STATE swap_state);
	
	void reset_recompute_pool();
	
	void reset_swap_block();
	
	void init_swap_block();
	
	bool alloc_mem_by_swap_block(tensor_t<value_type>* tensor, net_comp dir);
	
	void printSWAPBLOCK(const char* str);

	void printRecomputePool(const char* str);
	
	void printMemPool(const char* str);
};
    
} // namespace ATP

#endif //ATP_INITIALIZER_H
