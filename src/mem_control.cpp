#include <mem_control.h>
#include <util/mem_util.h>
//#define MEM_DEBUG

namespace ATP{


template <class value_type>
void mem_controller_t<value_type>::init(registry_t<value_type> *r) {
    this->reg = r;

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
		// if (workspace_pool != NULL) {
		// 	// free(workspace_pool);
		// 	workspace_pool = NULL;
		// }
		// if (pool_ptr != NULL) {
		// 	// free(pool_ptr);
		// 	pool_ptr = NULL;
		// }
		// if (shared_block_ptr != NULL) {
		// 	// free(shared_block_ptr);
		// 	shared_block_ptr = NULL;
		// }
		// if (shared_block_ptr2 != NULL) {
		// 	// free(shared_block_ptr2);
		// 	shared_block_ptr2 = NULL;
		// }
		// if (grad_shared_block_ptr != NULL) {
        //     // free(shared_block_ptr2);
		// 	shared_block_ptr2 = NULL;
        // }

    max_layer_id = -1;
	// 
    for (auto it = r->get_net_layers().begin(); it!=r->get_net_layers().end(); ++it) {
        if (it->first > max_layer_id) {
            max_layer_id = it->first;
        }
    }


// #ifdef RECOMPUTE_ON
// 	// 选择需要重计算的层
//     recomp = new recompute_t<value_type>(reg, (std::map<void *, mem_mode>*)&regulated_tensors, CHECKPOINT_LAYERS, max_layer_id);
// #endif
}

template <class value_type>
void mem_controller_t<value_type>::malloc_gpu_mem_pool(size_t pool_size) {
    printf("cudamalloc pool for net, size =  %zd\n", pool_size);
    cudaMalloc(&(this->pool_ptr), pool_size);
    
    this->pool_size = pool_size;
    this->pool_offset = 0;
    this->pool_free_size = this->pool_size;
}

template <class value_type>
void mem_controller_t<value_type>::free_gpu_mem_pool() {
    printf("this->pool_ptr = %zd\n", this->pool_ptr);
    if (this->pool_ptr != NULL) {
        printf("free gpu_pool\n");
        cudaFree(this->pool_ptr);
    }
    this->pool_offset = 0;
    this->pool_size = 0;
    this->pool_free_size = this->pool_size;
}

template <class value_type>
void mem_controller_t<value_type>::reset_all_gpu_mem_pool() {
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

template <class value_type>
void mem_controller_t<value_type>::alloc_mem_by_gpu_mem_pool(void** gpu_ptr, size_t size) {
    if (this->pool_free_size < size) {
        printf("Require %zdbytes but free pool size is only %zdbytes\n", size, (this->pool_size - pool_offset));
        return;
    }
    *gpu_ptr = this->pool_ptr + this->pool_offset;
    this->pool_offset += size;
    // this->pool_free_size = this->pool_size - this->pool_offset;
    this->pool_free_size -= size;
}

template <class value_type>
void mem_controller_t<value_type>::reset_gpu_mem_pool_offset() {
    this->pool_offset = 0;
    this->pool_free_size = this->pool_size;
}

template <class value_type>
void mem_controller_t<value_type>::clear_related_record() {
	regulated_tensors.erase(regulated_tensors.begin(), regulated_tensors.end());
	for (auto it = subsequent_forward.begin(); it != subsequent_forward.end(); ++it) {
		it->erase(it->begin(), it->end());
	}
	subsequent_forward.erase(subsequent_forward.begin(), subsequent_forward.end());
	for (auto it = subsequent_backward.begin(); it != subsequent_backward.end(); ++it) {
		it->erase(it->begin(), it->end());
	}
	subsequent_backward.erase(subsequent_backward.begin(), subsequent_backward.end());
	
} 

/*--------network profile---------*/
template <class value_type>
std::pair<double, double> mem_controller_t<value_type>::stash_tensors_for_profile(int curt_layer_id, net_comp dir) {

    std::vector<tensor_t<value_type>* >* tensors = NULL;
    if(dir == FORWARD) {
        tensors = reg->get_forward_dependency(curt_layer_id); // TO DO, we don't track the data layer!!
    } else if(dir == BACKWARD) {
        tensors = reg->get_backward_dependency(curt_layer_id);
    }
    if(tensors == NULL) return std::make_pair(0, 0);

    size_t total_mem = 0;
    double total_time  = 0;
    for(size_t i = 0; i < tensors->size(); i++) {
        tensor_t<value_type>* t = tensors->operator[](i);
        if( t->get_type() != DATA && t->get_type() != CONV_BUFF ) continue;
        if( t->get_type() == DATA) {
            total_mem += t->get_mem_size();
        }
//        t->atomic_set_state(GPU);
        t->stash_gpu_space();
        for(size_t j = 0; j < 10; j++) {
            double start = get_cur_time();
            t->CPUtoGPU();
            double end   = get_cur_time();
            total_time += (end - start);
        }
    }
    double avg_time = total_time / 10.0f;

    double mem_in_mb = total_mem/1000000.0f;

#ifdef MEM_DEBUG
    double curt_mem = query_gpu_mem();
    if (dir == FORWARD) {
        printf("------------forward:%d(%f MB) stash------------\n", curt_layer_id, curt_mem);
    } else {
        printf("------------backward:%d(%f MB) stash-----------\n", curt_layer_id, curt_mem);
    }
    this->print_regulated_tensors();
#endif
    return std::make_pair(avg_time, mem_in_mb);
}

template <class value_type>
void mem_controller_t<value_type>::free_tensors_for_profile(int curt_layer_id, net_comp dir) {
    std::vector<tensor_t<value_type>* >* tensors = NULL;
    if(dir == FORWARD) {
        tensors = reg->get_forward_dependency(curt_layer_id); // TO DO, we don't track the data layer!!
    } else if(dir == BACKWARD) {
        tensors = reg->get_backward_dependency(curt_layer_id);
    }
    if(tensors == NULL) return;

    size_t total_mem = 0;
    double total_time  = 0;
    for(size_t i = 0; i < tensors->size(); i++) {
        tensor_t<value_type>* t = tensors->operator[](i);
        if( t->get_type() != DATA && t->get_type() != CONV_BUFF ) continue;
        if( t->get_type() == DATA) {
            total_mem += t->get_mem_size();
        }
//        t->atomic_set_state(VOID);
        t->free_gpu_space();
    }

    double avg_time = total_time / 10.0f;
#ifdef MEM_DEBUG
    double curt_mem = query_gpu_mem();
    if (dir == FORWARD) {
        printf("------------forward:%d(%f MB) update------------\n", curt_layer_id, curt_mem);
    } else {
        printf("------------backward:%d(%f MB) update-----------\n", curt_layer_id, curt_mem);
    }
    this->print_regulated_tensors();
#endif
}
/*-----------------------------------*/


template <class value_type>
void mem_controller_t<value_type>::reset_tensor_state() {
#ifdef LIVENESS
    for (auto it = regulated_tensors.begin(); it != regulated_tensors.end(); it++ ) {
        it->first->free_gpu_space(VOID);
    }
#endif
}


template <class value_type>
void mem_controller_t<value_type>::print_layer_type(int layer_id, net_comp dir) {
    std::map<int, void* > net_layers     = reg->get_net_layers();
    base_layer_t<value_type>* curt_layer = (base_layer_t<value_type>*) net_layers.find(layer_id)->second;
    LAYER curt_layer_type                = curt_layer->get_layer_type();
    /*
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
    CONCAT  = 10,
    FORK_L  = 11,
    JOIN_L  = 12
     */

    if( curt_layer_type == CONV ) {
        printf("@layer %d, CONV\n", layer_id);
    } else if( curt_layer_type == POOL ) {
        printf("@layer %d, POOL\n", layer_id);
    } else if( curt_layer_type == ACT ) {
        printf("@layer %d, ACT\n", layer_id);
    } else if( curt_layer_type == BN ) {
        printf("@layer %d, BN\n", layer_id);
    } else if( curt_layer_type == FC ) {
        printf("@layer %d, FC\n", layer_id);
    } else if( curt_layer_type == LRN ) {
        printf("@layer %d, LRN\n", layer_id);
    } else if( curt_layer_type == PADDING ) {
        printf("@layer %d, PADDING\n", layer_id);
    } else if( curt_layer_type == DATA_L ) {
        printf("@layer %d, DATA_L\n", layer_id);
    } else if( curt_layer_type == DROPOUT ) {
        printf("@layer %d, DROPOUT\n", layer_id);
    } else if( curt_layer_type == CONCAT ) {
        printf("@layer %d, CONCAT\n", layer_id);
    } else if( curt_layer_type == FORK_L ) {
        printf("@layer %d, FORK_L\n", layer_id);
    } else if( curt_layer_type == JOIN_L ) {
        printf("@layer %d, JOIN_L\n", layer_id);
    } else if( curt_layer_type == SOFTMAX ) {
        printf("@layer %d, SOFTMAX\n", layer_id);
    } else {
        printf("@layer %d, UNKNOWN.....\n", layer_id);
    }
}
/*--------Interlayer shared memory----------*/

template <class value_type>
void mem_controller_t<value_type>::init_simulator_memory() {
    size_t net_total_size;
	size_t gpu_tensors_size;
	size_t gpu_pool_size;
	size_t swap_tensors_size;
	size_t swap_pool_size;
    size_t recompute_tensor_size;
	size_t recompute_pool_size;
    size_t max_grad_size;
	size_t b_data_pool_size;
    size_t QKV_buffer_size;
	size_t max_data_size;
	size_t max_tensor_size;
	size_t max_workspace_size;  
    size_t max_fragment_size;
    size_t max_layer_size;
    // size_t bottleneck_mem_size = live_anls->get_bottleneck_size();
    printf("after free_gpu_mem_pool the memory used:%f\n", BYTE_TO_MB(query_used_mem()));
    reg->get_size_info(
			&net_total_size, &gpu_tensors_size, &gpu_pool_size, 
			&swap_tensors_size, &swap_pool_size, &recompute_tensor_size, &recompute_pool_size, &max_fragment_size,
			&max_grad_size, &max_tensor_size, &max_data_size, &max_layer_size,
			&b_data_pool_size, &QKV_buffer_size, &max_workspace_size);
	// size_t bottleneck_data_size = live_anls->get_bottleneck_data_size();
    // printf("bottleneck_data_size = %f\n", BYTE_TO_MB(bottleneck_data_size));
    // size_t max_grad_size = live_anls->get_max_grad_size();

    printf("after free_gpu_mem_pool the memory used:%f\n", BYTE_TO_MB(query_used_mem()));
    this->malloc_gpu_mem_pool(max_layer_size);
    // cudaMalloc((void**)&this->ptr, max_layer_size);
	// cudaMalloc((void**)&this->ptr2, bottleneck_data_size);
    // cudaMalloc((void**)&this->ptr, bottleneck_mem_size);
	// cudaMalloc((void**)&this->ptr2, bottleneck_mem_size);
	value_type* cpu_block_ptr = (value_type*)malloc(max_layer_size);
	// value_type* cpu_block_ptr2 = (value_type*)malloc(max_layer_size);
    // value_type* cpu_block_ptr = (value_type*)malloc(bottleneck_mem_size);
	// value_type* cpu_block_ptr2 = (value_type*)malloc(bottleneck_mem_size);
	size_t total = max_layer_size / sizeof(value_type);
    // size_t total = bottleneck_mem_size / sizeof(value_type);
	for ( size_t i = 0; i < total; i++ ) {
		*(cpu_block_ptr+i) = 3.14;
		// *(cpu_block_ptr2+i) = 6.28;
	}
	cudaMemcpy((void*)this->pool_ptr, (void*)cpu_block_ptr, max_layer_size, cudaMemcpyHostToDevice);

	free(cpu_block_ptr);
	// free(cpu_block_ptr2);
	offset = 0;
	free_space = max_layer_size;
    // free_space = bottleneck_mem_size;
	offset2 = 0;
	free_space2 = max_layer_size;
    // free_space2 = bottleneck_mem_size;
	device2= 0;
}

template <class value_type>
void mem_controller_t<value_type>::printMemPool(const char* str) {
    printf("%s : pool_size=%zd, offset=%zd, free_size=%zd\n", str, pool_size, pool_offset, pool_free_size);
}

// #ifdef MULTI_SWAP_BLOCK

template <class value_type>
void mem_controller_t<value_type>::printSWAPBLOCK(const char* str) {
    printf("********%s********\n", str);
    for (int i = 0; i < swap_blocks.size(); i++) { 
        printf("swap_blocks[%d]: is_occup=%d size=%zd ", i, swap_blocks[i]->is_occup, swap_blocks[i]->block_size);
        switch(swap_blocks[i]->swap_state) {
            case NO_STATE: printf(" NO_STATE "); break;
            case READY: printf(" READY "); break;
            case SWAPPING: printf(" SWAPPING "); break;
            case DONE: printf(" DONE "); break;
            default: printf(" **** ");
        }
        printf(",\n");
        tensor_t<value_type>* t = swap_blocks[i]->tensor;
        if (t == NULL) {
            printf("no tensor here\n");
        }
        else {
            t->printTensorState(" ");
        }
    } 
    printf("swap_ready_list: ");
    for (auto it = swap_ready_list.begin(); it != swap_ready_list.end(); it++) {
        printf("[%d] ", (*it)->id);
    }
    printf("\n************************************\n\n");
}

// template <class value_type>
// void mem_controller_t<value_type>::reset_recompute_block( ) {
//     recompute_ready_list.clear();
//     for (int i = 0; i < recompute_blocks.size(); i++) {
//         recompute_blocks[i]->tensor_id = -1;
//         recompute_blocks[i]->tensor = NULL;
//         recompute_blocks[i]->is_occup = false;
//         recompute_ready_list.push_back(recompute_blocks[i]);
//     }   
// }

template <class value_type>
void mem_controller_t<value_type>::set_swap_block(int id, tensor_t<value_type>* tensor) {
		swap_blocks[id]->tensor = tensor;
		swap_blocks[id]->is_occup = true;
		swap_blocks[id]->tensor_id = tensor->get_tensor_id();
		swap_blocks[id]->swap_state = READY;
	}

template <class value_type>
void mem_controller_t<value_type>::set_swap_block_state(int id, SWAP_STATE swap_state) {
	swap_blocks[id]->swap_state = swap_state;
}

template <class value_type>
void mem_controller_t<value_type>::reset_swap_block( ) {
    
    swap_ready_list.clear();
    for (int i = 0; i < swap_blocks.size(); i++) {
        swap_blocks[i]->tensor_id = -1;
        swap_blocks[i]->tensor = NULL;
        swap_blocks[i]->is_occup = false;
        swap_blocks[i]->swap_state = NO_STATE;
        swap_ready_list.push_back(swap_blocks[i]);
    }   
}

template <class value_type>
void mem_controller_t<value_type>::init_swap_block() {
    SWAP_BLOCK *sb[SWAP_BLOCK_NUM];
    swap_blocks.clear();
	for (int i = 0; i < SWAP_BLOCK_NUM; i++) {
        sb[i] = new SWAP_BLOCK();
		swap_blocks.push_back(sb[i]);
        swap_blocks[i]->id = i;
        swap_blocks[i]->is_occup = false;
        swap_blocks[i]->swap_state = NO_STATE;
	}
    swap_ready_list.clear();
    for (int i = 0; i < swap_blocks.size(); i++) {
        swap_blocks[i]->tensor_id = -1;
        swap_blocks[i]->tensor = NULL;
        swap_blocks[i]->is_occup = false;
        swap_blocks[i]->swap_state = NO_STATE;
        swap_ready_list.push_back(swap_blocks[i]);
    }
}

template <class value_type>
bool mem_controller_t<value_type>::find_free_swap_block(net_comp dir, int* swap_block_id, tensor_t<value_type>* tensor, bool has_block_id) {
    if (dir == FORWARD) {
        if (tensor->get_swap_block_id() != -1) {  // has been allocated
            *swap_block_id = -1;  // return *swap_block_id = -1 to indicate this tensor needn't to be allocated again
            return true;
        }
        for (auto it = swap_ready_list.begin(); it != swap_ready_list.end(); it++) {
            if ((*it)->tensor == NULL) {
                *swap_block_id = (*it)->id;
                swap_ready_list.remove(*it);
                swap_ready_list.push_back(swap_blocks[*swap_block_id]);
                return true;
            }
            else if ((*it)->tensor->get_data_state() == FORWARD_DELETE_OK) {
                *swap_block_id = (*it)->id;
                swap_ready_list.remove(*it);
                swap_ready_list.push_back(swap_blocks[*swap_block_id]);
                return true;
            }
        }
        return false;
    }
    else {  // dir == BACKWARD
        // if tensor 
        // printf("tensor%d->get_get_prefetch_block_id=%d data_position=%d\n", tensor->get_tensor_id(), tensor->get_swap_block_id(), tensor->get_data_position());
        if (tensor->get_prefetch_block_id() != -1) {  // tensor has been allocated
            *swap_block_id = -1;
            return true;
        }
        if (tensor->get_data_position() == IN_GPU || tensor->get_data_position() == IN_CPU_GPU) {  // tensor is still in GPU, it will happen in the last swapping tensor
            // printf("tensor%d->get_data_position() = %d, swap_block_id = %d\n", tensor->get_tensor_id(), tensor->get_data_position(), tensor->get_swap_block_id());
            *swap_block_id = tensor->get_swap_block_id();
            swap_ready_list.remove(swap_blocks[*swap_block_id]);
            swap_ready_list.push_back(swap_blocks[*swap_block_id]);
            // printf("tensor->get_data_position() = %d, swap_block_id = %d\n", tensor->get_data_position(), tensor->get_swap_block_id());
            return true;
        }
        for (auto it = swap_ready_list.begin(); it != swap_ready_list.end(); it++) {
            if ((*it)->tensor == NULL) {
                *swap_block_id = (*it)->id;
                swap_ready_list.remove(*it);
                swap_ready_list.push_back(swap_blocks[*swap_block_id]);
                return true;
            }
            else if ((*it)->tensor->get_data_state() == NO_COMPUTED) {
                *swap_block_id = (*it)->id;
                swap_ready_list.remove(*it);
                swap_ready_list.push_back(swap_blocks[*swap_block_id]);
                return true;
            }
        }
        return false;
    }
}

template <class value_type>
void mem_controller_t<value_type>::reset_recompute_pool() {
    recompute_ready_list.clear();
    for (int i = 0; i < RECOMPUTE_POOL_NUM; i++) {
        recomputing_pool_flag[i] = false;
        recompute_record[i] = NULL;
        recompute_ready_list.push_back(i);
    }   
}

template <class value_type>
void mem_controller_t<value_type>::printRecomputePool(const char* str) {
    printf("********%s********\n", str);
    for (int i = 0; i < RECOMPUTE_POOL_NUM; i++) { 
        printf("recompute_pool[%d]: is_occup=%s\n", i, (recomputing_pool_flag[i]? "true":"false"));
        tensor_t<value_type>* t = recompute_record[i];
        if (t == NULL) {
            printf("no tensor here\n");
        }
        else {
            t->printTensorState(" ");
        }
    } 
    printf("recompute_ready_list: ");
    for (auto it = recompute_ready_list.begin(); it != recompute_ready_list.end(); it++) {
        printf("[%d] ", (*it));
    }
    printf("\n************************************\n\n");
}

template <class value_type>
bool mem_controller_t<value_type>::find_free_recompute_block(net_comp dir, int* recompute_pool_id, tensor_t<value_type>* tensor, bool has_block_id) {
    if (dir == FORWARD) {
        // printf("get_recompute_pool_id = %d\n", tensor->get_recompute_pool_id());
        if (tensor->get_recompute_pool_id() != -1) {  // has been allocated
            return true;
        }
        for (auto it = recompute_ready_list.begin(); it != recompute_ready_list.end(); it++) {
            // printf("*it = %d\n", *it);
            if (recomputing_pool_flag[*it] == false) {
                recompute_ready_list.remove(*it);
                recompute_ready_list.push_back(*it);
                *recompute_pool_id = *it;
                // recompute_record[i] = tensor;
                recomputing_pool_flag[*it] = true;
                return true;
            }
            else {
                if (recompute_record[*it]->get_data_state() == FORWARD_DELETE_OK) {
                    *recompute_pool_id = *it;
                    recompute_ready_list.remove(*it);
                    recompute_ready_list.push_back(*it);
                    // printf("forward layer%d tensor%d->get_data_position() = %d, recompute_block_id = %d\n", tensor->get_layer_id(), tensor->get_tensor_id(), tensor->get_data_position(), *recompute_pool_id);
                    // recompute_record[i] = tensor;
                    recomputing_pool_flag[*it] = true;
                    return true;
                }
            }
        }
        return false;
    }
    else {
        if (tensor->get_backward_recompute_pool_id() != -1) {  // has been allocated
            return true;
        }
        if (tensor->get_data_position() == IN_GPU) {
            *recompute_pool_id = tensor->get_recompute_pool_id();
            recompute_ready_list.remove(*recompute_pool_id);
            recompute_ready_list.push_back(*recompute_pool_id);
            // printf("tensor%d->get_data_position() = %d, recompute_block_id = %d\n", tensor->get_tensor_id(), tensor->get_data_position(), *recompute_pool_id);
            return true;
        }
        for (auto it = recompute_ready_list.begin(); it != recompute_ready_list.end(); it++) {
            if (recomputing_pool_flag[*it] == false) {
                *recompute_pool_id = *it;
                recompute_ready_list.remove(*it);
                recompute_ready_list.push_back(*it);
                recomputing_pool_flag[*it] = true;
                // recompute_record[i] = tensor;
                return true;
            }
            else {
                if (recompute_record[*it]->get_data_state() == NO_COMPUTED) {
                    *recompute_pool_id = *it;
                    recompute_ready_list.remove(*it);
                    recompute_ready_list.push_back(*it);
                    recomputing_pool_flag[*it] = true;
                    // recompute_record[i] = tensor;
                    return true;
                }
            }
        }
    }
}

// #define DEBUG
template <class value_type>
bool mem_controller_t<value_type>::PreAllocateRecompteSwapBlock(
    std::vector<tensor_t<value_type>* > *swap_tensors, std::vector<tensor_t<value_type>* > *prefetch_tensors) {
    std::vector<std::pair<int, net_comp> >* net_comp_route = reg->get_net_comp_route_ptr();
    std::map<int, void* >* net_layers  = reg->get_net_layers_ptr();
    int block_id = -1;
    int layer_id;
    for ( int k = 0; k < net_comp_route->size(); k++ ) {
        if ((*net_comp_route)[k].second == FORWARD) {
			layer_id = (*net_comp_route)[k].first;
			base_layer_t<value_type>* layer = (base_layer_t<value_type>*)(net_layers->find(layer_id)->second);
			if (layer->get_layer_structure() == SISO) {
                if (layer->get_layer_type() == RNN  || layer->get_layer_type() == SATTN) {
					// tensor_t<value_type>* reserve_buff = ((rnn_layer_t<value_type>*)layer)->get_rnn_reserve_buff();
                    tensor_t<value_type>* reserve_buff = ((base_network_layer_t<value_type>*)layer)->get_reserve_buff();
					reserve_buff->reset_all_data_state();
                    reserve_buff->reset_block_allocation();
				}
				tensor_t<value_type>* t_out = ((base_network_layer_t<value_type>*)layer)->get_f_out();
				t_out->reset_all_data_state();
                t_out->reset_block_allocation();
			}
			else {
				std::vector<tensor_t<value_type>*> t_outs = ((base_structure_t<value_type>*)layer)->get_outputs();
				for (int j = 0; j < t_outs.size(); j++) {
					t_outs[j]->reset_all_data_state();
                    t_outs[j]->reset_block_allocation();
				}
			}
		}
    }
    reset_swap_block();
    reset_recompute_pool();
    #ifdef DEBUG
        printf("PreAllocateRecompteSwapBlock start%d\n", 2);
    #endif
    for (size_t i = 0; i < net_comp_route->size(); i++) {
        if ((*net_comp_route)[i].second == FORWARD) {
            layer_id = (*net_comp_route)[i].first;
            base_layer_t<value_type>* layer = (base_layer_t<value_type>*) net_layers->find(layer_id)->second;
            if (layer->get_layer_structure() == MIMO) {
                std::vector<tensor_t<value_type>*> t_outs = ((base_structure_t<value_type>*)layer)->get_outputs();
                // printf("forward alloc swap_block to layer%d-type%d t_outs.size()=%d\n", layer_id, layer->get_layer_type(), t_outs.size());
                for (int j = 0; j < t_outs.size(); j++) {
                    if (t_outs[j]->get_position() == SHARED_GPU_POOL) {
                        #ifdef DEBUG
                            printf("forward alloc swap_block to layer%d-type%d t_outs[%d]%d\n", layer_id, layer->get_layer_type(), j, t_outs[j]->get_tensor_id());
                        #endif
                        if (find_free_swap_block(FORWARD, &block_id, t_outs[j], false) == false) {
                            printf("forward there is no block for tensor%d-layer%d, all blocks is still using\n", t_outs[0]->get_tensor_id(), t_outs[0]->get_layer_id());
                            printSWAPBLOCK("offload failed");
                            return false;
                        }               
                        else {
                            if (block_id != -1) {  // block == -1 means this tensor has been allcated
                                if (swap_blocks[block_id]->tensor != NULL) {
                                    swap_blocks[block_id]->tensor->set_data_position(IN_CPU);
                                }
                                t_outs[j]->set_swap_block_id(block_id);
                                swap_blocks[block_id]->is_occup = true;
                                swap_blocks[block_id]->tensor = t_outs[j];
                                swap_blocks[block_id]->tensor_id = t_outs[j]->get_tensor_id();
                                t_outs[j]->set_data_position(IN_GPU);
                                t_outs[j]->set_data_position(IN_CPU_GPU);
                                #ifdef DEBUG
                                    printf("swap layer%d-type%d tensor%d-DATA block_id[%d]\n", layer_id, layer->get_layer_type(), t_outs[j]->get_tensor_id(), block_id);
                                    printSWAPBLOCK("find swap_block done");
                                #endif
                            } 
                        }
                    }
                    else if (t_outs[j]->get_position() == RECOMPUTE_IN_BACKWARD) {
                        if (t_outs[j]->get_data_position() != IN_GPU) {
                            if (find_free_recompute_block(FORWARD, &block_id, t_outs[j], false) == false) {
                                printf("forward there is no recompute_pool for tensor%d-layer%d, all blocks is still using\n", t_outs[j]->get_tensor_id(), t_outs[j]->get_layer_id());
                                printSWAPBLOCK("forwrd recompute failed");
                                return false;
                            }
                            else {
                                if (recompute_record[block_id] != NULL) {
                                    recompute_record[block_id]->set_data_position(DELETED);
                                }
                                recomputing_pool_flag[block_id] = true;
                                recompute_record[block_id] = t_outs[j];
                                t_outs[j]->set_data_position(IN_GPU);
                            }
                        }
                    }
                }    
            }
            else {
                if (layer->get_layer_type() == RNN  || layer->get_layer_type() == SATTN) {
					// tensor_t<value_type>* reserve_buff = ((rnn_layer_t<value_type>*)layer)->get_rnn_reserve_buff();
                    tensor_t<value_type>* reserve_buff = ((base_network_layer_t<value_type>*)layer)->get_reserve_buff();
                    #ifdef DEBUG
                        printf("forward alloc swap_block to layer%d-type%d reserve_buff%d\n", layer_id, layer->get_layer_type(), reserve_buff->get_tensor_id());
                    #endif
                    if (reserve_buff->get_position() == SHARED_GPU_POOL) {
                        if (find_free_swap_block(FORWARD, &block_id, reserve_buff, false) == false) {
                            printf("forward there is no block for reserve_buff%d-type%d-layer%d, all blocks is still using\n", reserve_buff->get_tensor_id(), reserve_buff->get_type(), reserve_buff->get_layer_id());
                            printSWAPBLOCK("offload failed");
                            return false;
                        }
                        else {
                            if (swap_blocks[block_id]->tensor != NULL) {
                                swap_blocks[block_id]->tensor->set_data_position(IN_CPU);
                            }
                            reserve_buff->set_swap_block_id(block_id);
                            swap_blocks[block_id]->is_occup = true;
                            swap_blocks[block_id]->tensor = reserve_buff;
                            swap_blocks[block_id]->tensor_id = reserve_buff->get_tensor_id();
                            reserve_buff->set_data_position(IN_GPU);
                            reserve_buff->set_data_position(IN_CPU_GPU);
                            #ifdef DEBUG
                                printf("swap layer%d-type%d tensor%d-reserve_buff block_id[%d]\n", layer_id, layer->get_layer_type(), reserve_buff->get_tensor_id(), block_id);
                                printSWAPBLOCK("find swap_block done");
                            #endif
                            // swap_blocks[block_id]->is_occup = true;
                        }
                    }
				}
                tensor_t<value_type>* output = ((base_network_layer_t<value_type>*)layer)->get_f_out();
                if (output->get_position() == SHARED_GPU_POOL) {
                    #ifdef DEBUG
                        printf("forward alloc swap_block to layer%d-type%d swap_tensor%d\n", layer_id, layer->get_layer_type(), output->get_tensor_id());
                    #endif
                    if (find_free_swap_block(FORWARD, &block_id, output, false) == false) {
                        printf("forward there is no block for tensor%d-layer%d, all blocks is still using\n", output->get_tensor_id(), output->get_layer_id());
                        printSWAPBLOCK("offload failed");
                        return false;
                    }
                    else {
                        if (swap_blocks[block_id]->tensor != NULL) {
                            swap_blocks[block_id]->tensor->set_data_position(IN_CPU);
                        }
                        output->set_swap_block_id(block_id);
                        swap_blocks[block_id]->is_occup = true;
                        swap_blocks[block_id]->tensor = output;
                        swap_blocks[block_id]->tensor_id = output->get_tensor_id();
                        output->set_data_position(IN_GPU);
                        output->set_data_position(IN_CPU_GPU);
                        #ifdef DEBUG
                            printf("swap layer%d-type%d tensor%d-DATA block_id[%d]\n", layer_id, layer->get_layer_type(), output->get_tensor_id(), block_id);
                            printSWAPBLOCK("find swap_block done");
                        #endif
                        // swap_blocks[block_id]->is_occup = true;
                    }
                }
                else if (output->get_position() == RECOMPUTE_IN_BACKWARD) {
                    // printf("forward alloc recompute_block to layer%d-type%d recompute_tensor%d\n", layer_id, layer->get_layer_type(), output->get_tensor_id());
                    if (find_free_recompute_block(FORWARD, &block_id, output, false) == false) {
                        printf("forward there is no recompute_pool for tensor%d-layer%d, all blocks is still using\n", output->get_tensor_id(), output->get_layer_id());
                        printSWAPBLOCK("forwrd recompute failed");
                        return false;
                    }
                    else {
                        if (recompute_record[block_id] != NULL) {
                            recompute_record[block_id]->set_data_position(DELETED);
                        }
                        recomputing_pool_flag[block_id] = true;
                        recompute_record[block_id] = output;
                        output->set_data_position(IN_GPU);
                        output->set_recompute_pool_id(block_id);
                        // printf("layer%d-type%d recompute tensor%d block_id[%d]\n", layer_id, layer->get_layer_type(), output->get_tensor_id(), block_id);
                    }
                }
            }
            if (layer->get_layer_type() == RNN  || layer->get_layer_type() == SATTN) {
                // tensor_t<value_type>* reserve_buff = ((rnn_layer_t<value_type>*)layer)->get_rnn_reserve_buff();
                tensor_t<value_type>* reserve_buff = ((base_network_layer_t<value_type>*)layer)->get_reserve_buff();
                reserve_buff->set_data_state(FORWARD_DELETE_OK);
            }
            // layer->increase_input_cur_use_counter(FORWARD);
            layer->fake_run(FORWARD, reg);
            #ifdef DEBUG
                printSWAPBLOCK("forward done before update");
            #endif
            layer->update_input_state(FORWARD);
        }
        else {
            break;
        }
    }
    reset_swap_block();
    reset_recompute_pool();
    for (size_t i = 0; i < net_comp_route->size(); i++) {
        if ((*net_comp_route)[i].second == BACKWARD) {  // BACKWARD
            layer_id = (*net_comp_route)[i].first;
            base_layer_t<value_type>* layer = (base_layer_t<value_type>*) net_layers->find(layer_id)->second;
            if (layer->get_layer_structure() == MIMO) {
                std::vector<tensor_t<value_type>*> t_ins = ((base_structure_t<value_type>*)layer)->get_inputs();
                for (int j = 0; j < t_ins.size(); j++) {
                    if (t_ins[t_ins.size()-1-j]->get_use_counter(BACKWARD) > 0) {
                        if (t_ins[t_ins.size()-1-j]->get_position() == SHARED_GPU_POOL) {
                            #ifdef DEBUG
                                printf("backward alloc swap_block to layer%d-type%d swap_tensor%d\n", layer_id, layer->get_layer_type(), t_ins[j]->get_tensor_id());
                            #endif
                            if (find_free_swap_block(BACKWARD, &block_id, t_ins[t_ins.size()-1-j], false) == false) {
                                printf("backward there is no block for tensor%d-layer%d, all blocks is still using\n", 
                                    t_ins[t_ins.size()-1-j]->get_tensor_id(), t_ins[t_ins.size()-1-j]->get_layer_id());
                                printSWAPBLOCK("prefetch failed");
                                return false;
                            }
                            else {
                                if (block_id != -1) {
                                    if (swap_blocks[block_id]->tensor != NULL) {
                                        swap_blocks[block_id]->tensor->set_data_position(NO_DATA);
                                    }
                                    #ifdef DEBUG
                                        printf("prefetch layer%d-type%d tensor%d-DATA block_id[%d]\n", layer_id, layer->get_layer_type(), t_ins[t_ins.size()-1-j]->get_tensor_id(), block_id);
                                        printSWAPBLOCK("find prefetch_block done");
                                    #endif
                                    t_ins[t_ins.size()-1-j]->set_prefetch_block_id(block_id);
                                    swap_blocks[block_id]->is_occup = true;
                                    swap_blocks[block_id]->tensor = t_ins[t_ins.size()-1-j];
                                    swap_blocks[block_id]->tensor_id = t_ins[t_ins.size()-1-j]->get_tensor_id();
                                    t_ins[t_ins.size()-1-j]->set_data_position(IN_GPU);
                                    // printf("prefetch layer%d-type%d tensor%d block_id[%d]\n", layer_id, layer->get_layer_type(), t_ins[t_ins.size()-1-j]->get_tensor_id(), block_id);
                                    // printSWAPBLOCK("prefetch");
                                }
                            }
                        }
                        else if (t_ins[t_ins.size()-1-j]->get_position() == RECOMPUTE_IN_BACKWARD) {
                            // printf("backward alloc recompute_block to layer%d-type%d t_in%d\n", layer_id, layer->get_layer_type(), t_ins[t_ins.size()-1-j]->get_tensor_id());
                            if (find_free_recompute_block(BACKWARD, &block_id, t_ins[t_ins.size()-1-j], false) == false) {
                                printf("backward there is no recompute_pool for tensor%d-layer%d, all blocks is still using\n", t_ins[t_ins.size()-1-j]->get_tensor_id(), t_ins[t_ins.size()-1-j]->get_layer_id());
                                // printSWAPBLOCK("forwrd recompute failed");
                                return false;
                            }
                            else {
                                if (recompute_record[block_id] != NULL) {
                                    recompute_record[block_id]->set_data_position(NO_DATA);
                                }
                                recomputing_pool_flag[block_id] = true;
                                recompute_record[block_id] = t_ins[t_ins.size()-1-j];
                                t_ins[t_ins.size()-1-j]->set_data_position(IN_GPU);
                                // printf("layer%d-type%d recompute tensor%d block_id[%d]\n", layer_id, layer->get_layer_type(), t_ins[t_ins.size()-1-j]->get_tensor_id(), block_id);
                            }
                        }
                    }
                }
            }
            else {
                if (layer->get_layer_type() == DATA_L) continue;
                if (layer->get_layer_type() == RNN  || layer->get_layer_type() == SATTN) {
                    // tensor_t<value_type>* reserve_buff = ((rnn_layer_t<value_type>*)layer)->get_rnn_reserve_buff();
                    tensor_t<value_type>* reserve_buff = ((base_network_layer_t<value_type>*)layer)->get_reserve_buff();
                    if (reserve_buff->get_position() == SHARED_GPU_POOL) {
                        // printf("backward alloc swap_block to layer%d-type%d swap_reserve_buff%d\n", layer_id, layer->get_layer_type(), reserve_buff->get_tensor_id());
                        if (find_free_swap_block(BACKWARD, &block_id, reserve_buff, false) == false) {
                            printf("backward there is no block for reserve_buff%d-type%d-layer%d, all blocks is still using\n", reserve_buff->get_tensor_id(), reserve_buff->get_type(), reserve_buff->get_layer_id());
                            printSWAPBLOCK("prefetch failed");
                            return false;
                        }
                        else {
                            if (swap_blocks[block_id]->tensor != NULL) {
                                swap_blocks[block_id]->tensor->set_data_position(NO_DATA);
                            }
                            reserve_buff->set_prefetch_block_id(block_id);
                            swap_blocks[block_id]->is_occup = true;
                            swap_blocks[block_id]->tensor = reserve_buff;
                            swap_blocks[block_id]->tensor_id = reserve_buff->get_tensor_id();
                            reserve_buff->set_data_position(IN_GPU);
                            // printf("prefetch layer%d-type%d reserve_buff%d block_id[%d]\n", layer_id, layer->get_layer_type(), reserve_buff->get_tensor_id(), block_id);
                            // printSWAPBLOCK("prefetch");
                        }
                    }
                }
                tensor_t<value_type>* input = ((base_network_layer_t<value_type>*)layer)->get_f_in();
                if (input->get_position() == SHARED_GPU_POOL) {
                    #ifdef DEBUG
                        printf("backward alloc swap_block to layer%d-type%d swap_tensor%d\n", layer_id, layer->get_layer_type(), input->get_tensor_id());
                    #endif
                    if (find_free_swap_block(BACKWARD, &block_id, input, false) == false) {
                        printf("backward there is no block for tensor%d-layer%d, all blocks is still using\n", input->get_tensor_id(), input->get_layer_id());
                        printSWAPBLOCK("prefetch failed");
                        return false;
                    }
                    else {
                        if (block_id != -1) {
                            if (swap_blocks[block_id]->tensor != NULL) {
                                swap_blocks[block_id]->tensor->set_data_position(NO_DATA);
                            }
                            input->set_prefetch_block_id(block_id);
                            swap_blocks[block_id]->is_occup = true;
                            swap_blocks[block_id]->tensor = input;
                            swap_blocks[block_id]->tensor_id = input->get_tensor_id();
                            input->set_data_position(IN_GPU);
                            #ifdef DEBUG
                                printf("prefetch layer%d-type%d tensor%d-DATA block_id[%d]\n", layer_id, layer->get_layer_type(), input->get_tensor_id(), block_id);
                                printSWAPBLOCK("find prefetch_block done");
                            #endif
                            // printf("prefetch layer%d-type%d tensor%d block_id[%d]\n", layer_id, layer->get_layer_type(), input->get_tensor_id(), block_id);
                            // printSWAPBLOCK("prefetch");
                        }
                    }
                }
                else if (input->get_position() == RECOMPUTE_IN_BACKWARD) {
                    // printf("backward alloc recompute_block to layer%d-type%d recompute_tensor%d\n", layer_id, layer->get_layer_type(), input->get_tensor_id());
                    if (find_free_recompute_block(BACKWARD, &block_id, input, false) == false) {
                        printf("backward there is no recompute_pool for tensor%d-layer%d, all blocks is still using\n", input->get_tensor_id(), input->get_layer_id());
                        printSWAPBLOCK("backward recompute failed");
                        return false;
                    }
                    else {
                        if (recompute_record[block_id] != NULL) {
                            recompute_record[block_id]->set_data_position(NO_DATA);
                        }
                        recomputing_pool_flag[block_id] = true;
                        recompute_record[block_id] = input;
                        input->set_data_position(IN_GPU);
                        input->set_backward_recompute_pool_id(block_id);
                        // printf("layer%d-type%d recompute tensor%d block_id[%d]\n", layer_id, layer->get_layer_type(), input->get_tensor_id(), block_id);
                    }
                }
            }
            if (layer->get_layer_type() == RNN  || layer->get_layer_type() == SATTN) {
                // tensor_t<value_type>* reserve_buff = ((rnn_layer_t<value_type>*)layer)->get_rnn_reserve_buff();
                tensor_t<value_type>* reserve_buff = ((base_network_layer_t<value_type>*)layer)->get_reserve_buff();
                reserve_buff->set_data_state(NO_COMPUTED);
            }
            // layer->increase_input_cur_use_counter(BACKWARD);
            layer->fake_run(BACKWARD, reg);
            #ifdef DEBUG
                printf("layer%d done\n", layer_id);
                printSWAPBLOCK("backward done before update");
            #endif
            layer->update_input_state(BACKWARD);
            layer->update_output_state(BACKWARD);

        }
    }
    reset_swap_block();
    #ifdef DEBUG
        printf("PreAllocateRecompteSwapBlock end%d\n", 2);
    #endif
}
// #undef DEBUG

template <class value_type>
bool mem_controller_t<value_type>::alloc_mem_by_swap_block(tensor_t<value_type>* tensor, net_comp dir) {
    // int block_id;
    // if (find_free_swap_block(dir, &block_id, tensor) == false) {
    //     printf("there is no block for tensor%d\n", tensor->get_tensor_id());
    //     return false;
    // }
    // int block_id = tensor->get_swap_block_id();
    // printf("alloc block[%d] to tensor%d\n", block_id, tensor->get_tensor_id());
    
    int block_id;
    if (dir == FORWARD) {
        block_id = tensor->get_swap_block_id();
        #ifdef DEBUG
        printf("start alloc_mem_by_swap_block[%d] for tensor%d layer%d \n", block_id, tensor->get_tensor_id(), tensor->get_layer_id());
        #endif
        if (swap_blocks[block_id]->tensor == tensor) {
            return true;
        }
        if (swap_blocks[block_id]->is_occup != false) {
            if (swap_blocks[block_id]->tensor->get_data_state() != FORWARD_DELETE_OK) {
                printf("there is no block for tensor%d, block[%d] is still used, is_occup=%d\n", tensor->get_tensor_id(), block_id, swap_blocks[block_id]->is_occup);
                printSWAPBLOCK("forward alloc_mem_by_swap_block failed");
                exit(1);
            }
        } 
        // printf("alloc_mem_by_swap_block[%d] for tensor%d layer%d \n", block_id, tensor->get_tensor_id(), tensor->get_layer_id());
        // if (find_free_swap_block(dir, &block_id, tensor, false) == false) {
        //     printf("there is no block for tensor%d, all blocks is still using\n", tensor->get_tensor_id());
        //     printSWAPBLOCK("forward alloc_mem_by_swap_block failed");
        //     exit(1);
        // }
        SWAP_STATE swap_state = swap_blocks[block_id]->swap_state;
        while( swap_state == READY) {
            swap_state = swap_blocks[block_id]->swap_state;
            usleep(1);
            // printf("");  // // without it, thread will block in this while. i don't know why.
            // printf("tensor%d-layer%d swap_blocks[%d]->swap_state%d blocktensor%d-layer%d-state%d\n", 
            //     tensor->get_tensor_id(), tensor->get_layer_id(), block_id, swap_blocks[block_id]->swap_state, 
            //     swap_blocks[block_id]->tensor->get_tensor_id(), swap_blocks[block_id]->tensor->get_layer_id(), swap_blocks[block_id]->tensor->get_data_state());
            // std::this_thread::sleep_for(std::chrono::milliseconds(50));
        }
        if (swap_blocks[block_id]->tensor != NULL) {
            // printf("tensor%d-layer in block[%d] has been offloaded to CPU\n", swap_blocks[block_id]->tensor->get_tensor_id(), swap_blocks[block_id]->tensor->get_layer_id(), block_id);
            TENSOR_DATA_STATE ds = swap_blocks[block_id]->tensor->get_data_state();
            if (ds == STILL_USED) {
                printf("FORWARD there is no block for tensor%d-layer%d, block[%d] is still using\n", tensor->get_tensor_id(), tensor->get_layer_id(), block_id);
                return false;
            }
            swap_blocks[block_id]->tensor->set_data_position(IN_CPU);  // original tensor data in block is only in CPU now
        }
        // while(1);
        tensor->set_gpu_ptr((value_type*)swap_blocks[block_id]->block_ptr);
        swap_blocks[block_id]->is_occup = true;
        // tensor->set_swap_block_id(swap_blocks[block_id]->id);
        tensor->set_gpu_ptr((value_type*)(swap_blocks[block_id]->block_ptr));
        swap_blocks[block_id]->tensor = tensor;
        swap_blocks[block_id]->tensor_id = tensor->get_tensor_id();
        swap_blocks[block_id]->swap_state = READY;
        tensor->set_data_state(STILL_USED);
        tensor->set_swap_block_id(block_id);
        // swap_ready_list.remove(swap_blocks[block_id]);
        // swap_ready_list.push_back(swap_blocks[block_id]);
        // swap_ready_list.push_back(swap_blocks[block_id]);   // swap_ready_list can control the sequence of swapping 
        // printf("tensor%d-layer%d has been set into block[%d]\n", swap_blocks[block_id]->tensor->get_tensor_id(), swap_blocks[block_id]->tensor->get_layer_id(), block_id);
        #ifdef DEBUG
            printSWAPBLOCK("forward alloc_mem_by_swap_block done:");
        #endif
        return true;
    }
    else {  // dir == BACKWARD
        block_id = tensor->get_prefetch_block_id();
        tensor_t<value_type>* tensor_in_block = swap_blocks[block_id]->tensor;
        int block_tensor_id = swap_blocks[block_id]->tensor_id;  // last tensor in this block
        SWAP_STATE swap_state;
        // printf("alloc_mem_by_swap_block for tensor%d layer%d, block_id = %d, block_tensor%d\n", 
            // tensor->get_tensor_id(), tensor->get_layer_id(), block_id, block_tensor_id);
        // printSWAPBLOCK("backward alloc_mem_by_swap_block");
        if (tensor_in_block == tensor) {
            while(true) {
                swap_state = swap_blocks[block_id]->swap_state;
                if (swap_state == DONE) break; 
                // printf("");
                usleep(1);
            }
            tensor->set_data_position(IN_GPU);
            return true;
        }
        else {  
            if (tensor_in_block->get_data_state() == NO_COMPUTED) {
                while(true) { 
                    tensor_in_block = swap_blocks[block_id]->tensor;
                    // block_tensor_id = swap_blocks[block_id]->tensor_id;
                    if (tensor_in_block == tensor) break;
                    // if (block_tensor_id == tensor->get_tensor_id()) break;
                    // printf("");
                    usleep(1);
                }
                while(true) {
                    swap_state = swap_blocks[block_id]->swap_state;
                    if (swap_state == DONE) break; 
                    // printf("");
                    usleep(1);
                }
                tensor->set_data_position(IN_GPU);
                return true;
            }
            else {  // tensor_in_block != the tensor of this layer, tensor_in_block is still previous swap_tensor with still used
                printf("BACKWARD there is no block for tensor%d-layer%d, block[%d] is still using\n", tensor->get_tensor_id(), tensor->get_layer_id(), block_id);
                printSWAPBLOCK("block locked");
                return false;
            }
        }
    }
}

// #define DEBUG
template <class value_type>
void mem_controller_t<value_type>::swap_ctrl(net_comp dir, MODEL_TYPE model_type, tensor_t<value_type>* tensor, double* pur_swap_time) {
    // if (model_type == CNN_NETWORK) 
    {
        if (dir == FORWARD) {
            // printf("tensor%d is ready to offload\n", tensor->get_tensor_id());
            TENSOR_DATA_POSITION data_position = tensor->get_data_position();
            while(data_position != IN_GPU && data_position != IN_CPU_GPU) {
                data_position = tensor->get_data_position();
                #ifdef DEBUG
                    // printf("tensor%d-layer%d ", tensor->get_tensor_id(), tensor->get_layer_id());  // without it, thread will block in this while. i don't know why.
                #endif
                usleep(1);
                // printf("");
                #ifdef DEBUG
                    // printf("tensor%d->data_position = %d\n", tensor->get_tensor_id(), data_position);
                #endif
            }
            if (data_position == IN_GPU) {
                // printf("start offload tensor%d-layer%d-block[%d]-data_position%d to CPU block[%d]->tensor%d-layer%d\n", 
                //     tensor->get_tensor_id(), tensor->get_layer_id(), tensor->get_swap_block_id(), tensor->get_data_position(), tensor->get_swap_block_id(), swap_blocks[tensor->get_swap_block_id()]->tensor->get_tensor_id(), swap_blocks[tensor->get_swap_block_id()]->tensor->get_layer_id());
                if (tensor->get_use_counter(BACKWARD) == 0 && tensor->get_type() != RNN_RESERVE) {  // tensor need not be use in backward, Just update the state without actually offloading
                // if (false) {
                    if (pur_swap_time != NULL) {  // recode swapping time
                        *pur_swap_time = 0.0f;
                        // printf("out tensor%d->mem=%zd-time%lf\n", tensor->get_tensor_id(), tensor->get_mem_size(), *pur_swap_time);
                    }
                    else {

                    }
                }
                else {
                    if (pur_swap_time != NULL) {  // recode swapping time
                        double start = get_cur_time();
                        tensor->async_gpu_to_cpu();
                        synchronize(GPU2CPU_STREAM);
                        *pur_swap_time = get_cur_time() - start;
                        // printf("out tensor%d->mem=%zd-time%lf\n", tensor->get_tensor_id(), tensor->get_mem_size(), *pur_swap_time);
                    }
                    else {
                        tensor->async_gpu_to_cpu();                                 
                        synchronize(GPU2CPU_STREAM);
                    }
                }
                swap_blocks[tensor->get_swap_block_id()]->swap_state = DONE;
                swap_blocks[tensor->get_swap_block_id()]->tensor->set_data_position(IN_CPU_GPU);
                #ifdef DEBUG
                    printf("tensor%d-layer%d-block[%d] has been offloaded\n", tensor->get_tensor_id(), tensor->get_layer_id(), tensor->get_swap_block_id());
                    printSWAPBLOCK("swap_ctrl");
                #endif
            }
        }
        else {  // dir == BACKWARD
            int block_id = tensor->get_prefetch_block_id();
            if (tensor->get_data_position() == IN_GPU || tensor->get_data_position() == IN_CPU_GPU) {
                swap_blocks[block_id]->swap_state = DONE;
                swap_blocks[block_id]->tensor->set_data_position(IN_GPU);
            }
            else {
                TENSOR_DATA_STATE ds;
                tensor_t<value_type>* tensor_in_block = swap_blocks[block_id]->tensor;
                if (tensor_in_block == tensor) {
                    // the data of last swap_tensor has not been cover, so it still in GPU
                    swap_blocks[block_id]->tensor->set_data_position(IN_GPU);
                    swap_blocks[block_id]->swap_state = DONE;
                    // printSWAPBLOCK("BACKWARD swap_ctrl");
                }
                else {
                    while(tensor_in_block != tensor) {  // tensor has not block
                        ds = swap_blocks[block_id]->tensor->get_data_state();
                        if (ds == NO_COMPUTED) break;
                        // printf("");
                        usleep(1);
                        // printSWAPBLOCK("BACKWARD tensor_in_block != tensor");
                    }
                    
                    tensor->set_gpu_ptr((value_type*)(swap_blocks[block_id]->block_ptr));
                    swap_blocks[block_id]->tensor = tensor;
                    swap_blocks[block_id]->tensor_id = tensor->get_tensor_id();
                    #ifdef DEBUG 
                    printf("\nstart prefetch tensor%d-layer%d-block[%d]-data_position%d to GPU block[%d]->tensor%d-layer%d\n", 
                        tensor->get_tensor_id(), tensor->get_layer_id(), tensor->get_swap_block_id(), tensor->get_data_position(), tensor->get_swap_block_id(), swap_blocks[tensor->get_swap_block_id()]->tensor->get_tensor_id(), swap_blocks[tensor->get_swap_block_id()]->tensor->get_layer_id());
                    #endif
                    swap_blocks[block_id]->tensor->set_data_state(STILL_USED);      
                    swap_blocks[block_id]->swap_state = READY;
                    if (tensor->get_data_position() == IN_CPU) {
                        if (pur_swap_time != NULL) {  // recode swapping time
                            double start = get_cur_time();
                            tensor->async_cpu_to_gpu();
                            synchronize(CPU2GPU_STREAM);
                            *pur_swap_time = get_cur_time() - start;
                        }
                        else {
                            tensor->async_cpu_to_gpu();
                            synchronize(CPU2GPU_STREAM);
                        }
                    }
                    swap_blocks[block_id]->swap_state = DONE;
                    swap_blocks[block_id]->tensor->set_data_position(IN_GPU);  
                    #ifdef DEBUG 
                        printf("\ntensor%d-layer%d-block[%d] has been prefetched\n", tensor->get_tensor_id(), tensor->get_layer_id(), tensor->get_swap_block_id());
                        printSWAPBLOCK("BACKWARD prefetched");
                    #endif
                }
            }  
        }
    } 
}
// #undef DEBUG

template <class value_type>
void mem_controller_t<value_type>::init_swap_memory_pool(size_t pool_size) {    // init_gpu_memory_pools

#ifdef POOL_MALLOC_MODE
    // multi swapping block
#ifdef MULTI_SWAP_BLOCK
    SWAP_BLOCK *sb[SWAP_BLOCK_NUM];
	for (int i = 0; i < SWAP_BLOCK_NUM; i++) {
        sb[i] = new SWAP_BLOCK();
		swap_blocks.push_back(sb[i]);
        swap_blocks[i]->id = i;
        swap_blocks[i]->is_occup = false;
        swap_blocks[i]->swap_state = NO_STATE;
	}
    for (int i = 0; i < swap_blocks.size(); i++) {
        swap_blocks[i]->block_size = pool_size / SWAP_BLOCK_NUM;
        this->alloc_mem_by_gpu_mem_pool(&(swap_blocks[i]->block_ptr), swap_blocks[i]->block_size);
        printf("after init swap_blocks[%d], pool_free_size = %zd\n", i, this->pool_free_size);
        swap_blocks[i]->is_occup = false;
        swap_blocks[i]->tensor = NULL;
        swap_ready_list.push_back(swap_blocks[i]);
    }
    reset_swap_block();
#else
    this->alloc_mem_by_gpu_mem_pool((void**)&shared_block_ptr, pool_size);
    shared_block_size_by_byte = pool_size;
    shared_block_offset = 0;
#endif
#else
	cudaMalloc((void**)&this->shared_block_ptr, pool_size);
	// cudaMalloc((void**)&this->shared_block_ptr2, pool_size);
	value_type* cpu_block_ptr = (value_type*)malloc(pool_size);
	// value_type* cpu_block_ptr2 = (value_type*)malloc(pool_size);
	size_t total = pool_size / sizeof(value_type);
	for ( size_t i = 0; i < total; i++ ) {
		*(cpu_block_ptr+i) = 3.14;
		// *(cpu_block_ptr2+i) = 6.28 + i;
	}
	// cudaMemcpy((void*)this->shared_block_ptr, (void*)cpu_block_ptr, pool_size, cudaMemcpyHostToDevice);
	// cudaMemcpy((void*)this->shared_block_ptr2, (void*)cpu_block_ptr2, pool_size, cudaMemcpyHostToDevice);
	shared_block_offset = 0;
	shared_block_size_by_byte = pool_size;
	shared_block_free_space = pool_size;
	// shared_block_offset2 = 0;
	// shared_block_size_by_byte2 = pool_size;
	// shared_block_free_space2 = pool_size;
    // printf("")
#endif
}

template <class value_type>
void mem_controller_t<value_type>::init_grad_memory_pool(size_t max_grad_size) {
#ifdef POOL_MALLOC_MODE
// printf("start init_grad_memory_pool max_grad_size = %zd\n", max_grad_size);
    this->alloc_mem_by_gpu_mem_pool((void**)&grad_shared_block_ptr, max_grad_size);
// printf("end init_grad_memory_pool max_grad_size = %zd\n", max_grad_size);
// while(1);
#else
    cudaMalloc((void**)&this->grad_shared_block_ptr, max_grad_size);
#endif
    value_type* grad_cpu_block_ptr = (value_type*)malloc(max_grad_size);
    size_t total = max_grad_size / sizeof(value_type);
	for ( size_t i = 0; i < total; i++ ) {
		*(grad_cpu_block_ptr+i) = 3.14;
		// *(cpu_block_ptr2+i) = 6.28 + i;
	}
    // printf("start init_grad_memory_pool max_grad_size = %zd\n", max_grad_size);
    cudaMemcpy((void*)this->grad_shared_block_ptr, (void*)grad_cpu_block_ptr, pool_size, cudaMemcpyHostToDevice);
    grad_shared_block_offset = 0;
	grad_shared_block_size_by_byte = max_grad_size;
	grad_shared_block_free_space = max_grad_size;
    // printf("end init_grad_memory_pool max_grad_size = %zd\n", max_grad_size);
    // alloc temp_grad_gpu_ptr from grad_shared_block_ptr
    auto net_layers = reg->get_net_layers();
    for (auto it = net_layers.begin(); it != net_layers.end(); it++) {
        base_layer_t<value_type>* layer = (base_layer_t<value_type> *) it->second;
        tensor_t<value_type>* tensor;
        if (layer->get_layer_type() == BN) {  
            // the weight_grad and bias_grad are obtained by one function; thus their ptr are different 
            // (both are much smaller than max_grad_size, so we can alloc memory for them from the grad_shared_block_ptr)
			tensor = ((base_network_layer_t<value_type>*)layer)->get_weight_grad();
            tensor->set_temp_gpu_ptr((value_type*)((void*)(this->grad_shared_block_ptr)+grad_shared_block_offset));
            grad_shared_block_offset += tensor->get_mem_size();
            tensor = ((base_network_layer_t<value_type>*)layer)->get_bias_grad();
            tensor->set_temp_gpu_ptr((value_type*)((void*)(this->grad_shared_block_ptr)+grad_shared_block_offset));
            grad_shared_block_offset = 0;
		}
        else if(layer->get_layer_type() == CONV) {
            tensor = ((base_network_layer_t<value_type>*)layer)->get_weight_grad();
            tensor->set_temp_gpu_ptr((value_type*)((void*)(this->grad_shared_block_ptr)));
            grad_shared_block_offset += tensor->get_mem_size();
            tensor = ((base_network_layer_t<value_type>*)layer)->get_bias_grad();
            tensor->set_temp_gpu_ptr((value_type*)((void*)(this->grad_shared_block_ptr)));
            grad_shared_block_offset = 0;
        }
        else if(layer->get_layer_type() == FC) {
            tensor = ((base_network_layer_t<value_type>*)layer)->get_weight_grad();
            tensor->set_temp_gpu_ptr((value_type*)((void*)(this->grad_shared_block_ptr)));
            grad_shared_block_offset += tensor->get_mem_size();
            tensor = ((base_network_layer_t<value_type>*)layer)->get_bias_grad();
            tensor->set_temp_gpu_ptr((value_type*)((void*)(this->grad_shared_block_ptr)));
            grad_shared_block_offset = 0;
        }
        else if(layer->get_layer_type() == RNN || layer->get_layer_type() == SATTN) {
            tensor = ((base_network_layer_t<value_type>*)layer)->get_weight_grad();
            tensor->set_temp_gpu_ptr((value_type*)((void*)(this->grad_shared_block_ptr)));
        }
        else {
            continue;
        }
    }
}

template <class value_type>
void mem_controller_t<value_type>::init_cpu_pool(size_t mem_size) {
    checkCudaErrors( cudaMallocHost(&(this->cpu_pool), mem_size) ); 
    cpu_pool_offset = 0;
    cpu_pool_free_size = mem_size;
    cpu_pool_size = mem_size;
}

template <class value_type>
void mem_controller_t<value_type>::alloc_mem_by_cpu_pool(void* cpu_ptr, size_t mem_size) {
    if (cpu_pool_free_size >= mem_size) {
        cpu_ptr = cpu_pool + cpu_pool_offset;
        cpu_pool_offset += mem_size;
        cpu_pool_free_size -= mem_size;
    } 
    else {
        printf("error: cpu_pool_free_size = %zd, need = %zd\n", cpu_pool_free_size, mem_size);
        exit(0);
    }
}

template <class value_type>
void mem_controller_t<value_type>::init_all_tensors(
        size_t gpu_pool_size, size_t swap_pool_size, size_t recompute_pool_size, size_t max_grad_size, size_t max_data_size, 
        size_t b_data_pool_size, size_t QKV_buffer_size, size_t max_workspace_size) 
{
    std::vector<tensor_t<value_type>* > *all_tensors = reg->get_all_tensors();
    int all_tensor_count = 0;
    int data_tensor_count = 0;
    size_t data_tensor_size = 0;
    size_t undata_tensor_size = 0;
    printf("before malloc_gpu_mem_pool query_used_mem = %f\n", BYTE_TO_MB(query_used_mem()));
    this->malloc_gpu_mem_pool(gpu_pool_size);
    printf("after malloc_gpu_mem_pool query_used_mem = %f\n", BYTE_TO_MB(query_used_mem()));
    size_t tensor_size = 0;
    void* gpu_ptr;
    printf("total tensor number = %d\n", all_tensors->size());

    alloc_mem_by_gpu_mem_pool(&gpu_ptr, max_workspace_size);
    workspace_pool = gpu_ptr;
    printMemPool("after init workspace_pool");
    // printf("alloc cpu buffer for swapped tensors: ");
    size_t cpu_mem_sum = 0;
    // for (auto t = all_tensors->begin(); t != all_tensors->end(); ++t) {
    //     if ((*t)->get_position() == SHARED_GPU_POOL) {
    //         // printf("1 t%d mem%f cpu_mem_sum=%f\n", (*t)->get_tensor_id(), BYTE_TO_MB((*t)->get_mem_size()), BYTE_TO_MB(cpu_mem_sum));
    //         // (*t)->acquireSpaceCPU((*t)->get_mem_size());
    //         cpu_mem_sum += (*t)->get_mem_size();
    //         // printf("2 t%d mem%f cpu_mem_sum=%f\n", (*t)->get_tensor_id(), BYTE_TO_MB((*t)->get_mem_size()), BYTE_TO_MB(cpu_mem_sum));
    //     }
    // }
    // init_cpu_pool(cpu_mem_sum);
    for (auto t = all_tensors->begin(); t != all_tensors->end(); ++t) {
        // printf("alloc mem for tensor%d-type%d(from layer%d) start\n", (*t)->get_tensor_id(), (*t)->get_type(), (*t)->get_layer_id());
        all_tensor_count++;
        // if (((*t)->get_position() == REMAIN_IN_GPU) && ((*t)->get_type() != B_DATA)) {
        // // if ((*t)->get_position() == REMAIN_IN_GPU) {
        if ((*t)->get_position() == SHARED_GPU_POOL) {
            // alloc_mem_by_cpu_pool((void*)(*t)->get_cpu_ptr(), (*t)->get_mem_size());
            (*t)->acquireSpaceCPU((*t)->get_mem_size()/sizeof(value_type));
        }
        if ((*t)->get_position() == REMAIN_IN_GPU && (*t)->get_type() != B_DATA && (*t)->get_type() != QKV_DATA && (*t)->get_type() != DQKV_DATA
            && (*t)->get_type() != CONV_BUFF && (*t)->get_type() != RNN_BUFF ) 
        {
            tensor_size = (*t)->get_mem_size();
            alloc_mem_by_gpu_mem_pool(&gpu_ptr, tensor_size);
            (*t)->set_gpu_ptr((value_type*)gpu_ptr);
            (*t)->set_data_position(NO_DATA);
            (*t)->set_data_state(NO_COMPUTED);
            // (*t)->printTensorData("init_all_tensors", 2);
            // (*t)->printTensorData("init_all_tensors", 2);
            if ((*t)->get_type() == PARAM || (*t)->get_type() == RNN_PARAM) {
                (*t)->CPUtoGPU();
                (*t)->set_data_position(IN_GPU);
                (*t)->set_data_state(STILL_USED);
            }
            else if ((*t)->get_type() == AUX) { 
                // for full_connected_layer bias_mu
                for (size_t i = 0; i < (*t)->get_N(); i++) {
                    (*t)->set_scalar(0, 0, 0, i, 1);
                    printf("AUX mem = %f\n", BYTE_TO_MB((*t)->get_mem_size()));
                }
            }
            // (*t)->printTensorData("t", 2);
        }
        if ((*t)->get_type() == CONV_BUFF || (*t)->get_type() == RNN_BUFF) {
            (*t)->set_gpu_ptr((value_type*)workspace_pool);
        }
    }
    // while(1);
    printf("after init tensor REMAIN_IN_GPU, pool_free_size = %zd\n", this->pool_free_size);
    for (int i = 0; i < B_DATA_POOL_NUM; i++) {
        alloc_mem_by_gpu_mem_pool(&gpu_ptr, b_data_pool_size/B_DATA_POOL_NUM );
        this->b_data_pool_ptr[i] = gpu_ptr;
        this->b_data_pool_flag[i] = 0;
        b_data_pool_tensors[i] = NULL;
    }
    printMemPool("after init b_data_pool");

    for (int i = 0; i < 3; i++) {
        alloc_mem_by_gpu_mem_pool(&gpu_ptr, QKV_buffer_size/6 );
        this->QKV_buffer[i] = gpu_ptr;
    }
    for (int i = 0; i < 3; i++) {
        alloc_mem_by_gpu_mem_pool(&gpu_ptr, QKV_buffer_size/6 );
        this->dQKV_buffer[i] = gpu_ptr;
    }
    printMemPool("after init QKV_buffer");

    this->init_grad_memory_pool(max_grad_size);
    printMemPool("after init_grad_memory_pool");

#ifdef RECOMPUTE_ON
    for (int i = 0; i < RECOMPUTE_POOL_NUM; i++) {
        alloc_mem_by_gpu_mem_pool(&gpu_ptr, recompute_pool_size/RECOMPUTE_POOL_NUM); 
        this->recomputing_pool[i] = (value_type*)gpu_ptr;
        this->recomputing_pool_flag[i] = 0;
        recompute_record[i] = NULL;
    }
#endif
    
#ifdef SWAP_ON
    this->init_swap_memory_pool(swap_pool_size);
    printSWAPBLOCK("after init_swap_memory_pool");
    printMemPool("after init_swap_memory_pool");
    // for (int )
	// value_type* cpu_block_ptr = (value_type*)malloc(swap_pool_size);
	// // value_type* cpu_block_ptr2 = (value_type*)malloc(pool_size);
	// size_t total = pool_size / sizeof(value_type);
	// for ( size_t i = 0; i < total; i++ ) {
	// 	*(cpu_block_ptr+i) = 3.14;
	// 	// *(cpu_block_ptr2+i) = 6.28 + i;
	// }
	// cudaMemcpy((void*)this->shared_block_ptr, (void*)cpu_block_ptr, pool_size, cudaMemcpyHostToDevice);
#endif
    
#ifdef MULTI_SWAP_BLOCK
    printSWAPBLOCK("after init_all_tensors");
    // printMemPool("after init b_data_pool");
#endif
}

template <class value_type>
void mem_controller_t<value_type>::init_all_tensor_gpu_memory() {
// 	std::vector<tensor_t<value_type>* > *all_tensors = reg->get_all_tensors();
//     int all_tensor_count = 0;
//     int data_tensor_count = 0;
//     size_t data_tensor_size = 0;
//     size_t undata_tensor_size = 0;

// // #ifdef POOL_MALLOC_MODE
//     // size_t net_total_size = reg->get_total_size_pool_malloc_mode();
//     // printf("before malloc_gpu_mem_pool query_used_mem = %f\n", BYTE_TO_MB(query_used_mem()));
//     this->malloc_gpu_mem_pool(net_total_size);
//     printf("after malloc_gpu_mem_pool query_used_mem = %f\n", BYTE_TO_MB(query_used_mem()));
//     size_t tensor_size = 0;
//     void* gpu_ptr;
//     for (auto t = all_tensors->begin(); t != all_tensors->end(); ++t) {
//         // printf("alloc mem for tensor%d-type%d(from layer%d) start\n", (*t)->get_tensor_id(), (*t)->get_type(), (*t)->get_layer_id());
//         all_tensor_count++;
//         if ((*t)->get_position() == REMAIN_IN_GPU) {
//             tensor_size = (*t)->get_mem_size();
//             // gpu_ptr = (*t)->get_gpu_ptr();
//             alloc_mem_by_gpu_mem_pool(&gpu_ptr, tensor_size);
//             (*t)->set_gpu_ptr((value_type*)gpu_ptr);
//             if ((*t)->get_type() == PARAM) {
//                 (*t)->CPUtoGPU();
//             }
//             else if ((*t)->get_type() == AUX) { 
//                 // for full_connected_layer bias_mu
//                 for (size_t i = 0; i < (*t)->get_N(); i++) {
//                     (*t)->set_scalar(0, 0, 0, i, 1);
//                 }
//             }
//         }
//         // printf("alloc mem %zdbytes for tensor%d-type%d(from layer%d) done\n", tensor_size, (*t)->get_tensor_id(), (*t)->get_type(), (*t)->get_layer_id());
//     }
//     // init ALL PARAM, just cpu2gpu, PARAM data has been random in fsetup and bsetup

// // #endif
//     printf("init_all_tensor_gpu_memory, all_tensor_count = %d, net_total_size = %d\n", all_tensor_count, net_total_size);
//     // printf("data_tensor_size = %f, undata_tensor_size = %f\n", BYTE_TO_MB(data_tensor_size), BYTE_TO_MB(undata_tensor_size));
//     // while(1);
}

template <class value_type>
void mem_controller_t<value_type>::free_tensor_shared_memory() {
	checkCudaErrors(cudaFree(this->ptr));
	this->ptr = NULL;
	offset = 0;
	free_space = 0;
	device = 0;
	checkCudaErrors(cudaFree(this->ptr2));
	this->ptr2 = NULL;
	offset2 = 0;
	free_space2 = 0;
	device2 = 0;
}

template <class value_type>
void mem_controller_t<value_type>::stash_tensor_shared_memory(int layer_id, net_comp dir, network_stage stage) {

	#ifdef DEBUG
    	auto tmp = reg->get_net_layers().find(layer_id);
	#endif

	base_layer_t<value_type>* current_layer = (base_layer_t<value_type>*)reg->get_net_layers().find(layer_id)->second;

	std::vector<std::vector<tensor_t<value_type> *> > *ins;
    std::vector<tensor_t<value_type>* >* tensors;
	// 取得ins列表
    if (dir == FORWARD) {
        // ins = (std::vector<std::vector<tensor_t<value_type> *> > *) live_anls->get_f_stash_tensors();
        tensors = reg->get_forward_dependency(layer_id);
    } 
	else if (dir == BACKWARD) {
        // ins = (std::vector<std::vector<tensor_t<value_type> *> > *) live_anls->get_b_stash_tensors();
        tensors = reg->get_backward_dependency(layer_id);
    }

    // 遍历layer_id的dependent tensors
    int tt = 0; 
    // printf("layer %d\n", layer_id);
	// if ((layer_id%2) == 0) {
    if (true) {
		for (auto it = tensors->begin(); it != tensors->end(); ++it) {
			tensor_t<value_type>* t = *it;  // 取ins(layer_id)的一个tensor
			// if (t->get_type() != DATA && t->get_type() != B_DATA
            //     && t->get_type() != RNN_BUFF && t->get_type() != CONV_BUFF && t->get_type() != RNN_RESERVE) 
            // {
            //     // printf("tensorid = %d, type = %d, %x\n", t->get_tensor_id(), t->get_type(), t->get_gpu_ptr());
            //     // t->printTensor("param result = ");
			// 	continue;
			// }
			// t->stash_specified_gpu_space(ptr, offset);
            void* gpu_ptr = t->get_gpu_ptr();
            alloc_mem_by_gpu_mem_pool(&gpu_ptr, t->get_mem_size());
            t->set_gpu_ptr((value_type*)gpu_ptr);
            // if (current_layer->get_layer_type() == ACT) {
                // printf("stash_tensor_shared_memory layer%d tensorid = %d, type = %d, %x, size=%d\n", layer_id, t->get_tensor_id(), t->get_type(), t->get_gpu_ptr(), t->get_mem_size());
            // }
 
            // t->printTensor("param result = ");
			// offset += t->get_scalar_count();
            offset += t->get_mem_size();
			tt++;
            if (layer_id == 2) {
                // t->printTensor("layer%d data result = ");
                // while(1);
            }
		}
	}
}

template <class value_type>
void mem_controller_t<value_type>::stash_tensor_malloc_all(int layer_id, net_comp dir, network_stage stage) {

	#ifdef DEBUG
    	auto tmp = reg->get_net_layers().find(layer_id);
	#endif

	auto current_layer = reg->get_net_layers().find(layer_id);

	std::vector<std::vector<tensor_t<value_type> *> > *ins;
    std::vector<tensor_t<value_type>* >* tensors;

    // 取得dependency列表
    if (dir == FORWARD) {
        tensors = reg->get_forward_dependency(layer_id);
    } 
	else if (dir == BACKWARD) {
        tensors = reg->get_backward_dependency(layer_id);
    }
    for (auto it = tensors->begin(); it != tensors->end(); ++it) {
        tensor_t<value_type> *t = *it;  // 取ins(layer_id)的一个tensor
        if (t->get_type() != DATA && t->get_type() != CONV_BUFF) continue;
		t->stash_gpu_space();
    }

    // printf("\n end stash_tensor_shared_memory\n\n");
}

template <class value_type>
void mem_controller_t<value_type>::update_tensor_state_shared_memory(int layer_id, net_comp dir, network_stage stage) {
    this->reset_gpu_mem_pool_offset();
	std::vector<tensor_t<value_type>* >* tensors;
	if (dir == FORWARD) {
        // ins = (std::vector<std::vector<tensor_t<value_type> *> > *) live_anls->get_f_stash_tensors();
        tensors = reg->get_forward_dependency(layer_id);
    } 
	else if (dir == BACKWARD) {
        // ins = (std::vector<std::vector<tensor_t<value_type> *> > *) live_anls->get_b_stash_tensors();
        tensors = reg->get_backward_dependency(layer_id);
    }
	for (auto it = tensors->begin(); it != tensors->end(); ++it) {
        tensor_t<value_type> *t = *it;  // 取ins(layer_id)的一个tensor
        if (t->get_type() != DATA) {
            continue;
        }
	}
	this->offset = 0;
	this->offset2 = 0;
	
#ifdef MEM_DEBUG
    printf("\n-------outs------\n");
    for (int i = 1; i < outs->size(); ++i) {
        printf("---layer %d\n", i);
        for (auto it = outs->operator[](i).begin(); it!=outs->operator[](i).end(); ++it) {
            printf("%p ", (*it));
        }
        printf("\n");
    }
#endif	
}

template <class value_type>
void mem_controller_t<value_type>::update_tensor_state_free_all(int layer_id, net_comp dir, network_stage stage) {

    std::vector<tensor_t<value_type>* >* tensors;
    if (dir == FORWARD) {
        tensors = reg->get_forward_dependency(layer_id);
    } 
	else if (dir == BACKWARD) {
        tensors = reg->get_backward_dependency(layer_id);
    }
    for (auto it = tensors->begin(); it != tensors->end(); ++it) {
        tensor_t<value_type> *t = *it;  // 取ins(layer_id)的一个tensor
        if (t->get_type() != DATA && t->get_type() != CONV_BUFF) continue;
		t->free_gpu_space(VOID);
    }
	
#ifdef MEM_DEBUG
    printf("\n-------outs------\n");
    for (int i = 1; i < outs->size(); ++i) {
        printf("---layer %d\n", i);
        for (auto it = outs->operator[](i).begin(); it != outs->operator[](i).end(); ++it) {
            printf("%p ", (*it));
        }
        printf("\n");
    }
#endif	
}

/*--------control--start--------------------*/
// #define DEBUG
template <class value_type>
int mem_controller_t<value_type>::stash_tensor(int layer_id, base_layer_t<value_type>* layer, net_comp dir, network_stage stage, std::stack<int>* recompute_layers_stack) 
{
	tensor_t<value_type>* last_tensor;
	std::vector<tensor_t<value_type>*> last_tensors;
	base_layer_t<value_type>* last_layer;
	std::map<int, void* >* net_layers  = reg->get_net_layers_ptr();
    // printf("stash_tensor layerid = %d\n", layer_id);
    // #ifdef DEBUG
    //     int test_layer_id = 1877;
    //     base_layer_t<value_type>* test_layer = (base_layer_t<value_type>*) net_layers->find(test_layer_id)->second;
    //     tensor_t<value_type>* test_tensor = ((base_network_layer_t<value_type>*)test_layer)->get_f_out();
    //     test_tensor->printTensorData("test_tensor stash_tensor", 2);
    // #endif
    if (dir == RECOMPUTE) {
        if (layer->get_layer_type() == DATA_L) return 0;
        tensor_t<value_type>* t_out = ((base_network_layer_t<value_type>*)layer)->get_f_out();
        // if (layer->get_layer_position() == RECOMPUTE_LAYER) {
        if (t_out->get_position() == RECOMPUTE_IN_BACKWARD) {
            // printf("recompute layer")
            // tensor_t<value_type>* t_out = ((base_network_layer_t<value_type>*)layer)->get_f_out();
            int pool_id = t_out->get_backward_recompute_pool_id();
            t_out->set_gpu_ptr((value_type*)recomputing_pool[pool_id]);
            recomputing_pool_flag[pool_id] = true;
            if (recompute_record[pool_id] != NULL) {
                recompute_record[pool_id]->set_data_position(DELETED);
                #ifdef DEBUG
                    printf("dir%d: layer%d-type%d alloc space for tensor%d from recomputing_pool[%d], last tensor%d is %d\n", 
                        dir, layer->get_base_id(), layer->get_layer_type(), t_out->get_tensor_id(), pool_id, recompute_record[pool_id]->get_tensor_id(), recompute_record[pool_id]->get_data_position());
                #endif
            }
            recompute_record[pool_id] = t_out;
            // for (size_t i = 0; i < RECOMPUTE_POOL_NUM; i++) {
            //     if (recomputing_pool_flag[i] == false) {
            //         t_out->set_gpu_ptr((value_type*)recomputing_pool[i]);
            //         // t_out->stash_specified_gpu_space(recomputing_pool[i], shared_block_offset);
            //         recomputing_pool_flag[i] = 1;
            //         t_out->set_recompute_pool_id(i);
            //         if (recompute_record[i] != NULL) {
            //             recompute_record[i]->set_data_position(DELETED);  // the last tensor in the recomputing_pool[i] is deleted
            //             // printf("dir%d: layer%d-type%d alloc space for tensor%d from recomputing_pool[%d], last tensor%d is %d\n", 
            //             //     dir, layer->get_base_id(), layer->get_layer_type(), t_out->get_tensor_id(), i, recompute_record[i]->get_tensor_id(), recompute_record[i]->get_data_state());
            //         }
            //         recompute_record[i] = t_out;  // record that the t_out occupies the recomputing_pool[i] 
            //         break;
            //     }
            // }
            tensor_t<value_type>* t_in = ((base_network_layer_t<value_type>*)layer)->get_f_in();
            // printf("layer%d-type%d stash tensor%d-layer%d-position%d\n", 
            //     layer->get_base_id(), layer->get_layer_type(), t_in->get_tensor_id(), t_in->get_layer_id(), t_in->get_position());
            // printf("recompute_layers_stack.size() = %d\n", recompute_layers_stack->size());
            if (t_in->get_position() == RECOMPUTE_IN_BACKWARD) {
                // printf("dir%d: layer%d-type%d find tensor%d from layer%d is RECOMPUTE_IN_BACKWARD\n", 
                //             dir, layer->get_base_id(), layer->get_layer_type(), t_in->get_tensor_id(), t_in->get_layer_id());
                if (t_in->get_data_position() == DELETED) {
                    // printf("dir%d: layer%d-type%d find tensor%d is DELETED recompute_layers_stack_size=%d\n", 
                                // dir, layer->get_base_id(), layer->get_layer_type(), t_in->get_tensor_id(), recompute_layers_stack->size());
                    if (recompute_layers_stack->empty() || recompute_layers_stack->top() != t_in->get_layer_id()) {
                        recompute_layers_stack->push(t_in->get_layer_id());
                        // printf("recompute_layers_stack.size() = %d\n", recompute_layers_stack->size());
                    }
                    return 1;
                }
            }
            else if (t_in->get_position() == SHARED_GPU_POOL) {
                // printf("recompute layer%d need swap_tensor%d-layer%d\n", layer->get_base_id(), t_in->get_tensor_id(), t_in->get_layer_id());
                if (alloc_mem_by_swap_block(t_in, BACKWARD) == false) {
                    printf("backward there is not swap_block for tensor%d-layer%d when layer%d-type%d-RECOMPUTE!\n", t_in->get_tensor_id(), t_in->get_layer_id(), layer_id, layer->get_layer_type());
                    exit(1);
                }
            }
        }

    }

	else if (dir == FORWARD) {
        if (layer->get_layer_type() == DATA_L) return 0;
        #ifdef DEBUG
            printf("start forward layer%d-type%d\n", layer->get_base_id(), layer->get_layer_type());
        #endif
        // printSWAPBLOCK("before stash\n");
        if (layer->get_layer_structure() == SISO) {
            if (layer->get_layer_type() == RNN || layer->get_layer_type() == SATTN) {
                // tensor_t<value_type>* reserve_buff = ((rnn_layer_t<value_type>*)layer)->get_rnn_reserve_buff();
                tensor_t<value_type>* reserve_buff = ((base_network_layer_t<value_type>*)layer)->get_reserve_buff();
                // printf("reserve_buff->get_position() = %d\n", reserve_buff->get_data_position());
                if (reserve_buff->get_position() == SHARED_GPU_POOL) {
                    if (alloc_mem_by_swap_block(reserve_buff, FORWARD) == false) {
                        printf("forward there is not swap_block for reserve_buff%d-type%d layer%d-type%d!\n", reserve_buff->get_tensor_id(), reserve_buff->get_type(), layer_id, layer->get_layer_type());
                        printSWAPBLOCK("forward alloc_mem_by_swap_block");
                        // while(1);
                        exit(1);
                    }
                }
                if (layer->get_layer_type() == SATTN) {
                    for (int i = 0; i < 3; i++) {
                        ((self_attn_layer_t<value_type>*)layer)->set_QKV(FORWARD, i, (value_type*)get_OKV_buffer(FORWARD, i));
                    }
                }
            }
            tensor_t<value_type>* t_out = ((base_network_layer_t<value_type>*)layer)->get_f_out();
            // printf("layer%d-type%d\n", layer->get_base_id(), layer->get_layer_type());
            // ((base_network_layer_t<value_type>*)layer)->get_f_in()->printTensorData("stash_tensor", 2);
            #ifdef RECOMPUTE_ON   
            if (t_out->get_position() == RECOMPUTE_IN_BACKWARD) {
                int pool_id = t_out->get_recompute_pool_id();
                t_out->set_gpu_ptr((value_type*)recomputing_pool[pool_id]);
                recomputing_pool_flag[pool_id] = true;
                if (recompute_record[pool_id] != NULL) {
                    recompute_record[pool_id]->set_data_position(DELETED);
                    #ifdef DEBUG
                        printf("dir%d: layer%d-type%d alloc space for tensor%d from recomputing_pool[%d], last tensor%d is %d\n", 
                            dir, layer->get_base_id(), layer->get_layer_type(), t_out->get_tensor_id(), pool_id, recompute_record[pool_id]->get_tensor_id(), recompute_record[pool_id]->get_data_position());
                    #endif
                }
                recompute_record[pool_id] = t_out;
                size_t i;
                // for (i = 0; i < RECOMPUTE_POOL_NUM; i++) {
                //     if (recomputing_pool_flag[i] == false) {
                //         t_out->set_gpu_ptr((value_type*)recomputing_pool[i]);
                //         // t_out->stash_specified_gpu_space(recomputing_pool[i], shared_block_offset);
                //         recomputing_pool_flag[i] = 1;
                //         t_out->set_recompute_pool_id(i);
                //         if (recompute_record[i] != NULL) {
                //             recompute_record[i]->set_data_position(DELETED);  // the last tensor in the recomputing_pool[i] is deleted
                //             printf("dir%d: layer%d-type%d alloc space for tensor%d from recomputing_pool[%d], last tensor%d is %d\n", 
                //                 dir, layer->get_base_id(), layer->get_layer_type(), t_out->get_tensor_id(), i, recompute_record[i]->get_tensor_id(), recompute_record[i]->get_data_position());
                //         }
                //         else {
                //             printf("dir%d: layer%d-type%d alloc space for tensor%d from recomputing_pool[%d]\n", 
                //                 dir, layer->get_base_id(), layer->get_layer_type(), t_out->get_tensor_id(), i);
                //         }
                        
                //         recompute_record[i] = t_out;  // record that the t_out occupies the recomputing_pool[i] 
                //         break;
                //     }
                // } 
                // if ( i == RECOMPUTE_POOL_NUM) {
                //     printf("failed to alloc recomputing_pool for layer%d tensor%d\n", layer_id, t_out->get_tensor_id());
                //     exit(1);
                // }
            }     
            #endif
            if (t_out->get_position() == SHARED_GPU_POOL) {
                #ifdef DEBUG
                    printf("layer%d-type%d t_out-position%d-block%d\n", layer->get_base_id(), layer->get_layer_type(), t_out->get_position(), t_out->get_swap_block_id());
                #endif
                // printSWAPBLOCK("stash tensor");
                if (alloc_mem_by_swap_block(t_out, FORWARD) == false) {
                    printf("forward there is not swap_block for tensor%d layer%d-type%d!\n", t_out->get_tensor_id(), layer_id, layer->get_layer_type());
                    printSWAPBLOCK("forward alloc_mem_by_swap_block");
                    // while(1);
                    exit(1);
                }
                // printSWAPBLOCK("forward stash tensor done");
            }
        }
        else if (layer->get_layer_structure() == MIMO) {
            std::vector<tensor_t<value_type>*> t_outs = ((base_structure_t<value_type>*)layer)->get_outputs();
            for (int i = 0; i < t_outs.size(); i++) {
                // printSWAPBLOCK("stash tensor");
                if (t_outs[i]->get_position() == SHARED_GPU_POOL) {
                    #ifdef DEBUG
                        printf("layer%d-type%d t_outs[%d]-position%d-block%d\n", layer->get_base_id(), layer->get_layer_type(), i, t_outs[i]->get_position(), t_outs[i]->get_swap_block_id());
                    #endif
                    if (alloc_mem_by_swap_block(t_outs[i], FORWARD) == false) {
                        printf("forward there is not swap_block for tensor%d layer%d-type%d!\n", t_outs[i]->get_tensor_id(), layer_id, layer->get_layer_type());
                        printSWAPBLOCK("forward alloc_mem_by_swap_block");
                        exit(1);
                    }
                    #ifdef DEBUG
                        printSWAPBLOCK("");
                    #endif
                }
            }
        }
        #ifdef DEBUG
            printSWAPBLOCK("forward stash tensor done");
        #endif
	}
	else if (dir == BACKWARD) {  // BACKWARD   
        #ifdef DEBUG
            printf("BACKWARD: layer%d-type%d stash tensor start:\n", layer->get_base_id(), layer->get_layer_type());
            printSWAPBLOCK("BACKWARD stash tensor");
        #endif
		if (layer->get_layer_structure() == MIMO) { // alloc space to b_data from b_data_pool
            std::vector<tensor_t<value_type>* > b_datas = ((base_structure_t<value_type>*)layer)->get_b_data();
            int b_count = 0;
            for(size_t i = 0; i < b_datas.size(); i++) {
                // printf("layer%d-type%d should allocate memory to b_data%d\n", layer->get_base_id(), layer->get_layer_type(), b_datas[i]->get_tensor_id());
                for (int j = 0; j < B_DATA_POOL_NUM; j++) {
                    // if (b_data_pool_flag[j] == 0) {
                    if (b_data_pool_tensors[j] == NULL) {
                        // printf("layer%d-type%d allocate b_data_pool[%d] to b_data%d\n", layer->get_base_id(), layer->get_layer_type(), j, b_datas[i]->get_tensor_id());
                        b_datas[i]->set_gpu_ptr((value_type*)b_data_pool_ptr[j]);
                        b_datas[i]->set_b_data_pool_id(j);
                        b_data_pool_tensors[j] = b_datas[i];
                        b_datas[i]->set_data_state(STILL_USED);
                        b_datas[i]->set_data_position(IN_GPU);
                        b_count++;
                        break;
                    }
                    else if (b_data_pool_tensors[j]->get_data_state() != NO_COMPUTED) {
                        // printf("layer%d-type%d allocate b_data_pool[%d] to b_data%d\n", layer->get_base_id(), layer->get_layer_type(), j, b_datas[i]->get_tensor_id());
                        b_datas[i]->set_gpu_ptr((value_type*)b_data_pool_ptr[j]);
                        b_datas[i]->set_b_data_pool_id(j);
                        b_data_pool_tensors[j] = b_datas[i];
                        b_datas[i]->set_data_state(STILL_USED);
                        b_datas[i]->set_data_position(IN_GPU);
                        // b_data_pool_flag[j] = 1;
                        b_count++;
                        break;
                    }
                    // printf("layer%d-type%d allocate b_data_pool[%d] to b_data%d\n", layer->get_base_id(), layer->get_layer_type(), j, b_datas[i]->get_tensor_id());    
                }
            }
            if (b_count != b_datas.size()) {
                printf("layer%d allocating mem to b_datas error\n", layer->get_base_id());
                exit(1);
            } 
            std::vector<tensor_t<value_type>*> t_ins = ((base_structure_t<value_type>*)layer)->get_inputs();
            // printf("BACKWARD: layer%d-type%d stash tensor%d\n", layer->get_base_id(), layer->get_layer_type(), t_ins[0]->get_tensor_id());

            if (layer->get_layer_type() == JOIN_L || layer->get_layer_type() == FORK_L || layer->get_layer_type() == CONCAT) {
                // these three kinds of layer doesn't need t_in
                return 0;
            }

            #ifdef RECOMPUTE_ON
            bool recompute_flag = false;
            // for (int i = 0; i < t_ins.size(); i++) {
            for (int i = t_ins.size()-1; i >= 0; i--) {
                if (t_ins[i]->get_position() == RECOMPUTE_IN_BACKWARD) {
                    #ifdef DEBUG
                        printf("dir%d: layer%d-type%d find tensor%d from layer%d is RECOMPUTE_IN_BACKWARD\n", 
                            dir, layer->get_base_id(), layer->get_layer_type(), t_ins[i]->get_tensor_id(), t_ins[i]->get_layer_id());
                    #endif
                    if (t_ins[i]->get_data_position() == DELETED) {
                        // printf("dir%d: layer%d-type%d find tensor%d is DELETED recompute_layers_stack_size=%d\n", 
                        //     dir, layer->get_base_id(), layer->get_layer_type(), t_ins[i]->get_tensor_id(), recompute_layers_stack->size());
                        if (recompute_layers_stack->empty() || recompute_layers_stack->top() != t_ins[i]->get_layer_id()) {
                            recompute_layers_stack->push(t_ins[i]->get_layer_id());
                        }
                        recompute_flag = true;
                    }
                }
            }
            if (recompute_flag) return 1;
       
            #endif
            #ifdef MULTI_SWAP_BLOCK
            // for (int i = 0; i < t_ins.size(); i++) { 
            for (int i = t_ins.size()-1; i >= 0; i--) {
                if (t_ins[i]->get_position() == SHARED_GPU_POOL) {
                    #ifdef DEBUG
                        printf("layer%d t_ins[%d]tensor%d->get_position() = %d state=%d\n", layer_id, i, t_ins[i]->get_tensor_id(), t_ins[i]->get_position(), t_ins[i]->get_data_state());
                    #endif
                    if (alloc_mem_by_swap_block(t_ins[i], BACKWARD) == false) {
                        printf("backward there is not swap_block for tensor%d-layer%d when layer%d-type%d-BACKWARD!\n", t_ins[i]->get_tensor_id(), t_ins[i]->get_layer_id(), layer_id, layer->get_layer_type());
                        exit(1);
                    }
                }
            } 
            #endif
        }
		else {  // SISO
            // alloc space to b_data from b_data_pool
            // if (layer->get_layer_type() != SOFTMAX && layer->get_layer_type() != DATA_L) {
            if (layer->get_layer_type() != DATA_L) {
                tensor_t<value_type>* b_data = ((base_network_layer_t<value_type>*)layer)->get_b_data();
                // printf("layer%d-type%d should alloc memory to b_data%d\n", layer->get_base_id(), layer->get_layer_type(), b_data->get_tensor_id());
                int b_count = 0;
                for (int i = 0; i < B_DATA_POOL_NUM; i++) {
                    if (b_data_pool_tensors[i] == NULL) {
                        // printf("layer%d-type%d allocate b_data_pool[%d] to b_data%d\n", layer->get_base_id(), layer->get_layer_type(), i, b_data->get_tensor_id());
                        b_data->set_gpu_ptr((value_type*)b_data_pool_ptr[i]);
                        b_data->set_b_data_pool_id(i);
                        b_data_pool_tensors[i] = b_data;
                        b_data->set_data_state(STILL_USED);
                        b_data->set_data_position(IN_GPU);
                        // b_data_pool_flag[j] = 1;
                        b_count = 1;
                        break;
                    }
                    else if (b_data_pool_tensors[i]->get_data_state() == NO_COMPUTED) {
                        // printf("layer%d-type%d allocate b_data_pool[%d] to b_data%d, b_data_pool_tensors[%d]%d-state%d\n", layer->get_base_id(), layer->get_layer_type(), i, b_data->get_tensor_id(), i, b_data_pool_tensors[i]->get_tensor_id(), b_data_pool_tensors[i]->get_data_state());
                        b_data->set_gpu_ptr((value_type*)b_data_pool_ptr[i]);
                        b_data->set_b_data_pool_id(i);
                        b_data_pool_tensors[i] = b_data;
                        b_data->set_data_state(STILL_USED);
                        b_data->set_data_position(IN_GPU);
                        // b_data_pool_flag[j] = 1;
                        b_count = 1;
                        break;
                    }

                    // printf("layer%d-type%d allocate b_data_pool[%d] to b_data%d\n", layer->get_base_id(), layer->get_layer_type(), i, b_data->get_tensor_id());
                }
                if (b_count != 1) {
                    printf("layer%d allocating mem to b_data error\n", layer->get_base_id());
                    exit(1);
                }
                if (layer->get_layer_type() == RNN || layer->get_layer_type() == SATTN) {
                    // tensor_t<value_type>* reserve_buff = ((rnn_layer_t<value_type>*)layer)->get_rnn_reserve_buff();
                    tensor_t<value_type>* reserve_buff = ((base_network_layer_t<value_type>*)layer)->get_reserve_buff();
                    // printf("reserve_buff->get_position() = %d\n", reserve_buff->get_position());
                    #ifdef MULTI_SWAP_BLOCK
                    if (reserve_buff->get_position() == SHARED_GPU_POOL) {
                        if (alloc_mem_by_swap_block(reserve_buff, BACKWARD) == false) {
                            printf("backward there is not swap_block for reserve_buff%d-type%d layer%d when layer%d-type%d-backward!\n", 
                                reserve_buff->get_tensor_id(), reserve_buff->get_type(), layer->get_base_id(), layer->get_layer_type());
                            exit(1);
                        }
                    }
                    #endif
                    if (layer->get_layer_type() == SATTN) {
                        for (int i = 0; i < 3; i++) {
                            ((self_attn_layer_t<value_type>*)layer)->set_QKV(FORWARD, i, (value_type*)get_OKV_buffer(FORWARD, i));
                        }
                        for (int i = 0; i < 3; i++) {
                            ((self_attn_layer_t<value_type>*)layer)->set_QKV(BACKWARD, i, (value_type*)get_OKV_buffer(BACKWARD, i));
                        }
                    }
                }
                // printf("befor t_in %d\n", 666);
			    tensor_t<value_type>* t_in = ((base_network_layer_t<value_type>*)layer)->get_f_in();
                // printf("after t_in%d t_in->get_data_position = %d, t_in->get_gpu_ptr=%x\n", t_in->get_tensor_id(), t_in->get_data_position(), t_in->get_gpu_ptr());
                // t_in->printTensorData("t_in = ", 2);
            #ifdef RECOMPUTE_ON
                if (t_in->get_position() == RECOMPUTE_IN_BACKWARD) {
                    // printf("dir%d: layer%d-type%d find tensor%d from layer%d is RECOMPUTE_IN_BACKWARD\n", 
                    //     dir, layer->get_base_id(), layer->get_layer_type(), t_in->get_tensor_id(), t_in->get_layer_id());
                    if (t_in->get_data_position() == DELETED) {
                        base_layer_t<value_type>* last_layer = (base_layer_t<value_type>*) net_layers->find(t_in->get_layer_id())->second;
                        if (last_layer->get_layer_type() == FORK_L) {  
                            // FORK_L input = output, go to last_layer of FORK_L to recompute
                            last_layer = last_layer->get_prev()[0];
                            if (recompute_layers_stack->empty() || recompute_layers_stack->top() != last_layer->get_base_id()) {
                                recompute_layers_stack->push(last_layer->get_base_id());
                            }
                            return 1;
                        }
                        else {
                            if (recompute_layers_stack->empty() || recompute_layers_stack->top() != t_in->get_layer_id()) {
                                recompute_layers_stack->push(t_in->get_layer_id());
                            }
                            return 1;
                        }
                        // printf("dir%d: layer%d-type%d find tensor%d is DELETED recompute_layers_stack_size=%d\n", 
                        //     dir, layer->get_base_id(), layer->get_layer_type(), t_in->get_tensor_id(), recompute_layers_stack->size());
                        
                    }
                }
            #endif
                if (t_in->get_position() == SHARED_GPU_POOL) {
                    // printf("BACKWARD: layer%d-type%d stash tensor%d\n", layer->get_base_id(), layer->get_layer_type(), t_in->get_tensor_id());
                    // printf("t_in->get_position() = %d\n", t_in->get_position());
                #ifdef MULTI_SWAP_BLOCK
                    if (alloc_mem_by_swap_block(t_in, BACKWARD) == false) {
                        printf("backward there is not swap_block for tensor%d-layer%d when layer%d-type%d-backward!\n", t_in->get_tensor_id(), t_in->get_layer_id(), layer_id, layer->get_layer_type());
                        exit(1);
                    }
                #endif 
                }
            }
		}
        #ifdef DEBUG
            printf("BACKWARD: layer%d-type%d stash tensor done\n", layer->get_base_id(), layer->get_layer_type());
            printSWAPBLOCK("BACKWARD stash tensor done");
        #endif
	}
    return 0;

    #ifdef DEBUG
    auto tmp = reg->get_net_layers().find(layer_id);
    #endif
}
// #undef DEBUG

// #define DEBUG
template <class value_type>
void mem_controller_t<value_type>::update_tensor_state(int layer_id, base_layer_t<value_type>* layer, net_comp dir, network_stage stage) {
	tensor_t<value_type>* t2;
	LAYER aa = layer->get_layer_type();
	LAYER_STRUCTURE bb = layer->get_layer_structure();
	std::vector<base_layer_t<value_type>*> next_layers;
    std::map<int, void* >* net_layers  = reg->get_net_layers_ptr();
    
    if (dir == RECOMPUTE) {
        tensor_t<value_type>* t_in = ((base_network_layer_t<value_type>*)layer)->get_f_in();
        t_in->reset_cur_use_counter(FORWARD);
    }
	else if (dir == FORWARD) {
        // layer->increase_input_cur_use_counter(FORWARD);
        layer->update_input_state(FORWARD);
		if (layer->get_layer_structure() == SISO) {
            if (layer->get_layer_type() != DATA_L) {
                tensor_t<value_type>* t_in  = ((base_network_layer_t<value_type>*)layer)->get_f_in();
                // if (t_in->is_cur_use_count_zero()) {
                //     // 该层输入已经在fwd中用完了
                //     t_in->set_data_state(FORWARD_DELETE_OK);
                // }
            #ifdef RECOMPUTE_ON
                if (t_in->get_position() == RECOMPUTE_IN_BACKWARD) {
                    // printf("layer%d-type%d forward finished using tensor%d-layer%d as t_in, tensor%d-state%d\n", 
                    //     layer->get_base_id(), layer->get_layer_type(), t_in->get_tensor_id(), t_in->get_layer_id(), t_in->get_tensor_id(), t_in->get_data_state());
                    if (t_in->get_data_state() == FORWARD_DELETE_OK) {
                        recomputing_pool_flag[t_in->get_recompute_pool_id()] = 0;
                    }
                }
            #endif
            }
            if (layer->get_layer_type() != SOFTMAX) {
                tensor_t<value_type>* t_out = ((base_network_layer_t<value_type>*)layer)->get_f_out();
                t_out->set_data_position(IN_GPU);  // 更新状态
                if (layer->get_layer_type() == RNN  || layer->get_layer_type() == SATTN) {
                    // tensor_t<value_type>* reserve_buff = ((rnn_layer_t<value_type>*)layer)->get_rnn_reserve_buff(); 
                    tensor_t<value_type>* reserve_buff = ((base_network_layer_t<value_type>*)layer)->get_reserve_buff();
                    reserve_buff->set_data_position(IN_GPU);
                    reserve_buff->set_data_state(FORWARD_DELETE_OK);
                }
            }
		}
		else {  // MIMO
			std::vector<tensor_t<value_type>*>* t_ins = ((base_structure_t<value_type>*)layer)->get_inputs_ptr();
			for (int i = 0; i < t_ins->size(); i++) {
				// if((*t_ins)[i]->is_cur_use_count_zero()) {
				// 	(*t_ins)[i]->set_data_state(FORWARD_DELETE_OK);
				// }
			}
            #ifdef RECOMPUTE_ON
            for (int i = 0; i < t_ins->size(); i++) {
                if ((*t_ins)[i]->get_position() == RECOMPUTE_IN_BACKWARD) {
                    // printf("layer%d-type%d forward finished using tensor%d-layer%d as t_in, tensor%d-state%d\n", 
                    //     layer->get_base_id(), layer->get_layer_type(), (*t_ins)[i]->get_tensor_id(), (*t_ins)[i]->get_layer_id(), (*t_ins)[i]->get_tensor_id(), (*t_ins)[i]->get_data_state());
                    if ((*t_ins)[i]->get_data_state() == FORWARD_DELETE_OK) {
                        recomputing_pool_flag[(*t_ins)[i]->get_recompute_pool_id()] = 0;
                    }
                }
                
            }
            #endif 
            std::vector<tensor_t<value_type>* > t_outs = ((base_structure_t<value_type>*)layer)->get_outputs();
            for (int i = 0; i < t_outs.size(); i++) {
                t_outs[i]->set_data_position(IN_GPU);
            }   
		}
	}
	else if (dir == BACKWARD) {  // BACKWARD
        // layer->increase_input_cur_use_counter(BACKWARD);
        
        layer->increase_dy_cur_use_counter();
        layer->update_dy_state();
        layer->update_input_state(BACKWARD);
        layer->update_output_state(BACKWARD);
        // #ifdef DEBUG
        // int test_layer_id = 1877;
        // base_layer_t<value_type>* test_layer = (base_layer_t<value_type>*) net_layers->find(test_layer_id)->second;
        // tensor_t<value_type>* test_tensor = ((base_network_layer_t<value_type>*)test_layer)->get_f_out();
        // test_tensor->printTensorData("test_tensor update_tensor_state", 2);
        // test_tensor->printTensorData("test_tensor update_tensor_state", 2);
        // test_tensor->printTensorData("test_tensor update_tensor_state", 2);
        // #endif
        // printf("backward layer update tensor state%d \n", layer_id);
		if (layer->get_layer_structure() == SISO) {
            // printf("backward layer%d: ", layer_id);
           
            if (layer->get_layer_type() == RNN  || layer->get_layer_type() == SATTN) {
                // tensor_t<value_type>* reserve_buff = ((rnn_layer_t<value_type>*)layer)->get_rnn_reserve_buff(); 
                tensor_t<value_type>* reserve_buff = ((base_network_layer_t<value_type>*)layer)->get_reserve_buff();
                reserve_buff->set_data_position(NO_DATA);
                reserve_buff->set_data_state(NO_COMPUTED);
            }
            if (layer->get_layer_type() != DATA_L) {
                tensor_t<value_type>* t_in = ((base_network_layer_t<value_type>*)layer)->get_f_in();
                #ifdef DEBUG
                    t_in->print_cur_use_counter(BACKWARD);
                #endif
                // printf("layer%d backward finished using tensor%d-layer%d as t_in, tensor%d-state%d-data_position%d\n", 
                //         layer->get_base_id(), t_in->get_tensor_id(), t_in->get_layer_id(), t_in->get_tensor_id(), t_in->get_data_state(), t_in->get_data_position());
                #ifdef RECOMPUTE_ON
                // need to correct
                if (t_in->get_position() == RECOMPUTE_IN_BACKWARD) {
                    // printf("layer%d backward finished using tensor%d-layer%d as t_in, tensor%d-state%d\n", 
                    //     layer->get_base_id(), t_in->get_tensor_id(), t_in->get_layer_id(), t_in->get_tensor_id(), t_in->get_data_state());
                    if (t_in->get_data_state() == NO_COMPUTED) {
                        recomputing_pool_flag[t_in->get_recompute_pool_id()] = 0;
                    }
                }
                #endif
            }
            else { // DATA_L
                for (int i = 0; i < RECOMPUTE_POOL_NUM; i++) {
                    recomputing_pool_flag[i] = false;
                    recompute_record[i] = NULL;
                }
            }
		}
		else {  // MIMO layer
            std::vector<tensor_t<value_type>*> t_ins = ((base_structure_t<value_type>*)layer)->get_inputs();
            // printf("backward layer%d: ", layer_id);
        #ifdef RECOMPUTE_ON
            for (int i = 0; i < t_ins.size(); i++) {
                #ifdef DEBUG
                    t_ins[i]->print_cur_use_counter(BACKWARD);
                #endif
                // printf("layer%d backward finished using tensor%d-layer%d as t_in, tensor%d-state%d-data_position%d\n", 
                        // layer->get_base_id(), t_ins[i]->get_tensor_id(), t_ins[i]->get_layer_id(), t_ins[i]->get_tensor_id(), t_ins[i]->get_data_state(), t_ins[i]->get_data_position());
                if (t_ins[i]->get_position() == RECOMPUTE_IN_BACKWARD) {
                    if (t_ins[i]->get_data_state() == NO_COMPUTED) {
                        recomputing_pool_flag[t_ins[i]->get_recompute_pool_id()] = 0;
                    }
                }    
            }
        #endif

            if (layer->get_layer_type() == JOIN_L || layer->get_layer_type() == FORK_L || layer->get_layer_type() == CONCAT) {
                // if (t_ins[i]->get_position() == RECOMPUTE_IN_BACKWARD) {
                //     if (t_ins[i]->get_data_state() == FORWARD_DELETE_OK) {
                //         recomputing_pool_flag[t_ins[i]->get_recompute_pool_id()] = 0;
                //     }
                // }
                return;
            }

            else {
            }
		}
        // printSWAPBLOCK("update_tensor_state done");
	}

    // #ifdef BENCHMARK
    // // collect conv buff size
    // auto it = reg->get_net_layers().find(layer_id);
    // if (it != reg->get_net_layers().end()) {
    //     base_layer_t<value_type>* l = (base_layer_t<value_type>*)it->second;
    //     if (l->get_layer_type() == CONV) {
			
    //     }
    // }
    // #endif

    #ifdef LIVENESS
    // live_anls->update(layer_id, dir);
    #endif
    // #ifdef RECOMPUTE_ON
    //     recomp->offload_to_recompute(layer_id, dir, stage);
    // #endif
    
}
// #undef DEBUG

/*--------control--end--------------------*/


// 该函数设置“管理tensor”，将每层的数据tensor和CONV_BUFF加入regulated_tensors
template <class value_type>
void mem_controller_t<value_type>::set_regulated_tensors() {
    std::map<int, std::vector< tensor_t<value_type>* > > tensor_by_layer = reg->get_tensor_by_layer();
    typename std::map<int, std::vector< tensor_t<value_type>* > >::iterator it = tensor_by_layer.begin();
    for (it = tensor_by_layer.begin(); it != tensor_by_layer.end(); ++it) {
        std::vector<tensor_t<value_type>* > tensors = it->second;
        for(size_t i = 0; i < tensors.size(); i++) {
            if(tensors[i]->get_type() != DATA && tensors[i]->get_type() != CONV_BUFF ) {
				// 跳过数据tensor和Conv_BUFF的tensor（应该是指workspace）
                continue;
            }
            typename std::map<tensor_t<value_type>*, mem_mode>::iterator it2 = regulated_tensors.find(tensors[i]);
            if (it2 == regulated_tensors.end()) {
                regulated_tensors.insert( std::make_pair(tensors[i], VOID) );
            }
        }
    }
}

/*--------print helper start------------*/

template <class value_type>
void mem_controller_t<value_type>::print_required_tensor(int layer_id, net_comp dir) {
    std::vector<tensor_t<value_type>* >* tensors = NULL;
    if(dir == FORWARD) {
        tensors = reg->get_forward_dependency(layer_id); // TO DO, we don't track the data layer!!
    } else if(dir == BACKWARD) {
        tensors = reg->get_backward_dependency(layer_id);
    }
    if(tensors == NULL) return;
    for(size_t i = 0; i < tensors->size(); i++) {
        tensor_t<value_type>* t = tensors->operator[](i);
        printf("TENSOR NEEDED layer %d : %p state:%d type:%d gpu_ptr:%p\n",
               t->get_layer_id(), t, t->get_state(), t->get_type(), t->get_gpu_ptr() );
    }
}

template <class value_type>
void mem_controller_t<value_type>::print_regulated_tensors(bool log, int layer_id) {
    if (log) {
        int total=0, hit=0, miss=0;
        typename std::map<tensor_t<value_type> *, mem_mode>::iterator it = regulated_tensors.begin();
        for (it = regulated_tensors.begin(); it != regulated_tensors.end(); it++) {
            if (it->first->get_type() == CONV_BUFF || it->first->into_cnt == 0) {
                continue;
            }
            printf("layer %d tensor %p type %d: total=%d, hit=%d, miss=%d\n",
                   it->first->get_layer_id(), it->first, it->first->get_type(),
                   it->first->into_cnt, it->first->hit_cnt, it->first->miss_cnt);
            total += it->first->into_cnt;
            hit += it->first->hit_cnt;
            miss += it->first->miss_cnt;
        }
        printf("Summary total=%d, hit=%d, miss=%d\n", total, hit, miss);
        printf("hit / total = %f\n", (double)hit / (double) total);

        size_t conv_buff = 0;
        for (it = regulated_tensors.begin(); it != regulated_tensors.end(); it++) {
            if (it->first->get_type() == CONV_BUFF || it->first->into_cnt == 0) {
                conv_buff += it->first->get_mem_size();
                printf("layer %d  conv buff %zu  %.3fMB\n",
                       it->first->get_layer_id(), it->first->get_mem_size(), (double)(it->first->get_mem_size())/1024.0/1024.0);
            }
        }
        printf("conv buff = %zu  %.3fMB\n", conv_buff, (double)conv_buff/1024.0/1024.0);

    }

#ifdef DEBUG

    int  data_c          = 0;
    int  conv_buff_c     = 0;

    long total_data_type = 0;
    long total_data_type_gpu = 0;

    long total_conv_buff = 0;
    long total_conv_buff_gpu = 0;

    long total_gpu_data  = 0;
    long total_cpu_data  = 0;
    int total_live_cnt   = 0;
    int ff = 0;
    int bb = 0;

    typename std::map<tensor_t<value_type>*, mem_mode>::iterator it = regulated_tensors.begin();
    for( it = regulated_tensors.begin(); it != regulated_tensors.end(); it++ ) {
        tensor_t<value_type>* t = it->first;
        // or t->get_state() == CPU2GPU or t->get_state() == GPU2CPU
        // do not accumulate other state
        if ( t->get_gpu_ptr() != NULL ) {
            total_gpu_data += t->get_scalar_count();

            if (t->get_type() == DATA) {
                total_live_cnt += 1;

                bool flag = true;
                for (int layer = 3; layer <= max_layer_id; ++layer) {
                    std::vector<tensor_t<value_type> *> *f_tensors = reg->get_forward_dependency(layer);
                    for (auto tt = f_tensors->begin(); tt != f_tensors->end(); ++tt) {
                        if (t == (*tt)) {
                            ff += 1;
                            flag = false;
                            break;
                        }
                    }
                    if (!flag) {
                        break;
                    }
                }

                flag = true;
                for (int layer = 3; layer <= max_layer_id; ++layer) {
                    std::vector<tensor_t<value_type> *> *b_tensors = reg->get_backward_dependency(layer);
                    for (auto tt = b_tensors->begin(); tt != b_tensors->end(); ++tt) {
                        if (t == (*tt)) {
                            bb += 1;
                            flag = false;
                            break;
                        }
                    }
                    if (!flag) {
                        break;
                    }
                }
            }


        } else if( t->get_gpu_ptr() == NULL ) {
            total_cpu_data += t->get_scalar_count();
        }

        double tensor_size = t->get_mem_size()/1024.0f/1024.0f;
        if(t->get_type() == DATA) {
            data_c += 1;
            if (t->get_gpu_ptr() != NULL) {
                printf("size:%5.2fMB mem_mode:%2d layer:%d DATA tensor:%p gpu_ptr:%p\n", tensor_size, t->get_state(),
                       t->get_layer_id(), t, t->get_gpu_ptr());
                total_data_type_gpu += ((long) t->get_scalar_count());
            }
            total_data_type += ((long) t->get_scalar_count());
        } else if(t->get_type() == CONV_BUFF) {
            conv_buff_c += 1;
            if (t->get_gpu_ptr() != NULL) {
                printf("size:%5.2fMB mem_mode:%2d layer:%d BUFF tensor:%p gpu_ptr:%p\n", tensor_size, t->get_state(),
                       t->get_layer_id(), t, t->get_gpu_ptr());

                total_conv_buff_gpu += ((long) t->get_scalar_count());
            }
            total_conv_buff += ((long) t->get_scalar_count());
        } else {
            printf("ERROR UNTRACKED tensor:%p size:%5.2ffMB mem_mode:%d gpu_ptr:%p\n", t, tensor_size, t->get_state(), t->get_gpu_ptr());
        }
    }
    double total_data_type_mem   = total_data_type*sizeof(value_type)/1024.0f/1024.0f;
    double total_data_gpu_mem    = total_data_type_gpu* sizeof(value_type) / 1024.0f / 1024.0f;

    double total_conv_buff_mem   = total_conv_buff*sizeof(value_type)/1024.0f/1024.0f;
    double total_conv_buff_gpu_mem = total_conv_buff_gpu* sizeof(value_type) / 1024.0f / 1024.0f;

    double total_gpu_data_mem   = total_gpu_data*sizeof(value_type)/1024.0f/1024.0f;
    double total_cpu_data_mem    = total_cpu_data*sizeof(value_type)/1024.0f/1024.0f;

    printf("TOTAL CPU TENSOR:%8.2fMB, TOTAL GPU TENSOR:    %8.2fMB\n",
           total_cpu_data_mem, total_gpu_data_mem);
    printf("TOTAL DATA:      %8.2fMB, TOTAL GPU DATA:      %8.2fMB\n",
           total_data_type_mem, total_data_gpu_mem);
    printf("TOTAL CONV_BUFF: %8.2fMB, TOTAL GPU CONV_BUFF: %8.2fMB\n",
            total_conv_buff_mem, total_conv_buff_gpu_mem);

#ifdef LRU_ON
    lru_singleton::get_lru()->print_list();
#endif

    printf("free gpu memory : %f MB\n", BYTE_TO_MB(query_free_mem()));


#endif

}

INSTANTIATE_CLASS(mem_controller_t);

} //SuperNeuron namespace
