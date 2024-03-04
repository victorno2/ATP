#include <registry.h>
#include <layer/base_network_layer.h>

namespace ATP{
	
template <class value_type>
size_t registry_t<value_type>::get_total_undata_size_pool_malloc_mode() {
    // 从显存池中分配，只考虑512bytes对齐
    std::vector<tensor_t<value_type>* >* all_tensors = this->get_vector();
    size_t align_size = 512;
    size_t total_size = 0;
    size_t tensor_num = 0;
    size_t temp_size;
    for (size_t i = 0; i < all_tensors->size(); i++) {
        if (((*all_tensors)[i])->get_type() != DATA) {
            tensor_num++;
            temp_size = ((*all_tensors)[i])->get_mem_size();
        }
    }
#ifdef FAKE_TRAIN
    total_size += ((*all_tensors)[0])->get_mem_size() + ((*all_tensors)[1])->get_mem_size();
#endif
    printf("get_total_undata_size_pool_malloc_mode = %zdbytes = %fMB\n", total_size, BYTE_TO_MB(total_size));
    return total_size;
}

template <class value_type>
void registry_t<value_type>::get_size_info(size_t* net_total_size, size_t* gpu_tensors_size, size_t* gpu_pool_size, 
                                            size_t* swap_tensors_size, size_t* swap_pool_size, size_t* recompute_tensors_size, size_t* recompute_pool_size, size_t* max_fragment_size,
                                            size_t* max_grad_size, size_t* max_tensor_size, size_t* max_data_size, size_t* max_layer_size,
                                            size_t* b_data_pool_size, size_t* QKV_buffer_size, size_t* max_workspace_size)
{
    // 从显存池中分配，只考虑512bytes对齐
    size_t _net_total_size = 0;
    size_t _gpu_tensors_size = 0;
    size_t _gpu_pool_size = 0;
    size_t _swap_tensors_size = 0;
    size_t _recompute_tensor_size = 0;
    size_t _swap_pool_size = 0;
    size_t _recompute_pool_size = 0;
    size_t _max_grad_size = 0;
    size_t _b_data_pool_size = 0;
    size_t _max_b_data_size = 0;
    size_t _QKV_buffer_size = 0;
    size_t _max_QKV_size = 0;
    std::vector<tensor_t<value_type>* >* all_tensors = this->get_vector();
    size_t align_size = 512; 
    size_t tensor_num = 0;
    size_t temp_size;
    size_t _max_tensor_size = 0;
    size_t _max_data_size = 0;
    size_t _max_workspace_size = 0;
    size_t layer_temp = 0;
    size_t _max_layer_size = 0;
    size_t swappable_size = 0;
    size_t _min_tensor_size = 10737418240;
    size_t _max_fragment_size = 0;
    size_t data_sum = 0;
	for (auto it = this->net_layers.begin(); it != this->net_layers.end(); it++) {
        base_layer_t<value_type>* layer = (base_layer_t<value_type>*)it->second;
        std::vector<tensor_t<value_type>* >* f_deps = this->get_forward_dependency(layer->get_base_id());
        std::vector<tensor_t<value_type>* >* b_deps = this->get_backward_dependency(layer->get_base_id());
        // printf("1 layer%d-type%d\n", layer->get_base_id(), layer->get_layer_type());
        if (f_deps != NULL) {
            for (int i = 0; i < f_deps->size(); i++) {
                layer_temp += (*f_deps)[i]->get_mem_size();
            }
            if (_max_layer_size < layer_temp) {
                _max_layer_size = layer_temp;
                layer_temp = 0;
            }
        }
        // printf("2 layer%d-type%d\n", layer->get_base_id(), layer->get_layer_type());
        if (b_deps != NULL) {
            for (int i = 0; i < b_deps->size(); i++) {
                layer_temp += (*b_deps)[i]->get_mem_size();
            }
            if (_max_layer_size < layer_temp) {
                _max_layer_size = layer_temp;
                layer_temp = 0;
            }
        }
        // printf("3 layer%d-type%d\n", layer->get_base_id(), layer->get_layer_type());
        if (layer->get_layer_structure() == SISO) {
            tensor_t<value_type>* tensor = ((base_network_layer_t<value_type>*)layer)->get_weight_grad();
            temp_size = 0;
            if (tensor != NULL) {
                temp_size += tensor->get_mem_size();
            }
            tensor = ((base_network_layer_t<value_type>*)layer)->get_bias_grad();
            if (tensor != NULL) {
                temp_size += tensor->get_mem_size();
            }
            if (_max_grad_size < temp_size) {
                _max_grad_size = temp_size;
            }
        }
    }
    
    *max_layer_size = _max_layer_size;
    printf("after _max_layer_size = %d\n", *max_layer_size);
    for (size_t i = 0; i < all_tensors->size(); i++) {
        tensor_num++;
        if ((*all_tensors)[i]->get_type() != B_DATA && (*all_tensors)[i]->get_type() != CONV_BUFF && (*all_tensors)[i]->get_type() != RNN_BUFF) {
            _net_total_size += ((*all_tensors)[i])->get_mem_size();
        }
        // if (((*all_tensors)[i])->get_position() == REMAIN_IN_GPU) {  // REMAIN_IN_GPU tensors
        // if (((*all_tensors)[i])->get_position() == REMAIN_IN_GPU && (*all_tensors)[i]->get_type() != B_DATA) { 
        if (((*all_tensors)[i])->get_position() == REMAIN_IN_GPU && (*all_tensors)[i]->get_type() != B_DATA
            && (*all_tensors)[i]->get_type() != CONV_BUFF && (*all_tensors)[i]->get_type() != RNN_BUFF ) 
        {
            _gpu_tensors_size += ((*all_tensors)[i])->get_mem_size();
        }
        else {  // swap or recompute tensors
            if ((*all_tensors)[i]->get_position() == SHARED_GPU_POOL) {
                _swap_tensors_size += ((*all_tensors)[i])->get_mem_size();
            }  
            if ((*all_tensors)[i]->get_position() == RECOMPUTE_IN_BACKWARD) {
                _recompute_tensor_size += ((*all_tensors)[i])->get_mem_size();
            }
        }
        // if ((*all_tensors)[i]->get_type() == B_DATA) {
        //     _net_total_size -= ((*all_tensors)[i])->get_mem_size();  // B_DATA is in b_data_pool
        // }
        if (((*all_tensors)[i])->get_type() == DATA || ((*all_tensors)[i])->get_type() == RNN_RESERVE) {
            if (_max_tensor_size < ((*all_tensors)[i])->get_mem_size()) {
                _max_tensor_size = ((*all_tensors)[i])->get_mem_size();
            }
            if (_min_tensor_size > ((*all_tensors)[i])->get_mem_size()) {
                _min_tensor_size = ((*all_tensors)[i])->get_mem_size();
            }
        }
        if ((*all_tensors)[i]->get_type() == DATA) {
            data_sum += ((*all_tensors)[i])->get_mem_size();
            if(_max_data_size < ((*all_tensors)[i])->get_mem_size()) {
                _max_data_size = ((*all_tensors)[i])->get_mem_size();
            }
        }
        if ((*all_tensors)[i]->get_type() == CONV_BUFF || (*all_tensors)[i]->get_type() == RNN_BUFF) {
            if (_max_workspace_size < (*all_tensors)[i]->get_mem_size()) {
                _max_workspace_size = (*all_tensors)[i]->get_mem_size();
                // printf("tensor%d-type%d-layer%d _max_workspace_size = %zd\n", (*all_tensors)[i]->get_tensor_id(), (*all_tensors)[i]->get_type(), (*all_tensors)[i]->get_layer_id(), _max_workspace_size);
            }
        }
        if ((*all_tensors)[i]->get_type() == QKV_DATA || (*all_tensors)[i]->get_type() == DQKV_DATA) {
            if (_max_QKV_size < (*all_tensors)[i]->get_mem_size()) {
                _max_QKV_size = (*all_tensors)[i]->get_mem_size();
                // printf("tensor%d-type%d-layer%d _max_QKV_size = %zd\n", (*all_tensors)[i]->get_tensor_id(), (*all_tensors)[i]->get_type(), (*all_tensors)[i]->get_layer_id(), _max_workspace_size);
            }
        }
        if ((*all_tensors)[i]->get_type() == B_DATA) {
            if (_max_b_data_size < (*all_tensors)[i]->get_mem_size()) {
                _max_b_data_size = (*all_tensors)[i]->get_mem_size();
                // printf("tensor%d-type%d-layer%d _max_QKV_size = %zd\n", (*all_tensors)[i]->get_tensor_id(), (*all_tensors)[i]->get_type(), (*all_tensors)[i]->get_layer_id(), _max_workspace_size);
            }
        }
        /*
        if (((*all_tensors)[i])->get_type() == GRAD) {
            if (_max_grad_size < ((*all_tensors)[i])->get_mem_size()) {
                _max_grad_size = ((*all_tensors)[i])->get_mem_size();
            }
        }
		*/
    }
    _QKV_buffer_size = _max_QKV_size * 6;
    _b_data_pool_size = _max_b_data_size * B_DATA_POOL_NUM;

#ifdef FAKE_TRAIN
    _net_total_size += ((*all_tensors)[0])->get_mem_size() + ((*all_tensors)[1])->get_mem_size();
    _gpu_tensors_size += ((*all_tensors)[0])->get_mem_size() + ((*all_tensors)[1])->get_mem_size();
#endif

#ifdef RECOMPUTE_POOL_NUM
    _recompute_pool_size = _max_data_size * RECOMPUTE_POOL_NUM;
    _max_fragment_size += RECOMPUTE_POOL_NUM * (_max_data_size - _min_tensor_size);
    // _max_fragment_size += RECOMPUTE_POOL_NUM * (_max_data_size - _min_tensor_size);
#endif

#ifdef MULTI_SWAP_BLOCK
#ifdef FEATUREMAP_GRANULARITY
    _swap_pool_size = _max_data_size * SWAP_BLOCK_NUM;
    _max_fragment_size += SWAP_BLOCK_NUM * (_max_data_size - _min_tensor_size);
#endif
#ifdef TENSOR_GRANULARITY
    _swap_pool_size = _max_tensor_size * SWAP_BLOCK_NUM;
    _max_fragment_size += SWAP_BLOCK_NUM * (_max_tensor_size - _min_tensor_size);
#endif
#endif

    // _net_total_size += ((*all_tensors)[0])->get_mem_size() + ((*all_tensors)[1])->get_mem_size() + _max_grad_size;
    // _gpu_tensors_size += ((*all_tensors)[0])->get_mem_size() + ((*all_tensors)[1])->get_mem_size() + _max_grad_size;
    _net_total_size += _max_grad_size + _b_data_pool_size + _max_workspace_size + _QKV_buffer_size; // + _recompute_pool_size + _swap_pool_size;
    _gpu_tensors_size += _max_grad_size + _b_data_pool_size + _max_workspace_size + _QKV_buffer_size;
    _gpu_pool_size = _gpu_tensors_size; // + _swap_pool_size + _recompute_pool_size + _max_workspace_size;

#ifdef TRAINING_CONFIGURATION_SEARCH
    _net_total_size += _max_fragment_size;
    _gpu_pool_size += _swap_pool_size + _recompute_pool_size;
#endif
#ifdef RECOPUTING_SWAPPING_TRAINING
    _net_total_size += _max_fragment_size;
    _gpu_pool_size += _swap_pool_size + _recompute_pool_size;
#endif 

// #ifdef SWAP_ON
//     _gpu_pool_size += _swap_pool_size;
// #endif

// #ifdef RECOMPUTE_ON
//     _gpu_pool_size += _recompute_pool_size;
// #endif

    printf("data_sum = %f\n", BYTE_TO_MB(data_sum));

    size_t remaind = _net_total_size % 2097152;
    if (remaind != 0) { *net_total_size = _net_total_size - remaind + 2097152; }
    remaind = _gpu_pool_size % 2097152;
    if (remaind != 0) { *gpu_pool_size = _gpu_pool_size - remaind + 2097152; }

    *gpu_tensors_size = _gpu_tensors_size;
    *swap_tensors_size = _swap_tensors_size;
    *swap_pool_size = _swap_pool_size;
    *recompute_pool_size = _recompute_pool_size;
    *b_data_pool_size = _b_data_pool_size;
    *QKV_buffer_size = _QKV_buffer_size;
    *max_grad_size = _max_grad_size;
    *max_tensor_size = _max_tensor_size;
    *max_data_size = _max_data_size;
    *max_workspace_size = _max_workspace_size;
    printf("get_size_info:\n");
    printf("net_total_size = %zd = %f\n", *net_total_size, BYTE_TO_MB(*net_total_size));
    printf("gpu_tensors_size = %zd = %f\n", *gpu_tensors_size, BYTE_TO_MB(*gpu_tensors_size));
    printf("max_tensor_size = %zd = %f\n", _max_tensor_size, BYTE_TO_MB(_max_tensor_size));
    printf("gpu_pool_size = %zd = %f\n", *gpu_pool_size, BYTE_TO_MB(*gpu_pool_size));
    printf("b_data_pool_size = %zd = %f\n", *b_data_pool_size, BYTE_TO_MB(*b_data_pool_size));
    printf("QKV_buffer_size = %zd = %f\n", *QKV_buffer_size, BYTE_TO_MB(*QKV_buffer_size));
    printf("swap_tensors_size = %zd = %f\n", *swap_tensors_size, BYTE_TO_MB(*swap_tensors_size));
    printf("swap_pool_size = %zd = %f\n", *swap_pool_size, BYTE_TO_MB(*swap_pool_size));
    printf("recompute_tensor_size = %zd = %f\n", _recompute_tensor_size, BYTE_TO_MB(_recompute_tensor_size));
    printf("recompute_pool_size = %zd = %f\n", *recompute_pool_size, BYTE_TO_MB(*recompute_pool_size));
    printf("max_fragment_size = %zd = %f\n", *max_fragment_size, BYTE_TO_MB(*max_fragment_size));
    printf("max_grad_size = %zd = %f\n", *max_grad_size, BYTE_TO_MB(*max_grad_size));
    printf("max_workspace_size = %zd = %f\n", *max_workspace_size, BYTE_TO_MB(*max_workspace_size));
    printf("max_layer_size = %zd = %f\n", *max_layer_size, BYTE_TO_MB(*max_layer_size));
}

template <class value_type>
size_t registry_t<value_type>::get_inherent_size() {
    std::vector<tensor_t<value_type>* >* all_tensors = this->get_vector();
    size_t now_mem = query_used_mem();
    printf("now_mem = %f\n", BYTE_TO_MB(now_mem));
    for (auto it = all_tensors->begin(); it != all_tensors->end(); ++it) {
        tensor_t<value_type> *t = *it; 
        if (t->get_type() != DATA && t->get_type() != B_DATA
                && t->get_type() != RNN_BUFF && t->get_type() != CONV_BUFF) 
        {
            now_mem -= t->get_mem_size();
		}
    }
    return now_mem;
}

template <class value_type>
size_t registry_t<value_type>::get_total_size_v2() {
    // all tensor do cudamalloc according to layer sequence, except for data tensor
    // cudamalloc注意事项：
    // 1、cudamalloc每次分配2MB倍数的显存，如果存在没有刚好用满2MB倍数的显存分配，则2MB显存块中剩余的部分留给下次分配使用。
    // 2、显存中数据以512bytes为单位对齐
	std::vector<tensor_t<value_type>* >* all_tensors = this->get_vector();
	size_t total_size = 0;
    size_t cuda_mem_block = 2097152;  // 2MB
    size_t align_size = 512;
    size_t block_num = 0;
    size_t tensor_num = 0;
    size_t residual_mem = 0;
    size_t temp_size;

    int malloc_rank = 0;
    for (size_t i = 0; i < all_tensors->size(); i++) {
        if ((*all_tensors)[i]->get_type() != DATA) {
            tensor_num++;
            temp_size = ((*all_tensors)[i])->get_mem_size();
            // temp_size = (temp_size / align_size) * align_size + ((temp_size % align_size > 0 ? align_size : 0));  // align to 512
            // printf("tensor%d real_malloc_rank = %d   fake_malloc_rank = %d\n", (*all_tensors)[i]->get_tensor_id(), (*all_tensors)[i]->get_malloc_rank(), malloc_rank);
            // printf("(*all_tensors)[%d]->get_mem_size() = %zd, type = %d, layerid = %d\n", i, temp_size, (*all_tensors)[i]->get_type(), (*all_tensors)[i]->get_layer_id());
            printf("temp_size = %d\n", temp_size);
            if (temp_size < residual_mem) {
                residual_mem -= temp_size;
            }
            else {
                // temp_size -= residual_mem;
                while(true) {
                    total_size += cuda_mem_block;
                    residual_mem = cuda_mem_block;
                    block_num++;
                    if (temp_size > cuda_mem_block) {
                        // total_size += cuda_mem_block;
                        temp_size -= cuda_mem_block;
                    }
                    else {
                        residual_mem = cuda_mem_block - temp_size;
                        break;
                    }
                    // total_size += cuda_mem_block;
                    // block_num++;
                }
            }
            // if (malloc_rank % 10 == 0)
            // printf("when tensor%d-residual_mem=%zd total_size = %f block_num = %d\n", (*all_tensors)[i]->get_tensor_id(), residual_mem, BYTE_TO_MB(total_size)+582, block_num);
            malloc_rank++;
        }
    }
    // printf("after malloc undata tensor : total_size = %f, block_num = %zd, tensor_num = %zd\n", BYTE_TO_MB(total_size)+582, block_num, tensor_num);
    // printf("after malloc undata tensor : real_total_size = %f\n", BYTE_TO_MB(query_used_mem()));
    // while(1);
    for (size_t i = 0; i < all_tensors->size(); i++) {
        if ((*all_tensors)[i]->get_type() == DATA) {
            tensor_num++;
            temp_size = ((*all_tensors)[i])->get_mem_size();
            temp_size = (temp_size / align_size) * align_size + ((temp_size % align_size > 0 ? align_size : 0));  // align to 512
            if (temp_size < residual_mem) {
                residual_mem -= temp_size;
            }
            else {
                while(true) {
                    block_num++;
                    total_size += cuda_mem_block;
                    // residual_mem = cuda_mem_block;
                    if (temp_size > cuda_mem_block) {
                        // total_size += cuda_mem_block;
                        temp_size -= cuda_mem_block;
                    }
                    else {
                        residual_mem = cuda_mem_block - temp_size;
                        break;
                    }
                }
            }
        }
    }

    printf("total_size = %f, block_num = %zd, tensor_num = %zd\n", BYTE_TO_MB(total_size), block_num, tensor_num);
	return total_size + 582;
}

template <class value_type>
void registry_t<value_type>::delete_all_related_record() {
		// delete record and related tensors
		// printf("before delete_all_related_record mem_usage = %f\n", BYTE_TO_MB(query_used_mem()));
		tensor_t<value_type> *t;
		while (tensors_to_free.size() != 0) {
			t = tensors_to_free.back();
			delete t;
			tensors_to_free.pop_back();
			// printf("tensors_to_free.size() = %d\n", tensors_to_free.size());
		}
		tensors_to_free.erase(tensors_to_free.begin(), tensors_to_free.end());
		int oo = 0;
		for(auto it = outputs.begin(); it != outputs.end(); )
		{
			t = it->second;
			outputs.erase(it++);
			oo++;
			// printf("outputs.size() = %d\n", outputs.size());
		}
		outputs.erase(outputs.begin(), outputs.end());
		for(auto it = b_data.begin(); it != b_data.end(); )
		{
			t = it->second;
			b_data.erase(it++);
		}
		b_data.erase(b_data.begin(), b_data.end());
		for(auto it = bias.begin(); it != bias.end(); ) {
			t = it->second;
			bias.erase(it++);
		}
		bias.erase(bias.begin(), bias.end());
		
		for(auto it = weight.begin(); it != weight.end(); ) {
			t = it->second;
			weight.erase(it++);
		}
		weight.erase(weight.begin(), weight.end());
		for(auto it = bias_grad.begin(); it != bias_grad.end(); ) {
			t = it->second;
			bias_grad.erase(it++);
		}
		bias_grad.erase(bias_grad.begin(), bias_grad.end());
		for(auto it = weight_grad.begin(); it != weight_grad.end(); ) {
			t = it->second;
			weight_grad.erase(it++);
		}
		weight_grad.erase(weight_grad.begin(), weight_grad.end());
		for(auto it = bias_prev.begin(); it != bias_prev.end(); ) {
			t = it->second;
			bias_prev.erase(it++);
		}
		bias_prev.erase(bias_prev.begin(), bias_prev.end());
		for(auto it = weight_prev.begin(); it != weight_prev.end(); ) {
			t = it->second;
			// if(t != NULL) delete t;
			weight_prev.erase(it++);
		}
		weight_prev.erase(weight_prev.begin(), weight_prev.end());
		// printf("delete weight_prev done size = %d\n", weight_prev.size());
		std::vector<tensor_t<value_type>* > *ts;
		for(auto it = forward_dependency.begin(); it != forward_dependency.end(); ) {
			ts = &(it->second);
			while(ts->size() != 0) {
				t = ts->back();
				// printf("t = %x\n", t);
				ts->pop_back();
			}
			forward_dependency.erase(it++);
			// printf("forward_dependency.size() = %d\n", forward_dependency.size());
		}
		forward_dependency.erase(forward_dependency.begin(), forward_dependency.end());
		// printf("delete forward_dependency done = %d\n", forward_dependency.size());
		std::vector<int> *v;
		for(auto it = forward_dependency_by_tensor.begin(); it != forward_dependency_by_tensor.end(); ) {
			t = it->first;
			// if(t != NULL) delete t;
			v = &(it->second);
			while(v->size() != 0) {
				v->pop_back();
			}
			forward_dependency_by_tensor.erase(it++);
		}
		forward_dependency_by_tensor.erase(forward_dependency_by_tensor.begin(), forward_dependency_by_tensor.end());
		// printf("delete forward_dependency_by_tensor done = %d\n", forward_dependency_by_tensor.size());
		for(auto it = backward_dependency.begin(); it != backward_dependency.end(); ) {
			ts = &(it->second);
			while(ts->size() != 0) {
				t = ts->back();
				// if(t != NULL) delete t;
				ts->pop_back();
			}
			backward_dependency.erase(it++);
		}
		backward_dependency.erase(backward_dependency.begin(), backward_dependency.end());
		// printf("delete backward_dependency done = %d\n", backward_dependency.size());
		for(auto it = backward_dependency_by_tensor.begin(); it != backward_dependency_by_tensor.end(); ) {
			t = it->first;
			// if(t != NULL) delete t;
			v = &(it->second);
			while(v->size() != 0) {
				v->pop_back();
			}
			backward_dependency_by_tensor.erase(it++);
		}
		backward_dependency_by_tensor.erase(backward_dependency_by_tensor.begin(), backward_dependency_by_tensor.end());
		// printf("delete backward_dependency_by_tensor done\n");
		while(net_comp_route.size() != 0) {
			net_comp_route.pop_back();
		}
		net_comp_route.erase(net_comp_route.begin(), net_comp_route.end());
		// printf("delete net_comp_route done = %d\n", net_comp_route.size());
		while(net_test_route.size() != 0) {
			net_test_route.pop_back();
		}
		net_test_route.erase(net_test_route.begin(), net_test_route.end());
		// printf("delete net_test_route done = %d\n", net_test_route.size());
		for (auto it = net_layers.begin(); it != net_layers.end(); ) {
			// delete it->second;  // can not delete layers, just delete net_layers record
			net_layers.erase(it++);
		}
		net_layers.erase(net_layers.begin(), net_layers.end());
		// printf("delete net_layers done = %d\n", net_layers.size());
		for (auto it = tensor_by_layer.begin(); it != tensor_by_layer.end(); ) {
			ts = &(it->second);
			while(ts->size() != 0) {
				t = ts->back();
				// if(t != NULL) delete t;
				ts->pop_back();
			}
			tensor_by_layer.erase(it++);
		}
		tensor_by_layer.erase(tensor_by_layer.begin(), tensor_by_layer.end());
		// printf("delete tensor_by_layer done = %d\n", tensor_by_layer.size());
		test_data = NULL;
		train_data = NULL;
		test_label = NULL;
		train_label = NULL;
		// printf("after delete_all_related_record mem_usage = %f\n", BYTE_TO_MB(query_used_mem()));
}
	
template <class value_type>
void registry_t<value_type>::print_tensors_by_layers() {
    typename std::map<int, std::vector< tensor_t<value_type>* > >::iterator it = tensor_by_layer.begin();
    int data_c      = 0;
    int conv_buff_c = 0;
    long total_data        = 0;
    long total_grad        = 0;
    long total_aux         = 0;
    long total_param       = 0;
    long total_conv_buff   = 0;
    long total_bn_mean_var = 0;
    long total_data_source = 0;
    
    for (it = tensor_by_layer.begin(); it != tensor_by_layer.end(); ++it) {
        int layer_id = it->first;
        std::vector<tensor_t<value_type>* > tensors = it->second;
        for(size_t i = 0; i < tensors.size(); i++) {
            TENSOR_TYPE type = tensors[i]->get_type();
            if (type == DATA) {
                data_c += 1;
                total_data += ((long) tensors[i]->get_scalar_count());
                printf("@layer:%d->tensor:%p DATA  size %.3fMB\n", layer_id, tensors[i], (double)tensors[i]->get_mem_size()/1024.0f/1024.0f);
            } else if (type == GRAD) {
                total_grad += ((long) tensors[i]->get_scalar_count());
                printf("@layer:%d->tensor:%p GRAD  size %.3fMB\n", layer_id, tensors[i], (double)tensors[i]->get_mem_size()/1024.0f/1024.0f);
            } else if (type == PARAM) {
                total_param += ((long) tensors[i]->get_scalar_count());
                printf("@layer:%d->tensor:%p PARAM  size %.3fMB\n", layer_id, tensors[i], (double)tensors[i]->get_mem_size()/1024.0f/1024.0f);
            } else if (type == AUX) {
                total_aux += ((long) tensors[i]->get_scalar_count());
                printf("@layer:%d->tensor:%p AUX  size %.3fMB\n", layer_id, tensors[i], (double)tensors[i]->get_mem_size()/1024.0f/1024.0f);
            } else if (type == BN_MEAN_VAR) {
                total_bn_mean_var += ((long) tensors[i]->get_scalar_count());
                printf("@layer:%d->tensor:%p BN_MEAN_VAR  size %.3fMB\n", layer_id, tensors[i], (double)tensors[i]->get_mem_size()/1024.0f/1024.0f);
            } else if(type == CONV_BUFF) {
                conv_buff_c += 1;
                total_conv_buff += ((long) tensors[i]->get_scalar_count());
                printf("@layer:%d->tensor:%p CONV_BUFF  size %.3fMB\n", layer_id, tensors[i], (double)tensors[i]->get_mem_size()/1024.0f/1024.0f);
            } else if(type == DATA_SOURCE) {
                total_data_source += ( (long) tensors[i]->get_scalar_count() );
                printf("@layer:%d->tensor:%p DATA_SOURCE  size %.3fMB\n", layer_id, tensors[i], (double)tensors[i]->get_mem_size()/1024.0f/1024.0f);
            } else {
                printf("@layer:%d->tensor:%p unspecified  size %.3fMB\n", layer_id, tensors[i], (double)tensors[i]->get_mem_size()/1024.0f/1024.0f);
            }
        }
    }
    
    double total_aux_mem         = total_aux*sizeof(value_type)/1024.0f/1024.0f;
    double total_data_mem        = total_data*sizeof(value_type)/1024.0f/1024.0f;
    double total_grad_mem        = total_grad*sizeof(value_type)/1024.0f/1024.0f;
    double total_param_mem       = total_param*sizeof(value_type)/1024.0f/1024.0f;
    double total_conv_buff_mem   = total_conv_buff*sizeof(value_type)/1024.0f/1024.0f;
    double total_bn_mean_var_mem = total_bn_mean_var*sizeof(value_type)/1024.0f/1024.0f;
    double total_data_source_mem = total_data_source*sizeof(value_type)/1024.0f/1024.0f;
    double total_mem             = total_data_mem + total_conv_buff_mem + total_param_mem + total_grad_mem + total_aux_mem + total_bn_mean_var_mem + total_data_source_mem;
    
    printf("TOTAL %d DATA: %ld->%fMB\n", data_c, total_data, total_data_mem);
    printf("TOTAL %d CONV BUFF: %ld->%fMB \n", conv_buff_c, total_conv_buff, total_conv_buff_mem);
    printf("TOTAL PARAMS: %ld->%fMB \n", total_param, total_param_mem);
    printf("TOTAL GRAD: %ld->%fMB \n",   total_grad, total_grad_mem);
    printf("TOTAL AUX: %ld->%fMB \n", total_aux, total_aux_mem);
    printf("TOTAL BN_MEAN_VAR: %ld->%fMB \n", total_bn_mean_var, total_bn_mean_var_mem);
    printf("TOTAL DATA_SOURCE: %ld->%fMB \n", total_data_source, total_data_source_mem);
    printf("TOTAL MEM:%f\n", total_mem);
}

    
template <class value_type>
void registry_t<value_type>::register_tensors_by_layers() {
    std::vector<tensor_t<value_type>* >* all_tensors = this->get_vector();
    std::map<int, tensor_t<value_type>* >* all_weight = this->get_all_weight();
    std::map<int, tensor_t<value_type>* >* all_bias = this->get_all_bias();
    // printf("\n\nall_tensors->size() = %d\n\n", all_tensors->size());
    // printf("\n\nall_weight->size() = %d\n\n", all_weight->size());
    // printf("\n\nall_bias->size() = %d\n\n", all_bias->size());
    for(size_t i = 0; i < all_tensors->size(); i++) {
        tensor_t<value_type>* t = (*all_tensors)[i];
        int layer_id = t->get_layer_id();
        typename std::map<int, std::vector<tensor_t<value_type>* > >::iterator it = tensor_by_layer.find(layer_id);
        if ( it != tensor_by_layer.end() ) {
            it->second.push_back(t);
        } else {
            std::vector<tensor_t<value_type>* > v;
            v.push_back(t);
            tensor_by_layer.insert(std::pair<int, std::vector< tensor_t<value_type>* > >(layer_id, v) );
        }
    }
}
    
template <class value_type>
value_type registry_t<value_type>::get_grad_sqrsum() {
    value_type sum = 0;
    typename std::map<int, tensor_t<value_type>* >::iterator it = weight_grad.begin();
    for (it = weight_grad.begin(); it != weight_grad.end(); ++it) {
        if( it->second != NULL ){
            sum += it->second->squared_sum(&cublas_handle);
        }
    }
    it = bias_grad.begin();
    for (it = bias_grad.begin(); it != bias_grad.end(); ++it) {
        if( it->second != NULL ){
            sum += it->second->squared_sum(&cublas_handle);
        }
    }
    return sum;
}
    

    
template <class value_type>
void registry_t<value_type>::register_layer_param( int layer_id, tensor_t<value_type>* t,  std::map<int, tensor_t<value_type>* >& m, const char* str) {
    typename std::map<int, tensor_t<value_type>* >::iterator it = m.find( layer_id );
    if (it != m.end()) {
        printf("layer:%d %s already registerred!! do nothing\n", layer_id, str);
        exit(1);
    } else {
        m.insert( std::make_pair(layer_id, t) );
    }
#ifdef DEBUG
    print_registry(m, str);
#endif
}
    
template <class value_type>
void registry_t<value_type>::register_output(int source_layer_id, int dest_layer_id, tensor_t<value_type>* t ) {
    assert( t->get_type() == DATA || t->get_type() == DATA_SOURCE );
    d_key_t k(source_layer_id, dest_layer_id);
    typename std::map<d_key_t, tensor_t<value_type>*>::iterator it = outputs.find( k );
    
    if (it != outputs.end()) {
        printf("source layer:%d dest layer:%d output tensor already registerred!!\n", source_layer_id, dest_layer_id);
        exit(1);
    } else {
        outputs.insert( std::make_pair(k, t) );
    }
    
#ifdef DEBUG
    print_registry(outputs, "output");
#endif
}
    
template <class value_type>
void registry_t<value_type>::register_b_data(int source_layer_id, int dest_layer_id, tensor_t<value_type>* t ) {
    assert( t->get_type() == DATA );
    d_key_t k(source_layer_id, dest_layer_id);
    typename std::map<d_key_t, tensor_t<value_type>* >::iterator it = b_data.find( k );
    
    if (it != b_data.end()) {
        printf("source layer%d-type%d dest layer%d-type%d b_data tensor already registerred!!\n", 
            source_layer_id, ((base_layer_t<value_type>*)net_layers.find(source_layer_id)->second)->get_layer_type(), dest_layer_id, ((base_layer_t<value_type>*)net_layers.find(dest_layer_id)->second)->get_layer_type());
        exit(1);
    } else {
        b_data.insert( std::make_pair(k, t) );
    }
    
#ifdef DEBUG
    print_registry(b_data, "b_data");
#endif
}
    
template <class value_type>
tensor_t<value_type>* registry_t<value_type>::get_layer_param(int layer_id, std::map<int, tensor_t<value_type>* >& m) {
    
    typename std::map<int, tensor_t<value_type>* >::iterator it = m.find(layer_id);
    if( it != m.end()) {
        return it->second;
    } else {
        return NULL;
    }
}

template <class value_type>
tensor_t<value_type>* registry_t<value_type>::get_reg_output(int source_layer_id, int dest_layer_id) {
    
    d_key_t k(source_layer_id, dest_layer_id);
    typename std::map<d_key_t, tensor_t<value_type>* >::iterator it = outputs.find(k);
    if( it != outputs.end()) {
        return it->second;
    } else {
        return NULL;
    }
    
}
    
template <class value_type>
tensor_t<value_type>* registry_t<value_type>::get_reg_b_data(int source_layer_id, int dest_layer_id) {
    // printf("%d %d\n", source_layer_id, dest_layer_id);
    d_key_t k(source_layer_id, dest_layer_id);
    typename std::map<d_key_t, tensor_t<value_type>* >::iterator it = b_data.find(k);
    if( it != b_data.end()) {
        return it->second;
    } else {
        return NULL;
    }
    
}
    
template <class value_type>
void registry_t<value_type>::register_forward_dependency( int layer_id, tensor_t<value_type>* t ) {
    
    //to register the forward dependency by layers
    typename std::map<int, std::vector<tensor_t<value_type>* > >::iterator it1 = forward_dependency.find( layer_id );
    if ( it1 != forward_dependency.end() ) {
        it1->second.push_back(t);
    } else {
        std::vector<tensor_t<value_type>* > v;
        v.push_back(t);
        forward_dependency.insert(std::pair<int, std::vector< tensor_t<value_type>* > >(layer_id, v) );
    }
    //to register the forward dependency by tensors
    typename std::map<tensor_t<value_type>*, std::vector<int> >::iterator it2 = forward_dependency_by_tensor.find( t );
    if ( it2 != forward_dependency_by_tensor.end() ) {
        it2->second.push_back(layer_id);
    } else {
        std::vector<int> v;
        v.push_back(layer_id);
        forward_dependency_by_tensor.insert(std::pair<tensor_t<value_type>*, std::vector<int> >(t, v) );
    }

}
    
template <class value_type>
void registry_t<value_type>::register_backward_dependency( int layer_id, tensor_t<value_type>* t ) {
    //to register the backward dependency by layers
    typename std::map<int, std::vector<tensor_t<value_type>* > >::iterator it = backward_dependency.find( layer_id );
    if ( it != backward_dependency.end() ) {
        it->second.push_back(t);
    } else {
        std::vector<tensor_t<value_type>* > v;
        v.push_back(t);
        backward_dependency.insert(std::pair<int, std::vector< tensor_t<value_type>* > >(layer_id, v) );
    }
    
    //to register the backward dependency by tensors
    typename std::map<tensor_t<value_type>*, std::vector<int> >::iterator it2 = backward_dependency_by_tensor.find( t );
    if ( it2 != backward_dependency_by_tensor.end() ) {
        it2->second.push_back(layer_id);
    } else {
        std::vector<int> v;
        v.push_back(layer_id);
        backward_dependency_by_tensor.insert(std::pair<tensor_t<value_type>*, std::vector<int> >(t, v) );
    }

}
    
template <class value_type>
bool registry_t<value_type>::is_included(std::vector<tensor_t<value_type>* > &v, tensor_t<value_type>* t) {
    for(size_t i = 0; i < v.size(); i++) {
        if(v[i] == t) return true;
    }
    return false;
}
    
template <class value_type>
void registry_t<value_type>::print_dependency_by_tensors(std::map<tensor_t<value_type>*, std::vector<int> > &m, net_comp dir) {
    typename std::map<tensor_t<value_type>*, std::vector<int> >::iterator it = m.begin();
    for (it = m.begin(); it != m.end(); ++it) {
        std::vector<int > layers = it->second;
        for(size_t i = 0; i < layers.size(); i++) {
            TENSOR_TYPE type = it->first->get_type();
            if(dir == FORWARD) {
                if (type == DATA) {
                    printf("@tensor:%p forward needed by layer:%d DATA\n", it->first, layers[i]);
                } else if (type == GRAD) {
                    printf("@tensor:%p forward needed by layer:%d GRAD\n", it->first, layers[i]);
                } else if (type == PARAM) {
                    printf("@tensor:%p forward needed by layer:%d PARAM\n", it->first, layers[i]);
                } else if (type == AUX) {
                    printf("@tensor:%p forward needed by layer:%d AUX\n", it->first, layers[i]);
                } else if (type == BN_MEAN_VAR) {
                    printf("@tensor:%p forward needed by layer:%d BN_MEAN_VAR\n", it->first, layers[i]);
                } else if (type == CONV_BUFF) {
                    printf("@tensor:%p forward needed by layer:%d CONV_BUFF\n", it->first, layers[i]);
                } else if (type == DATA_SOURCE) {
                    printf("@tensor:%p forward needed by layer:%d DATA_SOURCE\n", it->first, layers[i]);
                } else {
                    printf("@tensor:%p forward needed by layer:%d UNSPECIFIED\n", it->first, layers[i]);
                }
            } else if(dir == BACKWARD) {
                if (type == DATA) {
                    printf("@tensor:%p backward needed by layer:%d DATA\n", it->first, layers[i]);
                } else if (type == GRAD) {
                    printf("@tensor:%p backward needed by layer:%d GRAD\n", it->first, layers[i]);
                } else if (type == PARAM) {
                    printf("@tensor:%p backward needed by layer:%d PARAM\n", it->first, layers[i]);
                } else if (type == AUX) {
                    printf("@tensor:%p backward needed by layer:%d AUX\n", it->first, layers[i]);
                } else if (type == BN_MEAN_VAR) {
                    printf("@tensor:%p backward needed by layer:%d BN_MEAN_VAR\n", it->first, layers[i]);
                } else if (type == CONV_BUFF) {
                    printf("@tensor:%p backward needed by layer:%d CONV_BUFF\n", it->first, layers[i]);
                } else if (type == DATA_SOURCE) {
                    printf("@tensor:%p backward needed by layer:%d DATA_SOURCE\n", it->first, layers[i]);
                } else {
                    printf("@tensor:%p backward needed by layer:%d UNSPECIFIED\n", it->first, layers[i]);
                }
            }
        }
    }

    
}
    
template <class value_type>
void registry_t<value_type>::print_dependency(std::map<int, std::vector< tensor_t<value_type>* > > &m, net_comp dir) {
    
    long total_data        = 0;
    long total_grad        = 0;
    long total_aux         = 0;
    long total_param       = 0;
    long total_bn_mean_var = 0;
    long total_conv_buff   = 0;
    long total_data_source = 0;
    
    std::vector<tensor_t<value_type>* > dict;
    
    typename std::map<int, std::vector< tensor_t<value_type>* > >::iterator it = m.begin();
    for (it = m.begin(); it != m.end(); ++it) {
        long layer_total = 0;
        int layer_id = it->first;
        std::vector<tensor_t<value_type>* > tensors = it->second;
        
        for(size_t i = 0; i < tensors.size(); i++) {
            TENSOR_TYPE type = tensors[i]->get_type();
            bool is_old = is_included(dict, tensors[i]);
            if(!is_old) dict.push_back(tensors[i]);
            layer_total += ((long) tensors[i]->get_scalar_count());
            if(dir == FORWARD) {
                if (type == DATA) {
                    if(!is_old) total_data += ((long) tensors[i]->get_scalar_count());
                    printf("@layer:%d forward depends on tensor:%p DATA\n", layer_id, tensors[i]);
                } else if (type == GRAD) {
                    if(!is_old) total_grad += ((long) tensors[i]->get_scalar_count());
                    printf("@layer:%d forward depends on tensor:%p GRAD\n", layer_id, tensors[i]);
                } else if (type == PARAM) {
                    if(!is_old) total_param += ((long) tensors[i]->get_scalar_count());
                    printf("@layer:%d forward depends on tensor:%p PARAM\n", layer_id, tensors[i]);
                } else if (type == AUX) {
                    if(!is_old) total_aux += ((long) tensors[i]->get_scalar_count());
                    printf("@layer:%d forward depends on tensor:%p AUX\n", layer_id, tensors[i]);
                } else if (type == BN_MEAN_VAR) {
                    if(!is_old) total_bn_mean_var += ((long) tensors[i]->get_scalar_count());
                    printf("@layer:%d forward depends on tensor:%p BN MEAN&VAR\n", layer_id, tensors[i]);
                } else if (type == CONV_BUFF) {
                    if(!is_old) total_conv_buff += ((long) tensors[i]->get_scalar_count());
                    printf("@layer:%d forward depends on tensor:%p CONV_BUFF\n", layer_id, tensors[i]);
                } else if(type == DATA_SOURCE) {
                    total_data_source += ( (long) tensors[i]->get_scalar_count() );
                    printf("@layer:%d->tensor:%p DATA_SOURCE\n", layer_id, tensors[i]);
                } else {
                    printf("@layer:%d forward depends on tensor:%p UNSPECIFIED\n", layer_id, tensors[i]);
                }
            } else if(dir == BACKWARD) {
                if (type == DATA) {
                    if(!is_old) total_data += ((long) tensors[i]->get_scalar_count());
                    printf("@layer:%d backward depends on tensor:%p DATA\n", layer_id, tensors[i]);
                } else if (type == GRAD) {
                    if(!is_old) total_grad += ((long) tensors[i]->get_scalar_count());
                    printf("@layer:%d backward depends on tensor:%p GRAD\n", layer_id, tensors[i]);
                } else if (type == PARAM) {
                    if(!is_old) total_param += ((long) tensors[i]->get_scalar_count());
                    printf("@layer:%d backward depends on tensor:%p PARAM\n", layer_id, tensors[i]);
                } else if (type == AUX) {
                    if(!is_old) total_aux += ((long) tensors[i]->get_scalar_count());
                    printf("@layer:%d backward depends on tensor:%p AUX\n", layer_id, tensors[i]);
                } else if (type == BN_MEAN_VAR) {
                    if(!is_old) total_bn_mean_var += ((long) tensors[i]->get_scalar_count());
                    printf("@layer:%d backward depends on tensor:%p BN MEAN&VAR\n", layer_id, tensors[i]);
                } else if (type == CONV_BUFF) {
                    if(!is_old) total_conv_buff += ((long) tensors[i]->get_scalar_count());
                    printf("@layer:%d backward depends on tensor:%p CONV_BUFF\n", layer_id, tensors[i]);
                } else if(type == DATA_SOURCE) {
                    total_data_source += ( (long) tensors[i]->get_scalar_count() );
                    printf("@layer:%d->tensor:%p DATA_SOURCE\n", layer_id, tensors[i]);
                } else {
                    printf("@layer:%d backward depends on tensor:%p UNSPECIFIED\n", layer_id, tensors[i]);
                }
            }
        }
        double total_layer_mem = layer_total*sizeof(value_type)/1000000.0f;
        printf("--------layer memory subtotal:%f--------\n", total_layer_mem);
    }
    double total_data_mem        = total_data*sizeof(value_type)/1000000.0f;
    double total_conv_buff_mem   = total_conv_buff*sizeof(value_type)/1000000.0f;
    double total_param_mem       = total_param*sizeof(value_type)/1000000.0f;
    double total_grad_mem        = total_grad*sizeof(value_type)/1000000.0f;
    double total_aux_mem         = total_aux*sizeof(value_type)/1000000.0f;
    double total_bn_mean_var_mem = total_bn_mean_var*sizeof(value_type)/1000000.0f;
    double total_data_source_mem = total_data_source*sizeof(value_type)/1000000.0f;
    double total_mem             = total_data_mem + total_conv_buff_mem + total_param_mem + total_grad_mem + total_aux_mem + total_bn_mean_var_mem + total_data_source_mem;
    
    printf("TOTAL DATA: %ld->%fMB \n", total_data, total_data_mem);
    printf("TOTAL CONV BUFF: %ld->%fMB \n", total_conv_buff, total_conv_buff_mem);
    printf("TOTAL PARAMS: %ld->%fMB \n", total_param, total_param_mem);
    printf("TOTAL GRAD: %ld->%fMB \n",   total_grad, total_grad_mem);
    printf("TOTAL AUX: %ld->%fMB \n", total_aux, total_aux_mem);
    printf("TOTAL BN_MEAN_VAR: %ld->%fMB \n", total_bn_mean_var, total_bn_mean_var_mem);
    printf("TOTAL DATA_SOURCE: %ld->%fMB \n", total_data_source, total_data_source_mem);
    printf("TOTAL MEM:%f\n", total_mem);

}



INSTANTIATE_CLASS(registry_t);

} //ATP namespace
