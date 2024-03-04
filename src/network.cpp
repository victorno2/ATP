#include <network.h>
#include <tensor.h>
#include <fstream>

namespace ATP{

template <class value_type>
void network_t<value_type>::test_output_swap_time(base_layer_t<value_type>* layer, net_comp dir) {
    tensor_t<value_type>* tensor;
    std::vector<tensor_t<value_type>*> tensors;
    if (dir == FORWARD) {
        // printf("test layer%d-type%d offload\n", layer->get_base_id(), layer->get_layer_type());
        if (layer->get_layer_type() == SOFTMAX) return;
        if (layer->get_layer_structure() == MIMO) {
            tensors = ((base_structure_t<value_type>*)layer)->get_outputs();
            for (int i = 0; i < tensors.size(); i++) {
                tensors[i]->set_swap_time(test_swap_time(tensors[i], OFFLOAD), OFFLOAD);
            }
        }
        else {
            tensor = ((base_network_layer_t<value_type>*)layer)->get_f_out();
            tensor->set_swap_time(test_swap_time(tensor, OFFLOAD), OFFLOAD);
        }
    }
    else {
        // printf("test layer%d-type%d fetch\n", layer->get_base_id(), layer->get_layer_type());
        if (layer->get_layer_type() == DATA_L) return;
        if (layer->get_layer_structure() == MIMO) {
            tensors = ((base_structure_t<value_type>*)layer)->get_inputs();
            for (int i = 0; i < tensors.size(); i++) {
                // printf("tensors[%d]-id=%d gpu_ptr=%x cpu_ptr=%x \n", i, tensors[i]->get_tensor_id(), tensors[i]->get_gpu_ptr(), tensors[i]->get_cpu_ptr());
                tensors[i]->set_swap_time(test_swap_time(tensors[i], FETCH), FETCH);
            }
        }
        else {
            tensor = ((base_network_layer_t<value_type>*)layer)->get_f_in();
            tensor->set_swap_time(test_swap_time(tensor, FETCH), FETCH);
        }
    }
}

template <class value_type>
void network_t<value_type>::test_output_swap_time_v2(base_layer_t<value_type>* layer, net_comp dir) {
    tensor_t<value_type>* tensor;
    std::vector<tensor_t<value_type>*> tensors;
    if (dir == FORWARD) {
        // printf("test layer%d-type%d offload\n", layer->get_base_id(), layer->get_layer_type());
        if (layer->get_layer_type() == SOFTMAX) return;
        if (layer->get_layer_structure() == MIMO) {
            tensors = ((base_structure_t<value_type>*)layer)->get_outputs();
            for (int i = 0; i < tensors.size(); i++) {
                tensors[i]->set_swap_time(test_swap_time_v2(tensors[i], layer, OFFLOAD), OFFLOAD);
            }
        }
        else {
            tensor = ((base_network_layer_t<value_type>*)layer)->get_f_out();
            tensor->set_swap_time(test_swap_time_v2(tensor, layer, OFFLOAD), OFFLOAD);
        }
    }
    else {
        // printf("test layer%d-type%d fetch\n", layer->get_base_id(), layer->get_layer_type());
        if (layer->get_layer_type() == DATA_L) return;
        if (layer->get_layer_structure() == MIMO) {
            tensors = ((base_structure_t<value_type>*)layer)->get_inputs();
            for (int i = 0; i < tensors.size(); i++) {
                // printf("tensors[%d]-id=%d gpu_ptr=%x cpu_ptr=%x \n", i, tensors[i]->get_tensor_id(), tensors[i]->get_gpu_ptr(), tensors[i]->get_cpu_ptr());
                tensors[i]->set_swap_time(test_swap_time_v2(tensors[i], layer, FETCH), FETCH);
            }
        }
        else {
            tensor = ((base_network_layer_t<value_type>*)layer)->get_f_in();
            tensor->set_swap_time(test_swap_time_v2(tensor, layer, FETCH), FETCH);
        }
    }
}

template <class value_type>
double network_t<value_type>::test_swap_time_v2(tensor_t<value_type>* tensor, base_layer_t<value_type>* layer, SWAP_DIR dir) {
    size_t tensor_size = tensor->get_mem_size();
    double start;
    double end;
    double temp;
    size_t t;
    if (dir == OFFLOAD) {
        start = get_cur_time();
        tensor->async_gpu_to_cpu();
        layer->forward(NET_TRAIN, &cublas_handle, &cudnn_handle, reg);
        cudaStreamSynchronize(stream_singleton::get_compute_stream());
        mem_controller.synchronize(GPU2CPU_STREAM);
        end = get_cur_time();
        if (offload_time_by_size.count(tensor->get_tensor_id())) {
            t = offload_test_times_by_tensor[tensor->get_tensor_id()];
            temp = offload_time_by_tensor[tensor->get_tensor_id()];

            offload_time_by_size[tensor->get_tensor_id()] = (end-start)/(double)t + temp*((double)(t-1))/(double)t;
            offload_test_times_by_tensor[tensor->get_tensor_id()] = t + 1;
        }
        else {
            offload_time_by_tensor[tensor->get_tensor_id()] = end - start;
            offload_test_times_by_tensor[tensor->get_tensor_id()] = 1;
        }
        temp = offload_time_by_tensor[tensor->get_tensor_id()];
        return temp;
    }
    else {
        start = get_cur_time();
        tensor->async_cpu_to_gpu();
        layer->backward(NET_TRAIN, &cublas_handle, &cudnn_handle, reg);
        cudaStreamSynchronize(stream_singleton::get_compute_stream());
        mem_controller.synchronize(CPU2GPU_STREAM);
        end = get_cur_time();
        if (fetch_time_by_size.count(tensor->get_tensor_id())) {
            t = fetch_test_times_by_tensor[tensor->get_tensor_id()];
            temp = fetch_time_by_tensor[tensor->get_tensor_id()];
            fetch_time_by_tensor[tensor->get_tensor_id()] = (end-start)/(double)t + temp*((double)(t-1))/(double)t;
            fetch_test_times_by_tensor[tensor->get_tensor_id()] = t + 1;
        }
        else {
            fetch_time_by_tensor[tensor->get_tensor_id()] = end - start;
            fetch_test_times_by_tensor[tensor->get_tensor_id()] = 1;
        }
        temp = fetch_time_by_tensor[tensor->get_tensor_id()];
        return temp;
    }
}

template <class value_type>
double network_t<value_type>::test_swap_time(tensor_t<value_type>* tensor, SWAP_DIR dir) {
    size_t tensor_size = tensor->get_mem_size();
    double start;
    double end;
    double temp;
    size_t t;
    if (dir == OFFLOAD) {
        start = get_cur_time();
        tensor->async_gpu_to_cpu();
        mem_controller.synchronize(GPU2CPU_STREAM);
        end = get_cur_time();
        if (offload_time_by_size.count(tensor_size)) {
            t = offload_test_times_by_size[tensor_size];
            temp = offload_time_by_size[tensor_size];

            offload_time_by_size[tensor_size] = (end-start)/(double)t + temp*((double)(t-1))/(double)t;
            offload_test_times_by_size[tensor_size] = t + 1;
        }
        else {
            offload_time_by_size[tensor_size] = end - start;
            offload_test_times_by_size[tensor_size] = 1;
        }
        temp = offload_time_by_size[tensor_size];
        return temp;
    }
    else {
        start = get_cur_time();
        tensor->async_cpu_to_gpu();
        mem_controller.synchronize(CPU2GPU_STREAM);
        end = get_cur_time();
        if (fetch_time_by_size.count(tensor_size)) {
            t = fetch_test_times_by_size[tensor_size];
            temp = fetch_time_by_size[tensor_size];
            fetch_time_by_size[tensor_size] = (end-start)/(double)t + temp*((double)(t-1))/(double)t;
            fetch_test_times_by_size[tensor_size] = t + 1;
        }
        else {
            fetch_time_by_size[tensor_size] = end - start;
            fetch_test_times_by_size[tensor_size] = 1;
        }
        temp = fetch_time_by_size[tensor_size];
        return temp;
    }
}

template <class value_type>
void network_t<value_type>::select_swap_layer_by_route(std::vector<int>* swap_layers_by_route, size_t sn) {
    std::map<int, void* > layers = reg->get_net_layers();
    std::vector<std::pair<int, net_comp> > net_comp = reg->get_net_comp_route();
    base_layer_t<value_type>* layer;
    size_t layer_id;
    int j = 0;
    int max_layer_id = this->get_max_layer_id();
    int forward_end_route = 0;
    int forward_end_layer_id = 0;
    for (int i = 0; i < net_comp.size(); i++) {
        if (net_comp[i].second == BACKWARD) {
            forward_end_route = i - 1;
            forward_end_layer_id = net_comp[forward_end_route].first;
            printf("forward_end_route=%d\n", forward_end_route);
            printf("forward_end_layer_id=%d\n", forward_end_layer_id);
            // exit(1);
            break;
        }
    }
    for (int i = 0; i < net_comp.size(); i++) {
        if (net_comp[i].second == FORWARD) {
            layer_id = net_comp[i].first;
            layer = (base_layer_t<value_type>*) layers.find(layer_id)->second;
            if (i >= forward_end_route - SWAP_BLOCK_NUM && i <= forward_end_route-1) {
                swap_layers_by_route->push_back(i);
                printf("select layer%d as swap layer\n", (*swap_layers_by_route)[j]);
                j++;
            }
            if (layer->get_layer_type() == CONV) {
                swap_layers_by_route->push_back(i);
                printf("select layer%d as swap layer\n", (*swap_layers_by_route)[j]);
                j++;
                sn--;
            }
            if (sn == 0) break; 
        }
    }
}

template <class value_type>
void network_t<value_type>::select_swap_layer(std::vector<int>* swap_layers, size_t sn) {
    std::map<int, void* > layers = reg->get_net_layers();
    std::vector<std::pair<int, net_comp> > net_comp = reg->get_net_comp_route();
    base_layer_t<value_type>* layer;
    size_t layer_id;
    int j = 0;
    int max_layer_id = this->get_max_layer_id();
    int forward_end_route = 0;
    int forward_end_layer_id = 0;
    for (int i = 0; i < net_comp.size(); i++) {
        if (net_comp[i].second == BACKWARD) {
            forward_end_route = i - 1;
            forward_end_layer_id = net_comp[forward_end_route].first;
            printf("forward_end_route=%d\n", forward_end_route);
            printf("forward_end_layer_id=%d\n", forward_end_layer_id);
            // exit(1);
            break;
        }
    }
    for (int i = 0; i < net_comp.size(); i++) {
        if (net_comp[i].second == FORWARD) {
            layer_id = net_comp[i].first;
            layer = (base_layer_t<value_type>*) layers.find(layer_id)->second;
            if (i >= forward_end_route - SWAP_BLOCK_NUM && i <= forward_end_route-1) {
                swap_layers->push_back(net_comp[i].first);
                printf("select layer%d as swap layer\n", (*swap_layers)[j]);
                j++;
            }
            if (layer->get_layer_type() == CONV) {
                swap_layers->push_back(layer_id);
                printf("select layer%d as swap layer\n", (*swap_layers)[j]);
                j++;
                sn--;
            }
            if (sn == 0) break; 
        }
    }
}

template <class value_type>
void network_t<value_type>::select_recompute_layer_rnn(std::vector<int>* recompute_layers, size_t rn) {
    std::map<int, void* > layers = reg->get_net_layers();
    base_layer_t<value_type>* layer;
    int i = 0;
    for (auto it = layers.begin(); it != layers.end(); it++) {
        layer = (base_layer_t<value_type>*)(it->second);
        if (layer->get_layer_type() == BN) {
            recompute_layers->push_back(it->first);
            printf("select layer%d as recomputing layer\n", (*recompute_layers)[i]);
            i++;
            rn--;
        }
        if (rn == 0) break;
        // if (i == 10) break;
    }
}

template <class value_type>
void network_t<value_type>::select_recompute_layer(std::vector<int>* recompute_layers, size_t rn) {
    std::map<int, void* > layers = reg->get_net_layers();
    base_layer_t<value_type>* layer;
    int i = 0;
    for (auto it = layers.begin(); it != layers.end(); it++) {
        layer = (base_layer_t<value_type>*)(it->second);
        // if (layer->get_layer_type() == ACT) {
        if (layer->get_layer_type() == ACT || layer->get_layer_type() == BN) {
        // if (layer->get_layer_type() == CONV) {
            if (((base_network_layer_t<value_type>*)layer)->get_f_out()->get_position() != SHARED_GPU_POOL) {
                recompute_layers->push_back(it->first);
                printf("select layer%d as recomputing layer\n", (*recompute_layers)[i]);
                rn--;
                i++;
            }
            
        }
        if (rn == 0) break;
    }
}

template <class value_type>
size_t network_t<value_type>::GetLayerNum() {
    return reg->get_net_layers_ptr()->size();
}


template <class value_type>
bool network_t<value_type>::SetRecomputeSwapTensorsbyRoute(int* code, size_t code_size) {
    std::vector<base_layer_t<value_type>*>* next_layers;
    std::map<int, void* > layers = reg->get_net_layers();
    std::vector<std::pair<int, net_comp> > net_comp_route = reg->get_net_comp_route();
    base_layer_t<value_type>* layer;
    // if (code_size != layers.size()) {
    //     printf("SetRecomputeSwapTensorsbyRoute false, code_size = %d != layers.size = %d", code_size, layers.size());
    //     return false;
    // }
    size_t re_size = 0;
    bool useless_flag = false;
    size_t counter = 0;
    size_t j = 0;
    for ( int i = 0; i < net_comp_route.size(); i++ ) {
        layer = (base_layer_t<value_type>*) layers.find(net_comp_route[i].first)->second;
        // #define DEBUG
        if (net_comp_route[i].second == FORWARD) {
            if (layer->get_layer_structure() == MIMO) {
                std::vector<tensor_t<value_type>*> tensors = ((base_structure_t<value_type>*)layer)->get_outputs();
                for (int k = 0; k < tensors.size(); k++) {
                    if (code[j] == 1) {
                        tensors[k]->set_position(SHARED_GPU_POOL);
                        #ifdef DEBUG
                            printf("layer%d-type%d: tensor%d->get_position() == %d\n", layer->get_base_id(), layer->get_layer_type(), tensors[k]->get_tensor_id(), tensors[k]->get_position());
                        #endif
                        // if (tensors[k]->get_backward_useful() == true) {
                        if (true) {
                            auto iter = std::find(this->swap_tensors.begin(), this->swap_tensors.end(), tensors[k]);
                            if (iter == this->swap_tensors.end()) {  // Prevent duplicate addition
                                this->swap_tensors.push_back(tensors[k]);
                            }
                        }
                    }
                    else if (code[j] == 2) {
                        tensors[k]->set_position(RECOMPUTE_IN_BACKWARD);
                        re_size += tensors[k]->get_mem_size();
                        #ifdef DEBUG
                            printf("layer%d-type%d: tensor%d->get_position() == %d\n", layer->get_base_id(), layer->get_layer_type(), tensors[k]->get_tensor_id(), tensors[k]->get_position());
                        #endif
                    }
                    else {
                        // tensors[k]->set_position(REMAIN_IN_GPU);
                        // #ifdef DEBUG
                        //     printf("layer%d-type%d: tensor%d->get_position() == %d\n", layer->get_base_id(), layer->get_layer_type(), tensors[k]->get_tensor_id(), tensors[k]->get_position());
                        // #endif
                    }
                    j++;
                    counter++;
                }
            }
            else {  // SISO layer
                if (layer->get_layer_type() == RNN || layer->get_layer_type() == SATTN) {
                    // tensor_t<value_type>* reserve_buff = ((rnn_layer_t<value_type>*)layer)->get_rnn_reserve_buff();
                    tensor_t<value_type>* reserve_buff = ((base_network_layer_t<value_type>*)layer)->get_reserve_buff();
                    if (code[j] == 1) {
                        reserve_buff->set_position(SHARED_GPU_POOL);
                        this->swap_tensors.push_back(reserve_buff);
                        #ifdef DEBUG
                            printf("layer%d-type%d: reserve_buff%d->get_position() == %d\n", layer->get_base_id(), layer->get_layer_type(), reserve_buff->get_tensor_id(), reserve_buff->get_position());
                        #endif
                    }
                    else {
                        reserve_buff->set_position(REMAIN_IN_GPU);
                        printf("layer%d-type%d: reserve_buff%d->get_position() == %d\n", layer->get_base_id(), layer->get_layer_type(), reserve_buff->get_tensor_id(), reserve_buff->get_position());
                    }
                    j++;
                    counter++;
                }
                tensor_t<value_type>* tensor = ((base_network_layer_t<value_type>*)layer)->get_f_out();
                if (code[j] == 1) {
                    // layer->set_layer_position(SWAP_LAYER);
                    tensor->set_position(SHARED_GPU_POOL);
                    #ifdef DEBUG
                        printf("layer%d-type%d: tensor%d->get_position() == %d\n",  layer->get_base_id(), layer->get_layer_type(), tensor->get_tensor_id(), tensor->get_position());
                    #endif
                    // if (tensor->get_backward_useful() == true) {
                    if (true) {
                        auto iter = std::find(this->swap_tensors.begin(), this->swap_tensors.end(), tensor);
                        if (iter == this->swap_tensors.end()) {  // Prevent duplicate addition
                            this->swap_tensors.push_back(tensor);
                        }
                    }
                }
                else if (code[j] == 2) {
                    // layer->set_layer_position(RECOMPUTE_LAYER);
                    tensor->set_position(RECOMPUTE_IN_BACKWARD);
                    re_size += tensor->get_mem_size();
                    this->recompute_tensors.push_back(tensor);
                    #ifdef DEBUG
                        printf("layer%d-type%d: tensor%d->get_position() == %d\n",  layer->get_base_id(), layer->get_layer_type(), tensor->get_tensor_id(), tensor->get_position());
                    #endif
                }
                else {
                    // layer->set_layer_position(REMAIN_LAYER);
                    // tensor->set_position(REMAIN_IN_GPU);
                    // #ifdef DEBUG
                    //     printf("layer%d-type%d: tensor%d->get_position() == %d\n",  layer->get_base_id(), layer->get_layer_type(), tensor->get_tensor_id(), tensor->get_position());
                    // #endif
                }
                j++;
                counter++;
            }   
        }
        // #undef DEBUG
        else {  // Ensure the sequence of prefetching in Backward propagation
            // printf("B SetRecomputeSwapTensorsbyRoute %d\n", i);
            if (layer->get_layer_type() == DATA_L) continue;
            if (layer->get_layer_structure() == MIMO) {
                if (layer->get_layer_type() != FORK_L && layer->get_layer_type() != JOIN_L && layer->get_layer_type() != CONCAT)
                {
                    std::vector<tensor_t<value_type>*> tensors = ((base_structure_t<value_type>*)layer)->get_inputs();
                    for (int k = tensors.size()-1; k >= 0; k--) {
                        #ifdef DEBUG
                            printf("tensor%d->get_position() == %d\n", tensors[k]->get_layer_id(), tensors[k]->get_position());
                        #endif
                        if (tensors[k]->get_position() == SHARED_GPU_POOL) {
                            if (tensors[k]->get_use_counter(BACKWARD) > 0 || tensors[k]->get_type() == RNN_RESERVE) {
                                auto iter = std::find(prefetch_tensors.begin(), prefetch_tensors.end(), tensors[k]);
                                if (iter == prefetch_tensors.end()) {  // Prevent duplicate addition
                                    prefetch_tensors.push_back(tensors[k]);
                                    #ifdef DEBUG
                                        printf("layer%d-type%d: tensor%d->get_position() == %d\n", layer->get_base_id(), layer->get_layer_type(), tensors[k]->get_tensor_id(), tensors[k]->get_position());
                                    #endif
                                }
                            } 
                            // this->prefetch_tensors.push_back(tensors[j]); 
                        }                   
                    }
                }   
            }
            else {
                if (layer->get_layer_type() == RNN || layer->get_layer_type() == SATTN) {
                    // tensor_t<value_type>* reserve_buff = ((rnn_layer_t<value_type>*)layer)->get_rnn_reserve_buff();
                    tensor_t<value_type>* reserve_buff = ((base_network_layer_t<value_type>*)layer)->get_reserve_buff();
                    if (reserve_buff->get_position() == SHARED_GPU_POOL) {
                        this->prefetch_tensors.push_back(reserve_buff);
                        #ifdef DEBUG
                            printf("layer%d-type%d: reserve_buff%d->get_position() == %d\n", layer->get_base_id(), layer->get_layer_type(), reserve_buff->get_tensor_id(), reserve_buff->get_position());
                        #endif
                    }
                }
                tensor_t<value_type>* tensor = ((base_network_layer_t<value_type>*)layer)->get_f_in();
                #ifdef DEBUG
                    printf("tensor%d->get_position() == %d\n", tensor->get_layer_id(), tensor->get_position());
                #endif
                if (tensor->get_position() == SHARED_GPU_POOL) {
                    if (tensor->get_use_counter(BACKWARD) > 0) {
                        this->prefetch_tensors.push_back(tensor);
                        #ifdef DEBUG
                            printf("layer%d-type%d: tensor%d->get_position() == %d\n",  layer->get_base_id(), layer->get_layer_type(), tensor->get_tensor_id(), tensor->get_position());
                        #endif
                    } 
                }
            }
        }
    }
    std::vector<tensor_t<value_type>* >* all_tensors = reg->get_vector();
    size_t _recompute_tensor_size = 0;
    for (int i = 0; i < all_tensors->size(); i++) {
        if ((*all_tensors)[i]->get_position() == RECOMPUTE_IN_BACKWARD) {
            _recompute_tensor_size += ((*all_tensors)[i])->get_mem_size();
        }
    }
    this->forward_pur_swap_times = new double[this->swap_tensors.size()];
    this->backward_pur_swap_times = new double[this->prefetch_tensors.size()];
    printf("counter = %d, code_size = %d, re_size = %.2f, _recompute_tensor_size = %.2f\n", counter, code_size, BYTE_TO_MB(re_size), BYTE_TO_MB(_recompute_tensor_size));
    if (counter != code_size) {
        return false;
    }
    else {
        return true;
    }
    // return true;
    // printf("swap_tensors_size = %d, prefetch_tensors_size = %d\n", this->swap_tensors.size(), this->prefetch_tensors.size());
    // if (this->swap_tensors.size() == this->prefetch_tensors.size()) {
    //     return true;
    // }
    // else {
    //     return false;
    // }
    // for ( int i = 0; i < net_comp_route.size(); i++ ) {
    //     if (net_comp_route[i].second == FORWARD) {
    //         layer = (base_layer_t<value_type>*) layers.find(net_comp_route[i].first)->second;
    //         printf("(L%d-t%d-p:", layer->get_base_id(), layer->get_layer_type());
    //         if (layer->get_layer_position() == SWAP_LAYER) {
    //             printf("S) ");
    //         }
    //         else if (layer->get_layer_position() == RECOMPUTE_LAYER) {
    //             printf("R) ");
    //         }
    //         else {
    //             printf("N) ");
    //         }
    //     }
    // }
    // printf("\n");
}

template <class value_type>
size_t network_t<value_type>::set_recompute_layer(std::vector<int>* recompute_layers) {
    auto layers = reg->get_net_layers();
    base_layer_t<value_type>* layer;
    base_network_layer_t<value_type>* net_layer;
    tensor_t<value_type>* tensor;
    for (size_t i = 0; i < recompute_layers->size(); i++) {
        layer = (base_layer_t<value_type>*)layers[(*recompute_layers)[i]];
        net_layer = (base_network_layer_t<value_type>*)layer;
        if ((*recompute_layers)[i] > max_layer_id) {
            printf("can not recompute layer%d!!!\n", (*recompute_layers)[i]);
            return 0;
        }
        else if (net_layer->get_f_out()->get_position() == SHARED_GPU_POOL) {
            printf("can not recompute layer%d!!!\n", (*recompute_layers)[i]);
            return 0;
        }
    }
    // return 0;
    for (size_t i = 0; i < recompute_layers->size(); i++) {
        layer = (base_layer_t<value_type>*)layers[(*recompute_layers)[i]];
        printf("%d: selected layer%d-type%d as a recompute layer\n", i, layer->get_base_id(), layer->get_layer_type());
        layer->set_layer_position(RECOMPUTE_LAYER);
        net_layer = (base_network_layer_t<value_type>* )layer;
        tensor = net_layer->get_f_out();
        tensor->set_position(RECOMPUTE_IN_BACKWARD);
        recompute_tensors.push_back(tensor);
        printf("recompute_layers:%d/%d  layers:%d/%d: layer%d layer%d-type%d\n", i, recompute_layers->size(), i, layers.size(), (*recompute_layers)[i], layer->get_base_id(), layer->get_layer_type());
    }
    printf("set_recompute_layer done %d\n", 666);
    return 0;
    // while(1);
}

template <class value_type>
size_t network_t<value_type>::set_swap_layer(std::vector<int>* swap_layers) {
    // check
    auto layers = reg->get_net_layers();
    for (size_t i = 0; i < swap_layers->size(); i++) {
        if ((*swap_layers)[i] > max_layer_id) {
            printf("can not swap layer%d!!!\n", (*swap_layers)[i]);
            return 0;
        }
    }
    int swap_layers_size[swap_layers->size()];
    size_t min_pool_size = 0;
    size_t min_pool_size_0 = 0;
    size_t min_pool_size_1 = 0;
    size_t temp_size_0 = 0;
    size_t temp_size_1 = 0;
    size_t sum = 0;
    base_layer_t<value_type>* layer;
    base_network_layer_t<value_type>* net_layer;
    base_structure_t<value_type>* structure_layer;
    tensor_t<value_type>* tensor;
    std::vector<tensor_t<value_type>*> tensors;
#ifndef MULTI_SWAP_BLOCK
    for (size_t i = 0; i < swap_layers->size(); i++) {
        layer = (base_layer_t<value_type>*)layers[(*swap_layers)[i]];
        layer->set_layer_position(SWAP_LAYER);
        if (layer->get_layer_structure() == MIMO) {
            structure_layer = (base_structure_t<value_type>* )layer;
            tensors = structure_layer->get_outputs();
            tensors[0]->set_position(SHARED_GPU_POOL);
            this->swap_tensors.push_back(tensors[0]);
            temp_size_0 = tensors[0]->get_mem_size();
            sum += temp_size_0;
            swap_layers_size[i] = temp_size_0;
            if (temp_size_0 > min_pool_size_0) {
                min_pool_size_0 = temp_size_0;
            }
        }
        else {
            net_layer = (base_network_layer_t<value_type>* )layer;
            // net_layer->set_layer_position(SWAP_LAYER);
            tensor = net_layer->get_f_out();
            tensor->set_position(SHARED_GPU_POOL);
            this->swap_tensors.push_back(tensor);
            layer->set_layer_position(SWAP_LAYER);
            if (i == 0) {
                tensor->set_first_last_pool_tensor(0);
            }
            else if (i == (swap_layers->size()-1)) {
                tensor->set_first_last_pool_tensor(1);
            }
            temp_size_1 = tensor->get_mem_size();
            swap_layers_size[i] = temp_size_1;
            // fflush(stdout);
            sum += temp_size_1;
            if (temp_size_1 > min_pool_size_1) {
                min_pool_size_1 = temp_size_1;
            }
        }
        min_pool_size = min_pool_size_0 > min_pool_size_1 ? min_pool_size_0 : min_pool_size_1; 
    }
#endif

#ifdef MULTI_SWAP_BLOCK
    std::vector<std::pair<int, net_comp> >* net_comp_route = reg->get_net_comp_route_ptr();
    std::map<int, void* >* net_layers  = reg->get_net_layers_ptr();
    // forward swap_tensors init
    for (size_t i = 0; i < swap_layers->size(); i++) {
        int layer_id = (*net_comp_route)[(*swap_layers)[i]].first;
        base_layer_t<value_type>* layer = (base_layer_t<value_type>*) net_layers->find(layer_id)->second;
        layer->set_layer_position(SWAP_LAYER);
        printf("%d layer%d-type%d is swap_layer ", i, layer_id, layer->get_layer_type());
        if (layer->get_layer_structure() == MIMO) {
            structure_layer = (base_structure_t<value_type>* )layer;
            tensors = structure_layer->get_outputs();
            tensors[0]->set_position(SHARED_GPU_POOL);
            printf("mimo tensor%d is swap_tensor\n", tensors[0]->get_tensor_id());
            this->swap_tensors.push_back(tensors[0]);
            temp_size_0 = tensors[0]->get_mem_size();
            sum += temp_size_0;
            swap_layers_size[i] = temp_size_0;
            if (temp_size_0 > min_pool_size_0) {
                min_pool_size_0 = temp_size_0;
            }
        }
        else {  // layer->get_layer_structure() == SISO
            net_layer = (base_network_layer_t<value_type>* )layer;
            // net_layer->set_layer_position(SWAP_LAYER);
            tensor = net_layer->get_f_out();
            tensor->set_position(SHARED_GPU_POOL);
            printf("siso tensor%d is swap_tensor\n", tensor->get_tensor_id());
            this->swap_tensors.push_back(tensor);
            if (i == 0) {
                tensor->set_first_last_pool_tensor(0);
            }
            else if (i == (swap_layers->size()-1)) {
                tensor->set_first_last_pool_tensor(1);
            }
            temp_size_1 = tensor->get_mem_size();
            swap_layers_size[i] = temp_size_1;
            sum += temp_size_1;
            if (temp_size_1 > min_pool_size_1) {
                min_pool_size_1 = temp_size_1;
            }
        }
        // printf("ddd %d", 666);
        min_pool_size = min_pool_size_0 > min_pool_size_1 ? min_pool_size_0 : min_pool_size_1; 
    }
    // backward prefetch_tensors init
    // printf("ddd %d", 666);
    printf("ddd %d", net_comp_route->size());
    for (size_t i = 0; i < net_comp_route->size(); i++) {
        if((*net_comp_route)[i].second == BACKWARD ) {
            int layer_id = (*net_comp_route)[i].first;
            base_layer_t<value_type>* layer = (base_layer_t<value_type>*) net_layers->find(layer_id)->second;
            if (layer->get_layer_structure() == MIMO) {
                std::vector<tensor_t<value_type>*> t_ins = ((base_structure_t<value_type>*)layer)->get_inputs();
                for (int j = 0; j < t_ins.size(); j++) {
                    // printf("%d t_in%d\n", i, t_ins[j]->get_tensor_id());
                    if (t_ins[j]->get_position() == SHARED_GPU_POOL) {
                        this->prefetch_tensors.push_back(t_ins[j]);
                    }
                }
            }
            else {  // SISO Layer
                // if (layer->get_layer_type() != DATA_L && layer->get_layer_type() != SOFTMAX && layer->get_layer_type() != CTCLOSS) {
                if (layer->get_layer_type() != DATA_L) {
                    tensor_t<value_type>* t_in = ((base_network_layer_t<value_type>*)layer)->get_f_in();
                    // printf("%d tensor%d\n", i, t_in->get_tensor_id());
                    if (t_in->get_position() == SHARED_GPU_POOL) {
                        this->prefetch_tensors.push_back(t_in);
                    }
                }
            }
        }
    }
    printf("offload tensor_list: ");
    for (int i = 0; i < swap_tensors.size(); i++) {
        printf("%d, ", swap_tensors[i]->get_tensor_id());
    }
    printf("total = %d\n", swap_tensors.size());
    printf("prefetch tensor_list: ");
    for (int i = 0; i < prefetch_tensors.size(); i++) {
        printf("%d, ", prefetch_tensors[i]->get_tensor_id());
    }
    printf("total = %d\n", prefetch_tensors.size());
    // for (size_t i = 0; i < swap_tensors.size(); i++) {
	// 	swap_tensors[i]->set_swap_block_id(i % SWAP_BLOCK_NUM);
    //     printf("swap_tensors%d will use block[%d]\n", swap_tensors[i]->get_tensor_id(), swap_tensors[i]->get_swap_block_id());
	// }
#endif
    printf("there are %d swap tensors size = %f, swap pool size = %f\n", swap_layers->size(), BYTE_TO_MB(sum), BYTE_TO_MB(min_pool_size));
    // mem_controller.init_swap_memory_pool(min_pool_size);
    return min_pool_size;
}

#ifdef MULTI_SWAP_BLOCK
template <class value_type>
void network_t<value_type>::swap_ctrl(tensor_t<value_type>* tensor, MODEL_TYPE model_type, net_comp dir, double *swap_ctrl_time, double *pur_swap_time) {
    bool flag = false;
    if (model_type == CNN_NETWORK) {
        if (dir == FORWARD) {
            // while(tensor->get_data_state() == NO_COMPUTED);
            if (swap_ctrl_time != NULL) {
                double start = get_cur_time();
                mem_controller.swap_ctrl(dir, model_type, tensor, pur_swap_time);
                *swap_ctrl_time = get_cur_time() - start;
            }
			else {
                mem_controller.swap_ctrl(dir, model_type, tensor, pur_swap_time);
            }
        } 
        else if (dir == BACKWARD) {  // BACKWARD
            if (swap_ctrl_time != NULL) {
                double start = get_cur_time();
                mem_controller.swap_ctrl(dir, model_type, tensor, pur_swap_time);
                *swap_ctrl_time = get_cur_time() - start;
            }
			else {
                mem_controller.swap_ctrl(dir, model_type, tensor, pur_swap_time);
            }
        } 
    }
    else {
        if (dir == FORWARD) {
            // while(tensor->get_data_state() == NO_COMPUTED);
            if (swap_ctrl_time != NULL) {
                double start = get_cur_time();
                mem_controller.swap_ctrl(dir, model_type, tensor, pur_swap_time);
                *swap_ctrl_time = get_cur_time() - start;
            }
			else {
                mem_controller.swap_ctrl(dir, model_type, tensor, pur_swap_time);
            }
        } 
        else if (dir == BACKWARD) {  // BACKWARD
            if (swap_ctrl_time != NULL) {
                double start = get_cur_time();
                mem_controller.swap_ctrl(dir, model_type, tensor, pur_swap_time);
                *swap_ctrl_time = get_cur_time() - start;
            }
			else {
                mem_controller.swap_ctrl(dir, model_type, tensor, pur_swap_time);
            }
        } 
    }
}
#endif

template <class value_type>
std::vector<double> network_t<value_type>::network_perf_profile() {
    std::vector<std::pair<int, net_comp> > net_comp_route = reg->get_net_comp_route();
    std::map<int, void* > net_layers  = reg->get_net_layers();
    double max_mem_used = 0;
    for(size_t i = 0; i < net_comp_route.size(); i++) {
        int layer_id = net_comp_route[i].first;
        net_comp dir = net_comp_route[i].second;
        // stash tensors
        std::pair<double, double> stat = mem_controller.stash_tensors_for_profile( layer_id, dir );
        double mem_stage_time =  stat.first;
        double total_mem      =  stat.second;
        double curt_mem = BYTE_TO_MB(query_used_mem());
        if(curt_mem > max_mem_used) max_mem_used = curt_mem;
        // execution
        base_layer_t<value_type>* b = (base_layer_t<value_type>*) net_layers.find(layer_id)->second;
        double start = get_cur_time();
        for(size_t j = 0; j < 10; j++) {
            if(dir == FORWARD) {
                b->forward(NET_TRAIN, &cublas_handle, &cudnn_handle, reg);
            } else if(dir == BACKWARD) {
                b->backward(NET_TRAIN, &cublas_handle, &cudnn_handle, reg);
            }
            //cudaStreamSynchronize(stream);
        }
        LAYER layer_type = b->get_layer_type();
        double end  = get_cur_time();
        double avg_time = (end - start)/10.0f;
        mem_controller.free_tensors_for_profile(layer_id, dir);
        double mem_time = mem_stage_time;
        if (dir == FORWARD) {
            printf("at layer id:%3d, type:%3d, compute_time:%5.5f, memory_time:%5.5f, total_mem:%5.5f\n", layer_id, layer_type, avg_time, mem_time, total_mem);
        } else {
            printf("at layer id:%3d, type:%3d, compute_time:%5.5f, memory_time:%5.5f, total_mem:%5.5f\n", layer_id*-1, layer_type, avg_time, mem_time, total_mem);
        }
    }
    printf("Max Memory used in profile:%f\n", max_mem_used);
    return std::vector<double>();
}

template <class value_type>
void network_t<value_type>::gradient_check_kernel(int l_id, size_t n, size_t c, size_t h, size_t w, tensor_t<value_type>* data, tensor_t<value_type>* diff, const char* str) {

    if( n >= data->get_N() ) {
        printf("n: %zu exceeds layer %d ~ N:%zu \n", n, l_id, data->get_N());
        return;
    }
    if( c >= data->get_C() ) {
        printf("c: %zu exceeds layer %d ~ C:%zu \n", c, l_id, data->get_C());
        return;
    }
    if( h >= data->get_H() ) {
        printf("h: %zu exceeds layer %d ~ H:%zu \n", h, l_id, data->get_H());
        return;
    }
    if( w >= data->get_W() ) {
        printf("w:%zu exceeds layer %d ~ W:%zu \n", w, l_id, data->get_W());
        return;
    }

    //analytical gradient
    value_type epsilon = 0.0001;
    value_type tmp     = data->get_scalar(n, c, h, w);
    data->set_scalar( n, c, h, w, tmp+epsilon );

    value_type loss1 = this->forward(NET_TRAIN);
    data->set_scalar( n, c, h, w, tmp-epsilon );
    value_type loss2 = this->forward(NET_TRAIN);

    value_type anly_grad = (loss1 - loss2)/(2*epsilon);
    //numerical gradient
    data->set_scalar( n, c, h, w, tmp); //set the weight
    this->forward(NET_TRAIN);
    this->backward();
    double num_grad = diff->get_scalar(n, c, h, w);

    printf("=> layer %d,n:%zu c:%zu h:%zu w:%zu num_grad:%f anly_grad:%f measure:%f loss1:%f loss2:%f %s\n", l_id, n, c, h, w, num_grad, anly_grad, math.abs_(anly_grad-num_grad)/math.max_(anly_grad, num_grad), loss1, loss2, str);
}

template <class value_type>
void network_t<value_type>::gradient_check(int layer_id) {
    tensor_t<value_type>* weight      = reg->get_reg_weight(layer_id);
    tensor_t<value_type>* weight_grad = reg->get_reg_weight_grad(layer_id);
    if(weight == NULL) {
        printf("layer:%d does not have params\n", layer_id);
        return;
    }
    size_t N = weight->get_N();
    size_t C = weight->get_C();
    size_t H = weight->get_H();
    size_t W = weight->get_W();

    for(size_t i = 0; i < N; i++) {
        for(size_t j = 0; j < C; j++) {
            for(size_t k = 0; k < H; k++) {
                for(size_t m = 0; m < W; m++) {
                    gradient_check_kernel( layer_id, i, j, k, m, weight, weight_grad, "weight");
                }
            }
        }
    }

    tensor_t<value_type>* bias        = reg->get_reg_bias(layer_id);
    tensor_t<value_type>* bias_grad   = reg->get_reg_bias_grad(layer_id);

    N = bias->get_N();
    C = bias->get_C();
    H = bias->get_H();
    W = bias->get_W();

    for(size_t i = 0; i < N; i++) {
        for(size_t j = 0; j < C; j++) {
            for(size_t k = 0; k < H; k++) {
                for(size_t m = 0; m < W; m++) {
                    gradient_check_kernel( layer_id, i, j, k, m, bias, bias_grad, "bias");
                }
            }
        }
    }
}

template <class value_type>
void network_t<value_type>::setup_test(base_layer_t<value_type>* test_data_layer, size_t iter) {

    if(!(this->is_forward_setup && this->is_backward_setup)) {
        printf("please setup the training before testing\n");
        exit(1);
    }

    this->test_iter              = iter;
    this->test_data_layer        = test_data_layer;

    test_data_layer->forward_setup (reg, &cudnn_handle);
    assert(this->test_data_layer != NULL);

    this->reg->register_net_layers(this->test_data_layer->get_base_id(), (void*) this->test_data_layer);
    this->reg->register_net_test_route(this->test_data_layer->get_base_id());
    this->reg->print_net_test_route();
    //points to the same layer, but the first network layer shall switch between these two
    this->is_testing_ready = true;
}

// #define DEBUG
template <class value_type>
void network_t<value_type>::simulated_forward_kernel(
	network_stage stage, base_layer_t<value_type>* b, std::vector<value_type>* loss,
	std::map<int, double> *layers_ft) {
	int layer_id;
    int exeflag = 0;		
	std::vector<std::pair<int, net_comp> >* net_comp_route = reg->get_net_comp_route_ptr();
    std::map<int, void* >* net_layers  = reg->get_net_layers_ptr();
    double start, end; 
    double start2, end2;
    LAYER layer_type;
    LAYER_STRUCTURE layer_structure;
    for(size_t i = 0; i < net_comp_route->size(); i++) {
        if((*net_comp_route)[i].second == FORWARD ) {
            exeflag = 1;
            // mem_controller.init_simulator_memory();
            start = get_cur_time();
            layer_id = (*net_comp_route)[i].first;
			// base_layer_t<value_type>* _b = (base_layer_t<value_type>*) net_layers.find(layer_id)->second;
            // stash tensors
#ifdef MALLOC_ALL
            printf("\nMALLOC_ALL\n");
            mem_controller.stash_tensor_malloc_all( layer_id, FORWARD , NET_TRAIN);
			// LRU 每当为特征图tensor申请了一个空间，就在lru表头插入该tensor
            // execution
            base_layer_t<value_type>* b = (base_layer_t<value_type>*) net_layers->find(layer_id)->second;
            printf("\nlayer_id=%d before b->forward(stage, &cublas_handle, &cudnn_handle, reg);\n", layer_id);
			// base_layer_t<value_type>* _b = (base_layer_t<value_type>*) net_layers.find(layer_id)->second;
            
            *loss = b->forward(stage, &cublas_handle, &cudnn_handle, reg);
            printf("\nlayer_id=%d after b->forward(stage, &cublas_handle, &cudnn_handle, reg);\n", layer_id);
            // update tensors
            mem_controller.update_tensor_state_free_all(layer_id, FORWARD, stage);
#else
            // printf("\nlayerid = %d before stash_tensor_shared_memory\n", layer_id);
			// LRU 每当为特征图tensor申请了一个空间，就在lru表头插入该tensor
            // executions			
            base_layer_t<value_type>* b = (base_layer_t<value_type>*) net_layers->find(layer_id)->second;
            layer_type = b->get_layer_type();
            layer_structure = b->get_layer_structure();
			mem_controller.stash_tensor_shared_memory( layer_id, FORWARD , NET_TRAIN);
            // printf("\nlayer_id=%d before b->forward(stage, &cublas_handle, &cudnn_handle, reg);\n", layer_id);
			start2 = get_cur_time();
            *loss = b->forward(stage, &cublas_handle, &cudnn_handle, reg);
			cudaStreamSynchronize(stream_singleton::get_compute_stream());
            end2 = get_cur_time();
            // if(layer_id == 1) printf("layer1 time = %lf\n", end2 - start2);
            // printf("\nlayer_id=%d after b->forward(stage, &cublas_handle, &cudnn_handle, reg) done;\n", layer_id);
            // update tensors
            mem_controller.update_tensor_state_shared_memory(layer_id, FORWARD, stage);
            end = get_cur_time();
            // if (cur_swap_time_test_times < max_swap_time_test_times) {
            //     test_output_swap_time(b, FORWARD);
            //     // test_output_swap_time_v2(b, FORWARD);
            // }
            layer_type_ft[layer_type] = layer_type_ft[layer_type] + (end - start);
            layer_type_num[layer_type] = layer_type_num[layer_type] + 1;
            if (layer_structure == SISO) {
                // printf("layer_structure[%d]\n", layer_structure);
                layer_type_mem[layer_type] = layer_type_mem[layer_type]
                    + (((base_network_layer_t<value_type>*)b)->get_f_out())->get_mem_size()/1024.0/1024.0;
            }
            else {
                
                std::vector<tensor_t<value_type>*> t_outs = ((base_structure_t<value_type>*)b)->get_outputs();
                for (int j = 0; j < t_outs.size(); j++) {  // fork_layer has only one input tensor
					layer_type_mem[layer_type] = layer_type_mem[layer_type] + t_outs[j]->get_mem_size()/1024.0/1024.0;  // fork_layer has only one input tensor
                }
            }
            // printf("layer_type_ft[%d] = %lf\n", layer_type, layer_type_ft[layer_type]);
			// printf("layer%d simulated_forward_kernel cost %lf\n", layer_id, end - start);
            
            // if (_b->get_layer_type() == DATA_L) {
            //     // DATA层不需要记录前向传播
            //     continue;
            // }
#endif
			// printf("forward finish layer %zu %d %d\n", i, layer_id, b->get_layer_type());
            // printf("layer_structure[%d]\n", layer_structure);
#ifdef DEBUG
            printf("forward finish layer %zu %d\n", i, layer_id);
            
#endif
        }
        if ((this->abandon <= 0)&&(exeflag==1)) {
            exeflag = 0;
            (*layers_ft)[layer_id] = end - start;
            (this->layers_ft)[layer_id] += end - start;
            (this->layers_ft2)[layer_id] += end2 - start2;
            this->f_time += end - start;
        }
    }
    printf("forward pure compute_time = %lf ", this->f_time);
    this->f_time = 0;
} 
// #undef DEBUG

// #define DEBUG
template <class value_type>
void network_t<value_type>::forward_kernel(network_stage stage, base_layer_t<value_type>* b, std::vector<value_type>* loss) {
    std::vector<std::pair<int, net_comp> >* net_comp_route = reg->get_net_comp_route_ptr();
    std::map<int, void* >* net_layers  = reg->get_net_layers_ptr();
    int layer_id;
    int exeflag = 0;
    double start, end; 
    double start2, end2;
	this->f_time = 0;
    for(size_t i = 0; i < net_comp_route->size(); i++) {
        
        if( (*net_comp_route)[i].second == FORWARD ) {
            
            exeflag = 1;
            layer_id = (*net_comp_route)[i].first;
            // stash tensors
			base_layer_t<value_type>* b = (base_layer_t<value_type>*) net_layers->find(layer_id)->second;
#ifdef DEBUG
            printf("forward start layer %zu %d %d\n", i, layer_id, b->get_layer_type());
#endif
            // printf("%zu forward start layer%d-type%d\n", i, layer_id, b->get_layer_type());
            mem_controller.stash_tensor( layer_id, b, FORWARD , NET_TRAIN, &recompute_layers_stack); // define in network.h
			// LRU 每当为特征图tensor申请了一个空间，就在lru表头插入该tensor
            // execution
            start2 = get_cur_time();
            start = get_cur_time();
            *loss = b->forward(stage, &cublas_handle, &cudnn_handle, reg);
			// cudaStreamSynchronize(stream_singleton::get_compute_stream());
            end = get_cur_time();
			end2 = get_cur_time();
            // printf("r%zu forward computation done layer%d-type%d\n", i, layer_id, b->get_layer_type());
            mem_controller.update_tensor_state(layer_id, b, FORWARD, stage);
            // printf("%zu forward finish layer%d-%d\n", i, layer_id, b->get_layer_type());
			// printf("layer%d-type%d forward cost %lf\n", layer_id, b->get_layer_type(), end2 - start2);
            // printf("forward finish layer %zu %d %d\n", i, layer_id, b->get_layer_type());
            // printf("forward finish layer %d %d\n", layer_id, b->get_layer_type());
#ifdef DEBUG
            printf("forward finish layer %zu %d %d\n", i, layer_id, b->get_layer_type());
#endif
        }
        
        if ((this->abandon <= 0)&&(exeflag==1)) {
            exeflag = 0;
			layers_ft[layer_id] += end - start;
            layers_ft2[layer_id] += end2 - start2;
			this->f_time += end - start;
        }
        
    }
    // cudaStreamSynchronize(stream_singleton::get_compute_stream());
    // cudaStreamSynchronize(stream_singleton::get_cpu2gpu_stream());
	printf("forward pure compute_time = %lf ", this->f_time);
    this->f_time = 0;
}
// #undef DEBUG

template <class value_type>
void network_t<value_type>::backward_with_update_kernel(base_layer_t<value_type>* l, size_t iter) {
    std::vector<std::pair<int, net_comp> > net_comp_route = reg->get_net_comp_route();
    std::map<int, void* > net_layers  = reg->get_net_layers();
    for( size_t i = 0; i < net_comp_route.size(); i++ ) {
        if( net_comp_route[i].second == BACKWARD ) {
            int layer_id = net_comp_route[i].first;
            // get tensors
            // execution
            base_layer_t<value_type>* b = (base_layer_t<value_type>*) net_layers.find(layer_id)->second;
			mem_controller.stash_tensor( layer_id, b, BACKWARD , NET_TRAIN, &recompute_layers_stack);
            b->backward(NET_TRAIN, &cublas_handle, &cudnn_handle, reg);
            // when backward finish, do the update immediately
            /*
            tensor_t<value_type>* weight_grad = this->reg->get_reg_weight_grad( layer_id );
            if( weight_grad != NULL ) {
                weight_grad->clip(this->get_cublas_handle());
                std::string idx = std::to_string( i );
                std::string filename = "gradient" + idx;
                weight_grad->writeToFile(filename.c_str());
            }
            */
            b->update(&cublas_handle, iter, solver);
            // update tensors
            mem_controller.update_tensor_state(layer_id, b, BACKWARD, NET_TRAIN);

#ifdef DEBUG
            printf("backward finish layer %d\n", layer_id);
#endif
        }
    }
    mem_controller.reset_tensor_state();
}
    
template <class value_type>
void network_t<value_type>::update_kernel(base_layer_t<value_type>* b, size_t iter) {
    for(size_t i = 0; i < reg->get_net_comp_route().size(); i++) {
        if( reg->get_net_comp_route()[i].second == FORWARD ) {
            int key = reg->get_net_comp_route()[i].first;
            base_layer_t<value_type>* b = (base_layer_t<value_type>*)reg->get_net_layers().find(key)->second;
            b->update(&cublas_handle, iter, solver);
            // printf("update_kernel layer %d\n", b->get_base_id());
        }
    }
}

template <class value_type>
void network_t<value_type>::grad_zero() {
    for(size_t i = 0; i < reg->get_net_comp_route().size(); i++) {
        if( reg->get_net_comp_route()[i].second == FORWARD ) {
            int key = reg->get_net_comp_route()[i].first;
            base_layer_t<value_type>* layer = (base_layer_t<value_type>*)reg->get_net_layers().find(key)->second;
            layer->grad_zero(&cublas_handle, solver);
        }
    }
    // auto net_layers = reg->get_net_layers();
    // for(auto it = net_layers.begin(); it != net_layers.end(); it++) {
    //     base_layer_t<value_type>* layer = (base_layer_t<value_type> *) it->second;
    //     layer->grad_zero(&cublas_handle, solver);
    // }
}

// #define DEBUG
template <class value_type>
void network_t<value_type>::simulated_backward_kernel(
	base_layer_t<value_type>* b, 
	std::map<int, double> *layers_bt) {
    int layer_id;
    int exeflag = 0;
    std::vector<std::pair<int, net_comp> >* net_comp_route = reg->get_net_comp_route_ptr();
    std::map<int, void* >* net_layers = reg->get_net_layers_ptr();
    double start, end; 
    double start2, end2;
    LAYER layer_type;
    LAYER_STRUCTURE layer_structure;
    // double ss, ee; 
    for( size_t i = 0; i < net_comp_route->size(); i++ ) {
        
        if( (*net_comp_route)[i].second == BACKWARD ) {
            exeflag = 1;
            layer_id = (*net_comp_route)[i].first;
			base_layer_t<value_type>* b = (base_layer_t<value_type>*) net_layers->find(layer_id)->second;
			if (b->get_layer_type() == DATA_L) {
                // DATA层不需要反向传播
                // printf("\nlayerid = %d before stash_tensor_shared_memory\n", layer_id);
				(*layers_bt)[layer_id] = 0;
                continue;
            }
#ifdef MALLOC_ALL
            mem_controller.stash_tensor_malloc_all( layer_id, BACKWARD, NET_TRAIN);
            b->backward(NET_TRAIN, &cublas_handle, &cudnn_handle, reg);
            mem_controller.update_tensor_state_free_all(layer_id, BACKWARD, NET_TRAIN);
#else
            // get tensors
            // printf("\nlayerid = %d before stash_tensor_shared_memory\n", layer_id);
            mem_controller.stash_tensor_shared_memory(layer_id, BACKWARD , NET_TRAIN);
            // printf("\nlayer_id=%d before b->backward(stage, &cublas_handle, &cudnn_handle, reg);\n", layer_id);
            start2 = get_cur_time();
            start = get_cur_time();
            b->backward(NET_TRAIN, &cublas_handle, &cudnn_handle, reg);
            layer_type = b->get_layer_type();
            layer_structure = b->get_layer_structure();
			cudaStreamSynchronize(stream_singleton::get_compute_stream());
            end2 = get_cur_time();
            // printf("\nlayer_id=%d after b->backward(stage, &cublas_handle, &cudnn_handle, reg);\n", layer_id);
            mem_controller.update_tensor_state_shared_memory(layer_id, BACKWARD, NET_TRAIN);
            end = get_cur_time();
            // if (cur_swap_time_test_times < max_swap_time_test_times) {
            //     test_output_swap_time(b, BACKWARD);
            //     // test_output_swap_time_v2(b, BACKWARD);
            // }
            layer_type_bt[layer_type] = layer_type_bt[layer_type] + (end - start);
            // layer_type_num[layer_type] = layer_type_num[layer_type] + 1;
            if (layer_structure == SISO) {
                layer_type_b_mem[layer_type] = layer_type_b_mem[layer_type]
                    + (((base_network_layer_t<value_type>*)b)->get_b_data())->get_mem_size()/1024.0/1024.0;
            }
            else {
                std::vector<tensor_t<value_type>*> b_data = ((base_structure_t<value_type>*)b)->get_b_data();
                for (int j = 0; j < b_data.size(); j++) {  // fork_layer has only one input tensor
					layer_type_b_mem[layer_type] = layer_type_b_mem[layer_type] + b_data[j]->get_mem_size()/1024.0/1024.0;  // fork_layer has only one input tensor
                }
            }
#endif
        #ifdef DEBUG
            printf("backward finish layer %zu %d %d\n", i, layer_id, b->get_layer_type());
        #endif
        }
        
        if ((i != 0)&&(exeflag==1)) {
            if (this->abandon <= 0) {
                exeflag = 0;
                (*layers_bt)[layer_id] = end - start;
                // (this->layers_bt)[layer_id] += end - start;
                // (this->layers_bt2)[layer_id] += end2 - start2;
                // layer_type_bt[layer_type] = layer_type_bt[layer_type] + (this->layers_bt2)[layer_id];
                this->b_time += end - start;
            }
		}
    }
    printf("   backward pure compute_time = %lf\n", this->b_time);
    this->b_time = 0;
    // backward_last_time = get_cur_time();
    // value_type sum = reg->get_grad_sqrsum();
    // average_backward_last_time += get_cur_time() - backward_last_time;
    //LOG(INFO)<<"###grad sum"<<sum<<"sqrted:"<<std::sqrt(sum);
}
// #undef DEBUG

// #define DEBUG
template <class value_type>
void network_t<value_type>::recompute_kernel(std::map<int, void* >* net_layers, base_layer_t<value_type>* layer, net_comp dir) {
    // printf("dir%d: recompute layer%d-type%d\n", dir, layer->get_base_id(), layer->get_layer_type());
    int recompute_flag = mem_controller.stash_tensor( layer->get_base_id(), layer, dir, NET_TRAIN, &recompute_layers_stack);
    // printf("recompute_flag = %d\n", recompute_flag);
    if (recompute_flag) {
        while (!recompute_layers_stack.empty()) {
            // printf("recompute_layers_stack.size = %d\n", recompute_layers_stack.size());
            base_layer_t<value_type>* recompute_layer = (base_layer_t<value_type>*) net_layers->find(recompute_layers_stack.top())->second;
            recompute_layers_stack.pop();
            recompute_kernel(net_layers, recompute_layer, RECOMPUTE);
        }
    }
    if (dir == RECOMPUTE) {
        // printf("start recompute recompute_layer%d-type%d\n", dir, layer->get_base_id(), layer->get_layer_type());
        layer->forward(NET_TRAIN, &cublas_handle, &cudnn_handle, reg);
        cudaStreamSynchronize(stream_singleton::get_compute_stream());
        // printf("end recompute_layer dir%d: recompute layer%d-type%d\n", dir, layer->get_base_id(), layer->get_layer_type());
    }
    else if (dir == BACKWARD) {
        // printf("start backward dir%d: backward_layer%d-type%d\n", dir, layer->get_base_id(), layer->get_layer_type());
        // printf("start backward dir%d: backward_layer%d-type%d\n", dir, layer->get_base_id(), layer->get_layer_type());
        layer->backward(NET_TRAIN, &cublas_handle, &cudnn_handle, reg);
        cudaStreamSynchronize(stream_singleton::get_compute_stream());
        // printf("end backward_layer dir%d: backward layer%d-type%d\n", dir, layer->get_base_id(), layer->get_layer_type());
    }
    mem_controller.update_tensor_state(layer->get_base_id(), layer, dir, NET_TRAIN);
    // #ifdef DEBUG
    //             int test_layer_id = 1877;
    //             base_layer_t<value_type>* test_layer = (base_layer_t<value_type>*) net_layers->find(test_layer_id)->second;
    //             tensor_t<value_type>* test_tensor = ((base_network_layer_t<value_type>*)test_layer)->get_f_out();
    //             test_tensor->printTensorData("test_tensor recompute_kernel 2", 2);
    //             test_tensor->printTensorData("test_tensor recompute_kernel 2", 2);
    //             test_tensor->printTensorData("test_tensor recompute_kernel 2", 2);
    //             #endif
}
// #undef DEBUG

// #define DEBUG
template <class value_type>
void network_t<value_type>::backward_kernel(base_layer_t<value_type>* b) {
    std::vector<std::pair<int, net_comp> >* net_comp_route = reg->get_net_comp_route_ptr();
    std::map<int, void* >* net_layers  = reg->get_net_layers_ptr();
    double start, end; 
    double start2, end2;
    int exeflag = 0;
    int layer_id;
	
    for( size_t i = 0; i < net_comp_route->size(); i++ ) {
        start = get_cur_time();
        int test_layer_id = 1877;
        base_layer_t<value_type>* test_layer;
        tensor_t<value_type>* test_tensor;
        // #ifdef DEBUG
        //     test_layer = (base_layer_t<value_type>*) net_layers->find(test_layer_id)->second;
        //     test_tensor = ((base_network_layer_t<value_type>*)test_layer)->get_f_out();
        //     test_tensor->printTensorData("test_tensor backward_kernel 1", 2);
        // #endif
        if( (*net_comp_route)[i].second == BACKWARD ) {
            // #ifdef DEBUG
            //     int test_layer_id = 1877;
            //     base_layer_t<value_type>* test_layer = (base_layer_t<value_type>*) net_layers->find(test_layer_id)->second;
            //     tensor_t<value_type>* test_tensor = ((base_network_layer_t<value_type>*)test_layer)->get_f_out();
            //     test_tensor->printTensorData("test_tensor 777", 2);
            // #endif
            // printf("backward start layer%d-type%d\n", b->get_base_id(), b->get_layer_type());
            exeflag = 1;
            // start = get_cur_time();
            layer_id = (*net_comp_route)[i].first;
            base_layer_t<value_type>* b = (base_layer_t<value_type>*) net_layers->find(layer_id)->second;
            
            #ifdef DEBUG
                printf("%d backward start1 layer%d-type%d\n", i, b->get_base_id(), b->get_layer_type());
            #endif
			// if ( b->get_layer_type() == DATA_L ) {
			// 	continue;
			// }
			// printf("layer%d backward start\n", layer_id);
    #ifdef RECOMPUTE_ON
            start = get_cur_time();
            recompute_kernel(net_layers, b, BACKWARD);
            end = get_cur_time();
            // #ifdef DEBUG
            //     test_layer = (base_layer_t<value_type>*) net_layers->find(test_layer_id)->second;
            //     test_tensor = ((base_network_layer_t<value_type>*)test_layer)->get_f_out();
            //     test_tensor->printTensorData("test_tensor backward_kernel 2", 2);
            // #endif
    #else
            mem_controller.stash_tensor( layer_id, b, BACKWARD, NET_TRAIN, &recompute_layers_stack);
			start2 = get_cur_time();
            // printf("%d backward start2 layer%d-type%d\n", i, b->get_base_id(), b->get_layer_type());
            b->backward(NET_TRAIN, &cublas_handle, &cudnn_handle, reg);
			// cudaStreamSynchronize(stream_singleton::get_compute_stream());
            // printf("%d backward finish2 layer%d-type%d\n", i, b->get_base_id(), b->get_layer_type());
            end2 = get_cur_time();
            mem_controller.update_tensor_state(layer_id, b, BACKWARD, NET_TRAIN);
    #endif
            
			// layers_bt[layer_id] += end - start;
			// this->b_time += layers_bt[layer_id];
			// printf("layer%d-type%d backward cost %lf\n", layer_id, b->get_layer_type(), end - start);
            // printf("%d backward finish layer%d-type%d\n", i, b->get_base_id(), b->get_layer_type());
            
    #ifdef DEBUG
            printf("%d backward finish2 layer%d-type%d time = %lf\n", i, b->get_base_id(), b->get_layer_type(), end - start);
    #endif
            
        }
        end = get_cur_time();
        // printf("layer%d-type%d backward cost %lf\n", layer_id, b->get_layer_type(), end - start);
        if ((i != 0)&&(exeflag==1)) {
            if (this->abandon <= 0) {
                exeflag = 0;
                // (*layers_bt)[layer_id] = end - start;
                (this->layers_bt)[layer_id] += end - start;
                (this->layers_bt2)[layer_id] += end2 - start2;
                // this->b_time += end2 - start2;
                this->b_time += end - start;
            }
		}
        
    }
    // cudaStreamSynchronize(stream_singleton::get_compute_stream());
    mem_controller.reset_swap_block();
	printf("   backward pure compute_time = %lf\n", this->b_time);
    this->b_time = 0;
    // value_type sum = reg->get_grad_sqrsum();
    //LOG(INFO)<<"###grad sum"<<sum<<"sqrted:"<<std::sqrt(sum);
}
// #undef DEBUG

template <class value_type>
void network_t<value_type>::forward_test(network_stage stage, base_layer_t<value_type>* b, std::vector<value_type>* acc) {
    
//NEED REPLACE the data layer in net_comp_route!!!!
    std::vector<std::pair<int, net_comp> > net_test_route = reg->get_net_test_route();
    std::map<int, void* > net_layers  = reg->get_net_layers();
    
    for(size_t i = 0; i < net_test_route.size(); i++) {
        if( net_test_route[i].second == FORWARD ) {
            // get the necessary tensors sorted out before calling forward
            int layer_id = net_test_route[i].first;
            // stash tensors
            
            base_layer_t<value_type>* b = (base_layer_t<value_type>*) net_layers.find(layer_id)->second;
			mem_controller.stash_tensor( layer_id, b, FORWARD , NET_INFER, &recompute_layers_stack);
            *acc = b->forward(stage, &cublas_handle, &cudnn_handle, reg);
            
            // update tensors
            mem_controller.update_tensor_state(layer_id, b, FORWARD, stage);
        }
    }     
    mem_controller.reset_tensor_state();
}

// fsetup_kernel 可获得前向传播的执行顺序net_comp_route，以及每一层依赖 的tensor
template <class value_type>
void network_t<value_type>::fsetup_kernel(base_layer_t<value_type>* b) {
    if(b == NULL) return;
	// printf("layer%d next layer are ", b->get_base_id());
	
    //conduct forward computation
    b->fcounter_inc();  // 每次进入setup层计数加1
	// 当层的计数少于
    if(b->get_fcounter() < b->get_prev_size() ) {
        return;
    }
	// 配置该层的cudnn相关参数
	// 设置层依赖的tensor，把依赖的tensor加入forward_dependency和forward_dependency_by_tensor
    b->forward_setup(reg, &cudnn_handle);
    // b->increase_input_use_counter(FORWARD);
	// 把（层id，该层本身）加入网络层列表
    this->reg->register_net_layers(b->get_base_id(), (void*) b);
	// 把（层id，前向）加入net_comp_route列表
    this->reg->register_net_comp_route(b->get_base_id(), FORWARD);
    // 取下一层，下一层也是一列表，以递归方式注册
    std::vector<base_layer_t<value_type>*> next = b->get_next();
	for(size_t i = 0; i < next.size(); i++) {
		// printf("%d ", next[i]->get_base_id());
	}
	// printf("\n");
    if(next.size() == 1) {
        //regular network layer
        fsetup_kernel(next[0]);
    } else if(next.size() > 1) {
        //fork layer 的前向传播存在分支
		for(size_t i = 0; i < next.size(); i++) {
            fsetup_kernel(next[i]);
        }
		/*
        for(size_t i = 1; i < next.size(); i++) {
            fsetup_kernel(next[i]);
        }
        fsetup_kernel(next[0]);
		*/	
    }
    b->reset_fc_counter();  // 当该层de的所有next层都注册完了，重置层计数
}

// bsetup_kernel只建立反向传播顺序，不修改层列表net_layers
template <class value_type>
void network_t<value_type>::bsetup_kernel(base_layer_t<value_type>* b) {
    if(b == NULL) return;
    //conduct forward computation
    b->fcounter_inc();
#ifdef DEBUG
    printf("@layer %p:%d fcounter:%zu get_next_size:%zu \n", b, b->get_base_id(), b->get_fcounter(), b->get_next_size());
#endif

    if(b->get_fcounter() < b->get_next_size() ) {
        return;
    }
    b->backward_setup(reg, &cudnn_handle);
    // b->increase_input_use_counter(BACKWARD);
    b->increase_dy_use_counter();
    this->reg->register_net_comp_route(b->get_base_id(), BACKWARD);
    std::vector<base_layer_t<value_type>*> prev = b->get_prev();
    if(prev.size() == 1) {
        //regular network layer
        bsetup_kernel(prev[0]);
    } else if(prev.size() > 1) {
        //fork layer
		for(size_t i = 0; i < prev.size(); i++) {
            // bsetup_kernel(prev[i]);
			bsetup_kernel(prev[prev.size()-i-1]);  // 尝试反过来注册，使前向和反向的顺序一致
        }
		/*
        for(size_t i = 1; i < prev.size(); i++) {
            // bsetup_kernel(prev[i]);
			bsetup_kernel(prev[prev.size()-i]);  // 尝试反过来注册，使前向和反向的顺序一致
        }
        bsetup_kernel(prev[0]);
		*/
    }
    b->reset_fc_counter();
}


template <class value_type>
void network_t<value_type>::test() {
    assert( this->test_data_layer != NULL );
    assert( this->train_data_layer != NULL );
    //let's swap the head of data layer
    //please note prev matters!!!
    //the first network layer will need prev to figure out the input
    base_layer_t<value_type>* train_l = this->train_data_layer;
    base_layer_t<value_type>* test_l  = this->test_data_layer;
    std::vector<base_layer_t<value_type>*> next_ls = train_l->get_next();
    assert(next_ls.size() == 1);
    base_layer_t<value_type>* next_l = next_ls[0];
    next_l->switch_prev_l_to(test_l);

    value_type cumulative_acc_top1 = 0;
    value_type cumulative_acc_top5 = 0;
    base_layer_t<value_type>* start = this->test_data_layer;

    for(size_t i = 0; i < this->test_iter; i++) {
        checkCudaErrors( cudaDeviceSynchronize() );
        std::vector<value_type> tmp;
        forward_test(NET_INFER, start, &tmp);
        cumulative_acc_top1 += tmp[0];
        cumulative_acc_top5 += tmp[1];
    }

    value_type test_accuracy_top1 = cumulative_acc_top1 / (value_type) this->test_iter;
    value_type test_accuracy_top5 = cumulative_acc_top5 / (value_type) this->test_iter;

    printf("-------test accuracy--top 1 %f top 5 %f-------\n", test_accuracy_top1, test_accuracy_top5);
    next_l->switch_prev_l_to(train_l);

}

INSTANTIATE_CLASS(network_t);

} //SuperNeuron namespace
