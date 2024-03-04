 
#include <ATPSearch.h>
#include <util/mem_util.h>

namespace ATP {

template <class value_type>
void ATPSearch<value_type>::printGenerationTime() {
	// std::map<int, void* > net_layers  = reg->get_net_layers();
	// for (auto iter = net_layers.begin(); iter != net_layers.end(); iter++) {
	// 	base_layer_t<value_type>* layer = (base_layer_t<value_type>*)iter->second;
	// }
}

template <class value_type>
void ATPSearch<value_type>::ResetSimulateTrainMemory() {
	mem_controller->reset_all_gpu_mem_pool();
}

template <class value_type>
size_t ATPSearch<value_type>::GetMaxThroughputBatchSize() {
	size_t selected_batchsize = 0;
	double max_throughput = 0;
	for (auto it = this->ideal_throughput_by_batchsize.begin(); it != this->ideal_throughput_by_batchsize.end(); it++) {
		printf("batchsize = %d, ThroughputUpperBound = %lf\n", it->first, it->second); 
		if (max_throughput < it->second) {
			selected_batchsize = it->first;
			max_throughput = it->second;
		}
	}
	printf("Best: batchsize = %d, ThroughputUpperBound = %lf\n", selected_batchsize, max_throughput); 
	return selected_batchsize;
}

template <class value_type>
void ATPSearch<value_type>::ResetThroughputModelLastRecord() {
	this->swap_tensors.clear();
	this->prefetch_tensors.clear();
	this->alternative_swap_tensors.clear();
	this->recompute_tensors.clear();
	this->layers_ft.clear();
	this->layers_bt.clear();
	this->layers_output_size.clear();
	printf("After ResetThroughputModelLastRecord, swap_tensors.size = %d, prefetch_tensors.size = %d, alternative_swap_tensors.size = %d, recompute_tensors.size = %d\n",
		this->swap_tensors.size(), this->swap_tensors.size(), this->swap_tensors.size(), this->swap_tensors.size());
}

template <class value_type>
void ATPSearch<value_type>::GetRecomputeSwapTensorScheme( ) {
	// std::vector<tensor_t<value_type>*> swap_tensors;
	// std::vector<tensor_t<value_type>*> prefetch_tensors;
	// size_t swapping_size = SetSwappingTensors(swapping_code, &(this->alternative_swap_tensors), NULL, NULL);
	std::map<int, void* > layers = reg->get_net_layers();
	std::vector<std::pair<int, net_comp> > net_route = reg->get_net_comp_route();
	printf("layers size = %d:\n", layers.size());
	size_t re_size = 0;
	size_t sw_size = 0;
	size_t code_size = 0;
	size_t max_swap_size = 0;
	size_t min_swap_size = 34359738368;
	size_t max_recomp_size = 0;
	size_t min_recomp_size = 34359738368;
	for (size_t i = 0; i < layers.size(); i++) {  // set rs_code_route
		if (net_route[i].second == FORWARD) {
			base_layer_t<value_type>* layer = (base_layer_t<value_type>*) layers.find(net_route[i].first)->second;
			if (layer->get_layer_structure() == SISO) {
				if (layer->get_layer_type() == RNN || layer->get_layer_type() == SATTN) {
					// tensor_t<value_type>* reserve_buff = ((rnn_layer_t<value_type>*)layer)->get_rnn_reserve_buff();
					tensor_t<value_type>* reserve_buff = ((base_network_layer_t<value_type>*)layer)->get_reserve_buff();
					if (reserve_buff->get_position() == SHARED_GPU_POOL) {
						printf("%d,", 1);
						sw_size += reserve_buff->get_mem_size();
						if (reserve_buff->get_mem_size() > max_swap_size) {
							max_swap_size = reserve_buff->get_mem_size();
						}
						if (reserve_buff->get_mem_size() < min_swap_size) {
							min_swap_size = reserve_buff->get_mem_size();
						}
					}
					else {
						printf("%d,", 0);
					}
					code_size++;
				}
				tensor_t<value_type>* t_out = ((base_network_layer_t<value_type>*)layer)->get_f_out();
				if (t_out->get_position() == SHARED_GPU_POOL) {
					printf("%d,", 1);
					sw_size += t_out->get_mem_size();
					if (t_out->get_mem_size() > max_swap_size) {
						max_swap_size = t_out->get_mem_size();
					}
					if (t_out->get_mem_size() < min_swap_size) {
						min_swap_size = t_out->get_mem_size();
					}
				}
				else if (t_out->get_position() == RECOMPUTE_IN_BACKWARD) {
					printf("%d,", 2);
					re_size += t_out->get_mem_size();
					if (t_out->get_mem_size() > max_recomp_size) {
						max_recomp_size = t_out->get_mem_size();
					}
					if (t_out->get_mem_size() < min_recomp_size) {
						min_recomp_size = t_out->get_mem_size();
					}
				}
				else {
					printf("%d,", 0);
				}
				code_size++;
			}
			else {
				std::vector<tensor_t<value_type>*> t_outs = ((base_structure_t<value_type>*)layer)->get_outputs();
				for (int j = 0; j < t_outs.size(); j++) {
					if (t_outs[j]->get_position() == SHARED_GPU_POOL) {
						printf("%d,", 1);
						sw_size += t_outs[j]->get_mem_size();
						if (t_outs[j]->get_mem_size() > max_swap_size) {
							max_swap_size = t_outs[j]->get_mem_size();
						}
						if (t_outs[j]->get_mem_size() < min_swap_size) {
							min_swap_size = t_outs[j]->get_mem_size();
						}
					}
					else if (t_outs[j]->get_position() == RECOMPUTE_IN_BACKWARD) {
						printf("%d,", 2);
						re_size += t_outs[j]->get_mem_size();
						if (t_outs[j]->get_mem_size() > max_recomp_size) {
							max_recomp_size = t_outs[j]->get_mem_size();
						}
						if (t_outs[j]->get_mem_size() < min_recomp_size) {
							min_recomp_size = t_outs[j]->get_mem_size();
						}
					}
					else {
						printf("%d,", 0);
					}
					code_size++;
				}
			}
		}
	}
	printf("\nCODE size = %d, re_size = %.2f, sw_size = %.2f\n", code_size, BYTE_TO_MB(re_size), BYTE_TO_MB(sw_size));
	if (min_recomp_size == 34359738368) min_recomp_size = 0;
	if (min_swap_size == 34359738368) min_swap_size = 0;
	size_t max_fragment_size = (SWAP_BLOCK_NUM * (max_swap_size - min_swap_size));// + (RECOMPUTE_POOL_NUM * (max_recomp_size - min_recomp_size));
	printf("\nMaxium Fragment Size = %zd, %fMB, %f GB\n\n", max_fragment_size, BYTE_TO_MB(max_fragment_size), BYTE_TO_MB(max_fragment_size)/1024.0);
	size_t j = 0;
	printf("GetRecomputeSwapTensorGACode done %d\n", j);
}

template <class value_type>
bool ATPSearch<value_type>::ThroughputUpperBound_v2(size_t batch_size, size_t* MR, size_t* MS, 
		std::vector<std::pair<tensor_t<value_type>*, GENERATE_EFFICIENCY*>>* gf_list, std::vector<tensor_t<value_type>*>* recompute_tensors, std::vector<tensor_t<value_type>*>* alternative_swap_tensors,
		double* tpub, double* iter_time, double* recomp_time, double* ideal_swap_time, double* tf, double* tb, double* f_time, double* b_time) { 
	size_t ME = excessive_size_by_batchsize[batch_size];  // minimal swapping size //
	double _tf = 0.0f;
	double _tb = 0.0f;
	double tp = 0.0f;
	bool re_flag = false;
	size_t _MR = *MR;
	size_t _MS = *MS;
	for (auto it = this->layers_ft.begin(); it != this->layers_ft.end(); it++) {
		_tf += it->second;
	}
	for (auto it = this->layers_bt.begin(); it != this->layers_bt.end(); it++) {
		_tb += it->second;
	}
	*tf = _tf;
	*tb = _tb;
	if (*MR < 512) {  // 512 is the minimal unit size of GPU memory,  MR < 512 means no tensor need to be recomputed.
		tp = (double)batch_size / (_tf + _tb);
		*recomp_time = 0.0;
		*f_time = ((double)*MS)/pcie_bandwith > _tf ? ((double)*MS)/pcie_bandwith : _tf;
		*b_time = ((double)*MS)/pcie_bandwith > _tb ? ((double)*MS)/pcie_bandwith : _tb;
		tp = (batch_size) / (*f_time + *b_time);
		*iter_time = *f_time + *b_time;
		*ideal_swap_time = 2.0  * ((double)*MS)/pcie_bandwith;
		*tpub = tp;
		printf("Batch size = %d, Case 1: ThroughputUpperBound = %f, ME = MS = %fMB, ideal swapping time < tf = %lf\n", batch_size, *tpub, BYTE_TO_MB(*MS), *tf);
		re_flag = true;
	}
	else {  // recomputation is used
		printf("MR = %f\n", BYTE_TO_MB(*MR));
		if (SelectRecomputingTensor_v2(MR, gf_list, recompute_tensors)) {
			printf("when batch size = %zd, SelectRecomputingTensor success, MR = %f, MS = %f\n", batch_size, BYTE_TO_MB(*MR), BYTE_TO_MB(*MS));
		}
		else {
			*MS = ME - *MR;
			printf("when batch size = %zd, SelectRecomputingTensor(MR=%f, MS=%f) fail, the total size of recompute_tensors is not enough because of some illegal recompute_tensors, the new MR=%f, new MS=%f\n", 
				batch_size, BYTE_TO_MB(_MR), BYTE_TO_MB(_MS), BYTE_TO_MB(*MR), BYTE_TO_MB(*MS));
		}
		*ideal_swap_time = 2.0  * ((double)(*MS))/pcie_bandwith;
		double _tr = 0.0;
		for (size_t i = 0; i < recompute_tensors->size(); i++) {  // total recomputating time
			size_t layer_id = (*recompute_tensors)[i]->get_layer_id();
			_tr += layers_ft[layer_id];
		}
		*recomp_time = _tr;
		*f_time = ((double)*MS)/pcie_bandwith > _tf ? ((double)*MS)/pcie_bandwith : _tf;
		*b_time = ((double)*MS)/pcie_bandwith > (_tb + _tr) ? ((double)*MS)/pcie_bandwith : (_tb + _tr);
		tp = ((double)batch_size) / (*f_time + *b_time);
		*iter_time = *f_time + *b_time;
		*ideal_swap_time = 2.0  * ((double)*MS)/pcie_bandwith;
		printf("Batch size = %d, Case 2: ThroughputUpperBound = %lf, tf = %lf, tb = %lf, tr = %lf, ideal_total_swapping_time = %lf\n", 
				batch_size, tp, *tf, *tb, *recomp_time, *ideal_swap_time);
		*tpub = tp;
		re_flag =  true;
	}
	if (re_flag == true) {
		GetAlternativeSwappingTensors(gf_list, &(this->recompute_tensors), &(this->alternative_swap_tensors));
		GetRecomputeSwapTensorScheme();
	}
	else return false;
	// GetAlternativeSwappingTensors
}


template <class value_type>
void ATPSearch<value_type>::GetMaxGenerationEfficiency(tensor_t<value_type>* tensor, GENERATE_EFFICIENCY* max_generation_efficiency) {
	double swap_efficiency = (double)(tensor->get_mem_size()) / tensor->get_swap_time(OFFLOAD);
	double recompute_efficiency = (double)(tensor->get_mem_size()) / layers_ft[tensor->get_layer_id()];
	// printf("layer%d tensor%d swap_time = %lf  recomputation_time = %lf\n", tensor->get_layer_id(), tensor->get_tensor_id(), swap_time, recomputation_time);
	if (recompute_efficiency > swap_efficiency) {
		max_generation_efficiency->operation = RECOMPUTED;
		max_generation_efficiency->generation_efficiency = recompute_efficiency;
	}
	else {
		max_generation_efficiency->operation = SWAPPED;
		max_generation_efficiency->generation_efficiency = swap_efficiency;
	}	
}

template <class value_type>
void ATPSearch<value_type>::QuickSortGenerationEfficiency(std::vector<std::pair<tensor_t<value_type>*, GENERATE_EFFICIENCY*>>* gf_list, int low, int high) {
	int left = low;
	int right = high;
	if(left >= right) {
        return;
    }
	std::pair<tensor_t<value_type>*, GENERATE_EFFICIENCY*> key = (*gf_list)[low];
	while (left < right) {
		while ((*gf_list)[right].second->generation_efficiency <= key.second->generation_efficiency && left < right) {
			right--;
		}
		// printf("after right--, left=%d, right=%d\n", left, right);
		(*gf_list)[left] = (*gf_list)[right];
		while ((*gf_list)[left].second->generation_efficiency >= key.second->generation_efficiency && left < right) {
			left++; 
		}
		// printf("after left++, left=%d, right=%d\n", left, right);
		(*gf_list)[right] = (*gf_list)[left];
	}
	(*gf_list)[right] = key;  // right == left
	for (int i = 0; i < gf_list->size(); i++) {
		// printf("%d-%lf ", i, (*gt_list)[i].second->min_generation_time);
	}
	// printf("after that, left=%d, right=%d\n", left, right);
	QuickSortGenerationEfficiency(gf_list, low, left-1);
	// printf("return a QuickSortGenerationTime, left=%d, right=%d\n", left, right);
	QuickSortGenerationEfficiency(gf_list, left+1, high);
	// printf("return a QuickSortGenerationTime, left=%d, right=%d\n", left, right);
}

// #define DEBUG
template <class value_type>
void ATPSearch<value_type>::SortGenerationEfficiencyList(std::vector<std::pair<tensor_t<value_type>*, GENERATE_EFFICIENCY*>>* gf_list) {
		std::map<int, void* > layers = reg->get_net_layers();
	std::vector<std::pair<int, net_comp> > net_route = reg->get_net_comp_route();
	tensor_t<value_type>* tensor;
	std::vector<tensor_t<value_type>*> tensors;
	std::vector<tensor_t<value_type>*> gt_list_tensors;
	size_t data_sum = 0;
	auto iter = gt_list_tensors.begin();
	// printf("SortGenerationTimeList layer_num = %d\n", layers.size());
	int j = 0;
	// for (int i = SWAP_BLOCK_NUM; i < net_route.size()-layers.size()-SWAP_BLOCK_NUM-1; i++) {
	for (int i = 0; i < net_route.size(); i++) {
		if (net_route[i].second == FORWARD) {
			base_layer_t<value_type>* layer = (base_layer_t<value_type>*) layers.find(net_route[i].first)->second;
			if ((i <= SWAP_BLOCK_NUM) || (i >= layers.size()-1-SWAP_BLOCK_NUM)) {  // the first and last few tensors should be swapped
				if (layer->get_layer_structure() == SISO) {
					if (layer->get_layer_type() == DATA_L || layer->get_layer_type() == SOFTMAX) {
						tensor = ((base_network_layer_t<value_type>*)layer)->get_f_out();
						GENERATE_EFFICIENCY* max_generation_efficiency = new GENERATE_EFFICIENCY{REMAINED, 0.0};
						iter = std::find(gt_list_tensors.begin(), gt_list_tensors.end(), tensor);
						if (iter == gt_list_tensors.end()) {  // Prevent duplicate addition
							gt_list_tensors.push_back(tensor);
							gf_list->push_back(std::pair<tensor_t<value_type>*, GENERATE_EFFICIENCY*>(tensor, max_generation_efficiency));
							data_sum += tensor->get_mem_size();
							j++;
						}
						// printf("sss tensor%d-layer%d-type%d operate=%d min_generation_time=%lf\n", 
						// 		tensor->get_tensor_id(), tensor->get_layer_id(), ((base_layer_t<value_type>*)layers.find(net_route[i].first)->second)->get_layer_type(), min_generation->pre_operate, min_generation->min_generation_time);	
					}
					else {
						if (layer->get_layer_type() == RNN || layer->get_layer_type() == SATTN) {
							// tensor = ((rnn_layer_t<value_type>*)layer)->get_rnn_reserve_buff();	
							tensor = ((base_network_layer_t<value_type>*)layer)->get_reserve_buff();																																																							
							GENERATE_EFFICIENCY* max_generation_efficiency = new GENERATE_EFFICIENCY{SWAPPED, (double)(tensor->get_mem_size())/tensor->get_swap_time(OFFLOAD)};
							iter = std::find(gt_list_tensors.begin(), gt_list_tensors.end(), tensor);
							if (iter == gt_list_tensors.end()) {  // Prevent duplicate addition
								gt_list_tensors.push_back(tensor);
								gf_list->push_back(std::pair<tensor_t<value_type>*, GENERATE_EFFICIENCY*>(tensor, max_generation_efficiency));
								data_sum += tensor->get_mem_size();
								j++;
							}
							// printf("ddd reserve_buff%d-layer%d-type%d operate=%d min_generation_time=%lf\n", 
							// reserve_buff->get_tensor_id(), reserve_buff->get_layer_id(), ((base_layer_t<value_type>*)layers.find(net_route[i].first)->second)->get_layer_type(), min_generation->pre_operate, min_generation->min_generation_time);
							tensor = ((base_network_layer_t<value_type>*)layer)->get_f_out();
							max_generation_efficiency = new GENERATE_EFFICIENCY{SWAPPED, (double)(tensor->get_mem_size())/tensor->get_swap_time(OFFLOAD)};
							iter = std::find(gt_list_tensors.begin(), gt_list_tensors.end(), tensor);
							if (iter == gt_list_tensors.end()) {  // Prevent duplicate addition
								gt_list_tensors.push_back(tensor);
								gf_list->push_back(std::pair<tensor_t<value_type>*, GENERATE_EFFICIENCY*>(tensor, max_generation_efficiency));
								data_sum += tensor->get_mem_size();
								j++;
							}
							// printf("rrr tensor%d-layer%d-type%d operate=%d min_generation_time=%lf\n", 
							// 		tensor->get_tensor_id(), tensor->get_layer_id(), ((base_layer_t<value_type>*)layers.find(net_route[i].first)->second)->get_layer_type(), min_generation->pre_operate, min_generation->min_generation_time);
							
						}
						else {
							tensor = ((base_network_layer_t<value_type>*)layer)->get_f_out();
							GENERATE_EFFICIENCY* max_generation_efficiency = new GENERATE_EFFICIENCY{SWAPPED, (double)(tensor->get_mem_size())/tensor->get_swap_time(OFFLOAD)};
							// GENERATE_EFFICIENCY* max_generation_efficiency = new GENERATE_EFFICIENCY{REMAINED, 0.0};
							// GetMaxGenerationEfficiency(tensor, max_generation_efficiency);
							iter = std::find(gt_list_tensors.begin(), gt_list_tensors.end(), tensor);
							if (iter == gt_list_tensors.end()) {  // Prevent duplicate addition
								gt_list_tensors.push_back(tensor);
								gf_list->push_back(std::pair<tensor_t<value_type>*, GENERATE_EFFICIENCY*>(tensor, max_generation_efficiency));
								data_sum += tensor->get_mem_size();
								j++;
							}
							// printf("rrr tensor%d-layer%d-type%d operate=%d min_generation_time=%lf\n", 
							// 		tensor->get_tensor_id(), tensor->get_layer_id(), ((base_layer_t<value_type>*)layers.find(net_route[i].first)->second)->get_layer_type(), min_generation->pre_operate, min_generation->min_generation_time);
						}	
					}
				}
				else {  // MIMO
					if (layer->get_layer_type() == FORK_L) {  // since fork's input == output and it has been set operation in other layer, so we should not consider its' operation 
						// tensors = ((base_structure_t<value_type>*)layer)->get_outputs();
						// GENERATE_EFFICIENCY *max_generation_efficiency = new GENERATE_EFFICIENCY{SWAPPED, (double)(tensors[0]->get_mem_size())/tensors[0]->get_swap_time(OFFLOAD)};
						// gf_list->push_back(std::pair<tensor_t<value_type>*, GENERATE_EFFICIENCY*>(tensors[0], max_generation_efficiency));
						// // printf("tensor%d-layer%d-type%d operate=%d min_generation_time=%lf\n", 
						// // 		tensor->get_tensor_id(), tensor->get_layer_id(), ((base_layer_t<value_type>*)layers.find(net_route[i].first)->second)->get_layer_type(), min_generation->pre_operate, min_generation->min_generation_time);
						// j++;
					}
					else if (layer->get_layer_type() == CONCAT || layer->get_layer_type() == JOIN_L) {
						tensors = ((base_structure_t<value_type>*)layer)->get_outputs();
						for (int k = 0; k < tensors.size(); k++) {
							GENERATE_EFFICIENCY *max_generation_efficiency = new GENERATE_EFFICIENCY{SWAPPED, (double)(tensors[k]->get_mem_size())/tensors[k]->get_swap_time(OFFLOAD)};
							// GENERATE_EFFICIENCY *max_generation_efficiency = new GENERATE_EFFICIENCY{REMAINED, (double)(tensors[k]->get_mem_size())/tensors[k]->get_swap_time(OFFLOAD)};
							iter = std::find(gt_list_tensors.begin(), gt_list_tensors.end(), tensors[k]);
							if (iter == gt_list_tensors.end()) {  // Prevent duplicate addition
								gt_list_tensors.push_back(tensors[k]);
								gf_list->push_back(std::pair<tensor_t<value_type>*, GENERATE_EFFICIENCY*>(tensors[k], max_generation_efficiency));
								data_sum += tensor->get_mem_size();
								j++;
							}
							// printf("tensor%d-layer%d-type%d operate=%d min_generation_time=%lf\n", 
							// 		tensor->get_tensor_id(), tensor->get_layer_id(), ((base_layer_t<value_type>*)layers.find(net_route[i].first)->second)->get_layer_type(), min_generation->pre_operate, min_generation->min_generation_time);
						}
					}
					else {
						continue;
					}
				}
			}
			else {
				// printf("layer%d-type%d\n", layer->get_base_id(), layer->get_layer_type());
				if (layer->get_layer_type() != JOIN_L && layer->get_layer_type() != FORK_L && layer->get_layer_type() != CONCAT 
					&& layer->get_layer_type() != DATA_L && layer->get_layer_type() != SOFTMAX) {
					if (layer->get_layer_type() == RNN || layer->get_layer_type() == SATTN) {
						// tensor = ((rnn_layer_t<value_type>*)layer)->get_rnn_reserve_buff();
						tensor = ((base_network_layer_t<value_type>*)layer)->get_reserve_buff();
						GENERATE_EFFICIENCY *max_generation_efficiency = new GENERATE_EFFICIENCY{SWAPPED, tensor->get_comm_time()};
						iter = std::find(gt_list_tensors.begin(), gt_list_tensors.end(), tensor);
						if (iter == gt_list_tensors.end()) {  // Prevent duplicate addition
							gt_list_tensors.push_back(tensor);
							gf_list->push_back(std::pair<tensor_t<value_type>*, GENERATE_EFFICIENCY*>(tensor, max_generation_efficiency));
							data_sum += tensor->get_mem_size();
							j++;
						}
						// printf("tensor%d-layer%d-type%d operate=%d min_generation_time=%lf\n", 
						// 	tensor->get_tensor_id(), tensor->get_layer_id(), ((base_layer_t<value_type>*)layers.find(net_route[i].first)->second)->get_layer_type(), min_generation->pre_operate, min_generation->min_generation_time);
			
						tensor = ((base_network_layer_t<value_type>*)layer)->get_f_out();
						max_generation_efficiency = new GENERATE_EFFICIENCY{SWAPPED, tensor->get_comm_time()};
						iter = std::find(gt_list_tensors.begin(), gt_list_tensors.end(), tensor);
						if (iter == gt_list_tensors.end()) {  // Prevent duplicate addition
							gt_list_tensors.push_back(tensor);
							gf_list->push_back(std::pair<tensor_t<value_type>*, GENERATE_EFFICIENCY*>(tensor, max_generation_efficiency));
							data_sum += tensor->get_mem_size();
							j++;
						}
						// printf("rrr tensor%d-layer%d-type%d operate=%d min_generation_time=%lf\n", 
						// 		tensor->get_tensor_id(), tensor->get_layer_id(), ((base_layer_t<value_type>*)layers.find(net_route[i].first)->second)->get_layer_type(), min_generation->pre_operate, min_generation->min_generation_time);
					}
					else {
						tensor = ((base_network_layer_t<value_type>*)layer)->get_f_out();
						GENERATE_EFFICIENCY *max_generation_efficiency = new GENERATE_EFFICIENCY{REMAINED, 0.0};
						GetMaxGenerationEfficiency(tensor, max_generation_efficiency);
						iter = std::find(gt_list_tensors.begin(), gt_list_tensors.end(), tensor);
						if (iter == gt_list_tensors.end()) {  // Prevent duplicate addition
							gt_list_tensors.push_back(tensor);
							gf_list->push_back(std::pair<tensor_t<value_type>*, GENERATE_EFFICIENCY*>(tensor, max_generation_efficiency));
							data_sum += tensor->get_mem_size();
							j++;
						}	
						// printf("tensor%d-layer%d-type%d operate=%d min_generation_time=%lf\n", 
						// 		tensor->get_tensor_id(), tensor->get_layer_id(), ((base_layer_t<value_type>*)layers.find(net_route[i].first)->second)->get_layer_type(), min_generation->pre_operate, min_generation->min_generation_time);	
					}	
				}
				else if (layer->get_layer_type() == JOIN_L || layer->get_layer_type() == CONCAT) {
					tensors = ((base_structure_t<value_type>*)layer)->get_outputs();
					for (int k = 0; k < tensors.size(); k++) {
						GENERATE_EFFICIENCY *max_generation_efficiency = new GENERATE_EFFICIENCY{SWAPPED, (double)(tensors[k]->get_mem_size())/tensors[k]->get_swap_time(OFFLOAD)};
						// GENERATE_EFFICIENCY *max_generation_efficiency = new GENERATE_EFFICIENCY{REMAINED, (double)(tensors[k]->get_mem_size())/tensors[k]->get_swap_time(OFFLOAD)};
						iter = std::find(gt_list_tensors.begin(), gt_list_tensors.end(), tensors[k]);
						if (iter == gt_list_tensors.end()) {  // Prevent duplicate addition
							gt_list_tensors.push_back(tensors[k]);
							gf_list->push_back(std::pair<tensor_t<value_type>*, GENERATE_EFFICIENCY*>(tensors[k], max_generation_efficiency));
							data_sum += tensor->get_mem_size();
							j++;
						}
						// printf("tensor%d-layer%d-type%d operate=%d min_generation_time=%lf\n", 
						// 		tensor->get_tensor_id(), tensor->get_layer_id(), ((base_layer_t<value_type>*)layers.find(net_route[i].first)->second)->get_layer_type(), min_generation->pre_operate, min_generation->min_generation_time);	
					}
				}
				else if (layer->get_layer_type() == FORK_L) {  // since fork's input == output and it has been set operation in other layer, so we should not consider its' operation 
					// tensors = ((base_structure_t<value_type>*)layer)->get_outputs();
					// GENERATE_EFFICIENCY *max_generation_efficiency = new GENERATE_EFFICIENCY{SWAPPED, (double)(tensors[0]->get_mem_size())/tensors[0]->get_swap_time(OFFLOAD)};
					// gf_list->push_back(std::pair<tensor_t<value_type>*, GENERATE_EFFICIENCY*>(tensors[0], max_generation_efficiency));
					// // printf("tensor%d-layer%d-type%d operate=%d min_generation_time=%lf\n", 
					// // 		tensor->get_tensor_id(), tensor->get_layer_id(), ((base_layer_t<value_type>*)layers.find(net_route[i].first)->second)->get_layer_type(), min_generation->pre_operate, min_generation->min_generation_time);
					// j++;
				}
				else {
					continue;
				}
			}
		}
	}
	printf("data_sum = %f\n", BYTE_TO_MB(data_sum));
	#ifdef DEBUG
		printf("gf_list->size() = %d\n", gf_list->size());
		for (int i = 0; i < gf_list->size(); i++) {
			printf("tensor%d-type%d-layer%d-type%d operation=%d max_generation_efficiency=%lf\n", 
				(*gf_list)[i].first->get_tensor_id(), (*gf_list)[i].first->get_type(), (*gf_list)[i].first->get_layer_id(), ((base_layer_t<value_type>*)layers.find((*gf_list)[i].first->get_layer_id())->second)->get_layer_type(), 
				(*gf_list)[i].second->operation, (*gf_list)[i].second->generation_efficiency);
		}
	#endif
	// while (1);
	double ts = get_cur_time();
	QuickSortGenerationEfficiency(gf_list, 0, gf_list->size()-1);
	double time = get_cur_time() - ts;
	
	#ifdef DEBUG
		for (int i = 0; i < gf_list->size(); i++) {
			printf("rank%d tensor%d-type%d-layer%d-type%d operate=%d max_generation_efficiency=%lf size=%f\n", 
				i, (*gf_list)[i].first->get_tensor_id(), (*gf_list)[i].first->get_type(), (*gf_list)[i].first->get_layer_id(), ((base_layer_t<value_type>*)layers.find((*gf_list)[i].first->get_layer_id())->second)->get_layer_type(), 
				(*gf_list)[i].second->operation, (*gf_list)[i].second->generation_efficiency, BYTE_TO_MB((*gf_list)[i].first->get_mem_size()));
		}
	#endif
	// while(1);
	// printf("time = %lf\n", time);
	// return recomputation_time < swap_time ? recomputation_time : swap_time;
}
// #undef DEBUG

template <class value_type>
void ATPSearch<value_type>::GetMinGenerationTime(tensor_t<value_type>* tensor, MIN_GENERATE* min_generation) {
	double swap_time = tensor->get_comm_time();
	double recomputation_time = layers_ft[tensor->get_layer_id()];
	// printf("layer%d tensor%d swap_time = %lf  recomputation_time = %lf\n", tensor->get_layer_id(), tensor->get_tensor_id(), swap_time, recomputation_time);
	if (recomputation_time < swap_time) {
		min_generation->pre_operate = RECOMPUTED;
		min_generation->min_generation_time = recomputation_time;
	}
	else {
		min_generation->pre_operate = SWAPPED;
		min_generation->min_generation_time = swap_time;
	}							
}

template <class value_type>
void ATPSearch<value_type>::QuickSortGenerationTime(std::vector<std::pair<tensor_t<value_type>*, MIN_GENERATE*>>* gt_list, int low, int high) {
	int left = low;
	int right = high;
	if(left >= right) {
        return;
    }
	std::pair<tensor_t<value_type>*, MIN_GENERATE*> key = (*gt_list)[low];
	while (left < right) {
		while ((*gt_list)[right].second->min_generation_time >= key.second->min_generation_time && left < right) {
			right--;
		}
		// printf("after right--, left=%d, right=%d\n", left, right);
		(*gt_list)[left] = (*gt_list)[right];
		while ((*gt_list)[left].second->min_generation_time <= key.second->min_generation_time && left < right) {
			left++; 
		}
		// printf("after left++, left=%d, right=%d\n", left, right);
		(*gt_list)[right] = (*gt_list)[left];
	}
	(*gt_list)[right] = key;  // right == left
	for (int i = 0; i < gt_list->size(); i++) {
		// printf("%d-%lf ", i, (*gt_list)[i].second->min_generation_time);
	}
	// printf("after that, left=%d, right=%d\n", left, right);
	QuickSortGenerationTime(gt_list, low, left-1);
	// printf("return a QuickSortGenerationTime, left=%d, right=%d\n", left, right);
	QuickSortGenerationTime(gt_list, left+1, high);
	// printf("return a QuickSortGenerationTime, left=%d, right=%d\n", left, right);
}

// #define DEBUG
template <class value_type>
void ATPSearch<value_type>::SortGenerationTimeList(std::vector<std::pair<tensor_t<value_type>*, MIN_GENERATE*>>* gt_list) {
	std::map<int, void* > layers = reg->get_net_layers();
	std::vector<std::pair<int, net_comp> > net_route = reg->get_net_comp_route();
	tensor_t<value_type>* tensor;
	// printf("SortGenerationTimeList layer_num = %d\n", layers.size());
	int j = 0;
	// for (int i = SWAP_BLOCK_NUM; i < net_route.size()-layers.size()-SWAP_BLOCK_NUM-1; i++) {
	for (int i = 0; i < net_route.size(); i++) {
		if (net_route[i].second == FORWARD) {
			base_layer_t<value_type>* layer = (base_layer_t<value_type>*) layers.find(net_route[i].first)->second;
			if ((i <= SWAP_BLOCK_NUM) || (i >= layers.size()-1-SWAP_BLOCK_NUM)) {  // the first and last few tensors should be swapped
				if (layer->get_layer_structure() == SISO) {
					if (layer->get_layer_type() == DATA_L || layer->get_layer_type() == SOFTMAX) {
						tensor = ((base_network_layer_t<value_type>*)layer)->get_f_out();
						MIN_GENERATE *min_generation = new MIN_GENERATE{REMAINED, 0.0};
						gt_list->push_back(std::pair<tensor_t<value_type>*, MIN_GENERATE*>(tensor, min_generation));
						// printf("sss tensor%d-layer%d-type%d operate=%d min_generation_time=%lf\n", 
						// 		tensor->get_tensor_id(), tensor->get_layer_id(), ((base_layer_t<value_type>*)layers.find(net_route[i].first)->second)->get_layer_type(), min_generation->pre_operate, min_generation->min_generation_time);
						j++;
					}
					else {
						if (layer->get_layer_type() == RNN || layer->get_layer_type() == SATTN) {
							// tensor_t<value_type>* reserve_buff = ((rnn_layer_t<value_type>*)layer)->get_rnn_reserve_buff();	
							tensor_t<value_type>* reserve_buff = ((base_network_layer_t<value_type>*)layer)->get_reserve_buff();																																																							
							MIN_GENERATE *min_generation = new MIN_GENERATE{SWAPPED, reserve_buff->get_comm_time()};
							gt_list->push_back(std::pair<tensor_t<value_type>*, MIN_GENERATE*>(reserve_buff, min_generation));
							// printf("ddd reserve_buff%d-layer%d-type%d operate=%d min_generation_time=%lf\n", 
							// 	reserve_buff->get_tensor_id(), reserve_buff->get_layer_id(), ((base_layer_t<value_type>*)layers.find(net_route[i].first)->second)->get_layer_type(), min_generation->pre_operate, min_generation->min_generation_time);
							j++;

							tensor_t<value_type>* tensor = ((base_network_layer_t<value_type>*)layer)->get_f_out();
							min_generation = new MIN_GENERATE{SWAPPED, tensor->get_comm_time()};
							gt_list->push_back(std::pair<tensor_t<value_type>*, MIN_GENERATE*>(tensor, min_generation));
							// printf("rrr tensor%d-layer%d-type%d operate=%d min_generation_time=%lf\n", 
							// 		tensor->get_tensor_id(), tensor->get_layer_id(), ((base_layer_t<value_type>*)layers.find(net_route[i].first)->second)->get_layer_type(), min_generation->pre_operate, min_generation->min_generation_time);
							j++;
						}
						else {
							MIN_GENERATE *min_generation = new MIN_GENERATE{SWAPPED, 0.0};
							tensor_t<value_type>* tensor = ((base_network_layer_t<value_type>*)layer)->get_f_out();
							GetMinGenerationTime(tensor, min_generation);
							gt_list->push_back(std::pair<tensor_t<value_type>*, MIN_GENERATE*>(tensor, min_generation));
							// printf("rrr tensor%d-layer%d-type%d operate=%d min_generation_time=%lf\n", 
							// 		tensor->get_tensor_id(), tensor->get_layer_id(), ((base_layer_t<value_type>*)layers.find(net_route[i].first)->second)->get_layer_type(), min_generation->pre_operate, min_generation->min_generation_time);
							j++;
						}
						
					}
				}
				else {
					tensor = ((base_structure_t<value_type>*)layer)->get_outputs()[0];
					MIN_GENERATE *min_generation = new MIN_GENERATE{SWAPPED, tensor->get_comm_time()};
					gt_list->push_back(std::pair<tensor_t<value_type>*, MIN_GENERATE*>(tensor, min_generation));
					// printf("tensor%d-layer%d-type%d operate=%d min_generation_time=%lf\n", 
					// 		tensor->get_tensor_id(), tensor->get_layer_id(), ((base_layer_t<value_type>*)layers.find(net_route[i].first)->second)->get_layer_type(), min_generation->pre_operate, min_generation->min_generation_time);
					j++;
				}
			}
			else {
				// printf("layer%d-type%d\n", layer->get_base_id(), layer->get_layer_type());
				if (layer->get_layer_type() != JOIN_L && layer->get_layer_type() != FORK_L && layer->get_layer_type() != CONCAT 
					&& layer->get_layer_type() != DATA_L && layer->get_layer_type() != SOFTMAX) {
					if (layer->get_layer_type() == RNN || layer->get_layer_type() == SATTN) {
						// tensor_t<value_type>* reserve_buff = ((rnn_layer_t<value_type>*)layer)->get_rnn_reserve_buff();
						tensor_t<value_type>* reserve_buff = ((base_network_layer_t<value_type>*)layer)->get_reserve_buff();
						MIN_GENERATE *min_generation = new MIN_GENERATE{SWAPPED, reserve_buff->get_comm_time()};
						gt_list->push_back(std::pair<tensor_t<value_type>*, MIN_GENERATE*>(reserve_buff, min_generation));
						// printf("tensor%d-layer%d-type%d operate=%d min_generation_time=%lf\n", 
						// 	tensor->get_tensor_id(), tensor->get_layer_id(), ((base_layer_t<value_type>*)layers.find(net_route[i].first)->second)->get_layer_type(), min_generation->pre_operate, min_generation->min_generation_time);
						j++;

						tensor_t<value_type>* tensor = ((base_network_layer_t<value_type>*)layer)->get_f_out();
						min_generation = new MIN_GENERATE{SWAPPED, tensor->get_comm_time()};
						gt_list->push_back(std::pair<tensor_t<value_type>*, MIN_GENERATE*>(tensor, min_generation));
						// printf("rrr tensor%d-layer%d-type%d operate=%d min_generation_time=%lf\n", 
						// 		tensor->get_tensor_id(), tensor->get_layer_id(), ((base_layer_t<value_type>*)layers.find(net_route[i].first)->second)->get_layer_type(), min_generation->pre_operate, min_generation->min_generation_time);
						j++;
					}
					else {
						tensor = ((base_network_layer_t<value_type>*)layer)->get_f_out();
						MIN_GENERATE *min_generation = new MIN_GENERATE{REMAINED, 0.0};
						GetMinGenerationTime(tensor, min_generation);
						gt_list->push_back(std::pair<tensor_t<value_type>*, MIN_GENERATE*>(tensor, min_generation));
						// printf("tensor%d-layer%d-type%d operate=%d min_generation_time=%lf\n", 
						// 		tensor->get_tensor_id(), tensor->get_layer_id(), ((base_layer_t<value_type>*)layers.find(net_route[i].first)->second)->get_layer_type(), min_generation->pre_operate, min_generation->min_generation_time);
						j++;
					}
					
				}
				else if (layer->get_layer_type() == JOIN_L || layer->get_layer_type() == FORK_L || layer->get_layer_type() == CONCAT) {
					tensor = ((base_structure_t<value_type>*)layer)->get_outputs()[0];
					MIN_GENERATE *min_generation = new MIN_GENERATE{SWAPPED, tensor->get_comm_time()};
					gt_list->push_back(std::pair<tensor_t<value_type>*, MIN_GENERATE*>(tensor, min_generation));
					// printf("tensor%d-layer%d-type%d operate=%d min_generation_time=%lf\n", 
					// 		tensor->get_tensor_id(), tensor->get_layer_id(), ((base_layer_t<value_type>*)layers.find(net_route[i].first)->second)->get_layer_type(), min_generation->pre_operate, min_generation->min_generation_time);
					j++;
				}
				else {
					continue;
				}
			}
		}
	}
	#ifdef DEBUG
		printf("gt_list->size() = %d\n", gt_list->size());
		for (int i = 0; i < gt_list->size(); i++) {
			printf("tensor%d-type%d-layer%d-type%d operate=%d min_generation_time=%lf\n", 
				(*gt_list)[i].first->get_tensor_id(), (*gt_list)[i].first->get_type(), (*gt_list)[i].first->get_layer_id(), ((base_layer_t<value_type>*)layers.find((*gt_list)[i].first->get_layer_id())->second)->get_layer_type(), 
				(*gt_list)[i].second->pre_operate, (*gt_list)[i].second->min_generation_time);
		}
	#endif
	// while (1);
	double ts = get_cur_time();
	QuickSortGenerationTime(gt_list, 0, gt_list->size()-1);
	double time = get_cur_time() - ts;
	
	#ifdef DEBUG
		for (int i = 0; i < gt_list->size(); i++) {
			printf("rank%d tensor%d-type%d-layer%d-type%d operate=%d min_generation_time=%lf size=%f\n", 
				i, (*gt_list)[i].first->get_tensor_id(), (*gt_list)[i].first->get_type(), (*gt_list)[i].first->get_layer_id(), ((base_layer_t<value_type>*)layers.find((*gt_list)[i].first->get_layer_id())->second)->get_layer_type(), 
				(*gt_list)[i].second->pre_operate, (*gt_list)[i].second->min_generation_time, BYTE_TO_MB((*gt_list)[i].first->get_mem_size()));
		}
	#endif
	// while(1);
	// printf("time = %lf\n", time);
	// return recomputation_time < swap_time ? recomputation_time : swap_time;
}
// #undef DEBUG

template <class value_type>
bool ATPSearch<value_type>::MinSavedMem(size_t batch_size) {
	size_t ME = excessive_size_by_batchsize[batch_size];  // minimal swapping size 
	size_t mr = 0;
	size_t ms1 = pcie_bandwith * forward_computing_time_by_batchsize[batch_size] * 1.0;  // the swapping size which can be covered by forward propagation
	size_t ms2 = 0;
	size_t savable_size = 0;
	size_t recomptable_size = 0;
	size_t swappable_size = 0;
	size_t max_output_tensor_size = 0;
	size_t min_output_tensor_size = 11811160064;
	std::map<int, void* > layers = reg->get_net_layers();
	base_layer_t<value_type>* layer;
	if (ME == 0) {
		printf("Can be trained directly, ME = %zd\n", ME);
		return true;
	}
	else {
		for (auto it = layers.begin(); it != layers.end(); it++) {
			layer = (base_layer_t<value_type>*)(it->second);
			if (layer->get_layer_type() != DATA_L && layer->get_layer_type() != SOFTMAX) {
				if (layer->get_layer_structure() == MIMO) {
					std::vector<tensor_t<value_type>* > tensors = ((base_structure_t<value_type>*)layer)->get_outputs();
					for (int i = 0; i < tensors.size(); i++) {
						savable_size += tensors[i]->get_mem_size();
						if (max_output_tensor_size < tensors[i]->get_mem_size()) {
							max_output_tensor_size = tensors[i]->get_mem_size();
						}
						if (min_output_tensor_size > tensors[i]->get_mem_size()) {
							min_output_tensor_size = tensors[i]->get_mem_size();
						}
					}
				}
				else {
					tensor_t<value_type>* tensor = ((base_network_layer_t<value_type>*)layer)->get_f_out();
					savable_size += tensor->get_mem_size();
					if (max_output_tensor_size < tensor->get_mem_size()) {
						max_output_tensor_size = tensor->get_mem_size();
					}
					if (min_output_tensor_size > tensor->get_mem_size()) {
						min_output_tensor_size = tensor->get_mem_size();
					}
				}
			}
		}
		savable_size -= ((SWAP_BLOCK_NUM * max_output_tensor_size) + (RECOMPUTE_POOL_NUM * max_output_tensor_size));
		// recomptable_size -= RECOMPUTE_POOL_NUM * max_output_tensor_size;
		// swappable_size -= SWAP_BLOCK_NUM * max_output_tensor_size;
		printf("savable_size=%f, max_output_tensor_size=%f, min_output_tensor_size==%f\n",
			BYTE_TO_MB(savable_size), BYTE_TO_MB(max_output_tensor_size), BYTE_TO_MB(min_output_tensor_size));
		if (ME > savable_size) {
			printf("Excessive size (%fMB) > savable size (%fMB), network can not be trained in this batchsize\n", BYTE_TO_MB(ME), BYTE_TO_MB(savable_size));
			return false;
		}
		else {
			return true;
		}
	}
}

template <class value_type>
bool ATPSearch<value_type>::DistributeMRMS_v2(size_t batch_size, std::vector<std::pair<tensor_t<value_type>*, GENERATE_EFFICIENCY*>>* gf_list, size_t* MR, size_t* MS) {
	size_t ME = excessive_size_by_batchsize[batch_size];  // minimal swapping size 
	size_t mr = 0;
	size_t ms1 = pcie_bandwith * forward_computing_time_by_batchsize[batch_size] * 1.0;  // the swapping size which can be covered by forward propagation
	size_t ms2 = 0;
	size_t savable_size = 0;
	size_t recomptable_size = 0;
	size_t swappable_size = 0;
	size_t max_swappable_tensor_size = 0;
	size_t min_swappable_tensor_size = 11811160064;
	size_t max_recomputable_tensor_size = 0;
	size_t min_recomputable_tensor_size = 11811160064;
	std::map<int, void* > layers = reg->get_net_layers();
	SortGenerationEfficiencyList(gf_list); 
	if (ME == 0) {
		*MR = *MS = 0;
		printf("Can be trained directly, ME = %zd\n", ME);
		return true;
	}
	else {
		for (size_t i = 0; i < gf_list->size(); i++) {
			if ((*gf_list)[i].second->operation != REMAINED) {
				tensor_t<value_type>* tensor = (*gf_list)[i].first;
				savable_size += tensor->get_mem_size();
				if ((*gf_list)[i].second->operation == SWAPPED) {
					swappable_size += tensor->get_mem_size();
					if (max_swappable_tensor_size < tensor->get_mem_size()) {
						max_swappable_tensor_size = tensor->get_mem_size();
					}
					if (min_swappable_tensor_size > tensor->get_mem_size()) {
						min_swappable_tensor_size = tensor->get_mem_size();
					}
				}
				else if ((*gf_list)[i].second->operation == RECOMPUTED) {
					recomptable_size += tensor->get_mem_size();
					if (max_recomputable_tensor_size < tensor->get_mem_size()) {
						max_recomputable_tensor_size = tensor->get_mem_size();
					}
					if (min_recomputable_tensor_size > tensor->get_mem_size()) {
						min_recomputable_tensor_size = tensor->get_mem_size();
					}
				}
			}
		}
		if (min_swappable_tensor_size == 11811160064) min_swappable_tensor_size = 0;
		if (min_recomputable_tensor_size == 11811160064) min_recomputable_tensor_size = 0;
		size_t max_fragment_size = (SWAP_BLOCK_NUM * (max_swappable_tensor_size - min_swappable_tensor_size)) + (RECOMPUTE_POOL_NUM * (max_recomputable_tensor_size - min_recomputable_tensor_size));
		printf("\nMaxium Fragment Size = %zd, %fMB, %f GB\n\n", max_fragment_size, BYTE_TO_MB(max_fragment_size), BYTE_TO_MB(max_fragment_size)/1024.0);
		// savable_size = savable_size - pool size

		// savable_size -= (SWAP_BLOCK_NUM * (max_swappable_tensor_size - min_swappable_tensor_size)) + (RECOMPUTE_POOL_NUM * (max_recomputable_tensor_size - min_recomputable_tensor_size));
		// recomptable_size -= RECOMPUTE_POOL_NUM * max_recomputable_tensor_size;
		// swappable_size -= SWAP_BLOCK_NUM * max_swappable_tensor_size;
		printf("savable_size=%f, recomptable_size=%f, swappable_size=%f, max_swappable_tensor_size=%f, min_swappable_tensor_size=%f, max_recomputable_tensor_size=%f, min_recomputable_tensor_size=%f\n",
			BYTE_TO_MB(savable_size), BYTE_TO_MB(recomptable_size), BYTE_TO_MB(swappable_size), BYTE_TO_MB(max_swappable_tensor_size), BYTE_TO_MB(min_swappable_tensor_size), BYTE_TO_MB(max_recomputable_tensor_size), BYTE_TO_MB(min_recomputable_tensor_size));
		if (ME > savable_size) {
			printf("Excessive size (%fMB) > savable size (%fMB), network can not be trained in this batchsize\n", BYTE_TO_MB(ME), BYTE_TO_MB(savable_size));
			
			return false;
		}
		else if (ms1 >= ME) {
			*MR = 0;
			*MS = ME;
			printf("Swapping only, ME = MS = %zd\n", ME);
			return true;
		}
		else {
			for (size_t i = 0; i < gf_list->size(); i++) {
				tensor_t<value_type>* tensor = (*gf_list)[i].first;
				if ((*gf_list)[i].second->operation == SWAPPED) {
					ms2 += tensor->get_mem_size();
				}
				else {  // gt_list[i].second->pre_operate == RECOMPUTED
					mr += tensor->get_mem_size();
				}
				if (ms2 + mr >= ME - ms1) break;
			}
			*MR = mr;
			*MS = ms1 + ms2;
			if (mr > recomptable_size) {
				printf("MR=%f > recomptable_size\n", BYTE_TO_MB(mr));
				*MR = recomptable_size;
				*MS += mr - recomptable_size;
			}
			printf("ME=%f, MR=%f, MS=%f, MS1=%f, MS2=%f \n", BYTE_TO_MB(ME), BYTE_TO_MB(*MR), BYTE_TO_MB(*MS), BYTE_TO_MB(ms1), BYTE_TO_MB(ms2));
			printf("MS + MR >= ME, Distribution Done!\n");
			return true;
		}
	}		
}

template <class value_type>
bool ATPSearch<value_type>::DistributeMSSwapOnly(size_t batch_size, std::vector<std::pair<tensor_t<value_type>*, GENERATE_EFFICIENCY*>>* gf_list, size_t* MR, size_t* MS) {
	size_t ME = excessive_size_by_batchsize[batch_size];  // minimal swapping size 
	size_t mr = 0;
	size_t ms1 = pcie_bandwith * forward_computing_time_by_batchsize[batch_size] * 1.0;  // the swapping size which can be covered by forward propagation
	size_t ms2 = 0;
	size_t savable_size = 0;
	size_t recomptable_size = 0;
	size_t swappable_size = 0;
	size_t max_swappable_tensor_size = 0;
	size_t min_swappable_tensor_size = 11811160064;
	size_t max_recomputable_tensor_size = 0;
	size_t min_recomputable_tensor_size = 11811160064;				
	std::map<int, void* > layers = reg->get_net_layers();
	SortGenerationEfficiencyList(gf_list); 
	if (ME == 0) {
		*MR = *MS = 0;
		printf("Can be trained directly, ME = %zd\n", ME);
		return true;
	}	
	else {
		for (size_t i = 0; i < gf_list->size(); i++) {
			tensor_t<value_type>* tensor = (*gf_list)[i].first;
			savable_size += tensor->get_mem_size();
		}
		// savable_size = savable_size - pool size
		// savable_size -= (SWAP_BLOCK_NUM * max_swappable_tensor_size);
		swappable_size = savable_size;
		printf("savable_size=%f, recomptable_size=%f, swappable_size=%f, max_swappable_tensor_size=%f, min_swappable_tensor_size=%f\n",
			BYTE_TO_MB(savable_size), BYTE_TO_MB(recomptable_size), BYTE_TO_MB(swappable_size), BYTE_TO_MB(max_swappable_tensor_size), BYTE_TO_MB(min_swappable_tensor_size));
		if (ME > savable_size) {
			printf("Excessive size (%fMB) > savable size (%fMB), network can not be trained in this batchsize\n", BYTE_TO_MB(ME), BYTE_TO_MB(savable_size));
			return false;
		}
		else {
			*MR = 0;
			*MS = ME;
			printf("Swapping only\n");
			return true;
		}
		printf("ME=%f, MS=%f\n", BYTE_TO_MB(ME), BYTE_TO_MB(*MS));
		printf("MS >= ME, Distribution Done!\n");
		return true;
	}		
}

template <class value_type>
bool ATPSearch<value_type>::IsRecomputingTensorLegal(tensor_t<value_type>* tensor) {
	std::map<int, void* > layers = reg->get_net_layers();
	size_t layer_id = tensor->get_layer_id();
	tensor_t<value_type>* t_out;
	base_layer_t<value_type>* layer = (base_layer_t<value_type>*)layers.find(layer_id)->second;
	base_layer_t<value_type>* next_layer = layer;
	base_layer_t<value_type>* last_layer = layer;
	int counter = 1;
	int max_contiguous_recomputation = RECOMPUTE_POOL_NUM;
	while (true) {
		next_layer = (next_layer->get_next())[0];
		if (next_layer->get_layer_structure() == SISO) {
			if (((base_network_layer_t<value_type>*)next_layer)->get_f_out()->get_position() == RECOMPUTE_IN_BACKWARD) {
				counter++;
			}
			else break;
		}						
		else break;
		if (counter > max_contiguous_recomputation) break;
	}
	if (counter > max_contiguous_recomputation) {
		printf("layer%d-tensor%d can not be recomputed\n", layer->get_base_id(), tensor->get_tensor_id());
		while (next_layer != layer) {
			t_out = ((base_network_layer_t<value_type>*)next_layer)->get_f_out();
			// printf("layer%d-tensor%d, ", next_layer->get_base_id(), t_out->get_tensor_id());
			next_layer = (next_layer->get_prev())[0];
		}
		// printf("and layer%d-tensor%d are all recomputed\n", next_layer->get_base_id(), t_out->get_tensor_id());
		return false;
	}
	while (true) {
		last_layer = (last_layer->get_prev())[0];
		if (next_layer->get_layer_structure() == SISO) {
			if (((base_network_layer_t<value_type>*)last_layer)->get_f_out()->get_position() == RECOMPUTE_IN_BACKWARD) {
				counter++;
			}
			else break;
		}
		else break;
	}
	if (counter > max_contiguous_recomputation) {
		printf("layer%d-tensor%d can not be recomputed\n", layer->get_base_id(), tensor->get_tensor_id());
		while (last_layer != layer) {
			t_out = ((base_network_layer_t<value_type>*)last_layer)->get_f_out();
			// printf("layer%d-tensor%d, ", last_layer->get_base_id(), t_out->get_tensor_id());
			last_layer = (last_layer->get_next())[0];
		}
		while (layer != next_layer) {
			t_out = ((base_network_layer_t<value_type>*)layer)->get_f_out();
			// printf("layer%d-tensor%d, ", last_layer->get_base_id(), t_out->get_tensor_id());
			layer = (layer->get_next())[0];
		}
		// printf("and layer%d-tensor%d are all recomputed\n", layer->get_base_id(), t_out->get_tensor_id());
		return false;
	}
	else {
		return true;
	}
}

template <class value_type>
void ATPSearch<value_type>::ResetTensorPosition() {
	std::map<int, void* > layers = reg->get_net_layers();
	auto tensors = reg->get_all_tensors();
	for (auto iter = tensors->begin(); iter != tensors->end(); iter++) {
		(*iter)->set_position(REMAIN_IN_GPU);
	}
	// for (auto iter = layers.begin(); iter != layers.end(); iter++) {
	// 	base_layer_t<value_type>* layer = (base_layer_t<value_type>*)iter->second;
	// 	if (layer->get_layer_structure() == SISO) {
	// 		tensor_t<value_type>* tensor = ((base_network_layer_t<value_type>*)layer)->get_f_out();
	// 		tensor->set_position(REMAIN_IN_GPU);
	// 		tensor_t<value_type>* reserve_buff = ((base_network_layer_t<value_type>*)layer)->get_reserve_buff();
	// 		if (reserve_buff != NULL) {
	// 			reserve_buff->set_position(REMAIN_IN_GPU);
	// 		}
	// 	}
	// 	else {  // layer->get_layer_structure() == MIMO
	// 		auto tensors = ((base_structure_t<value_type>*)layer)->get_outputs();
	// 		for (int i = 0; i < tensors.size(); i++) {
	// 			tensors[i]->set_position(REMAIN_IN_GPU);
	// 		}	
	// 	}
	// }
}

template <class value_type>
bool ATPSearch<value_type>::SelectRecomputingTensor_v2(size_t* MR, std::vector<std::pair<tensor_t<value_type>*, GENERATE_EFFICIENCY*>>* gf_list, std::vector<tensor_t<value_type>*>* recompute_tensors) {
	std::map<int, void* > layers = reg->get_net_layers();
	size_t mr = 0;
	size_t _MR = *MR;
	printf("_MR = %f\n", BYTE_TO_MB(_MR));
	for (auto iter = layers.begin(); iter != layers.end(); iter++) {
		// Reset tensor position into REMAIN_IN_GPU  
		base_layer_t<value_type>* layer = (base_layer_t<value_type>*)iter->second;
		tensor_t<value_type>* tensor;
		if (layer->get_layer_structure() == MIMO) {
			tensor = ((base_structure_t<value_type>*)layer)->get_outputs()[0];
			tensor->set_position(REMAIN_IN_GPU);
		}
		else {
			tensor = ((base_network_layer_t<value_type>*)layer)->get_f_out();
			tensor->set_position(REMAIN_IN_GPU);
		}
	}
	if (_MR < 512) {
		return true;
	}
	for (size_t i = 0; i < gf_list->size(); i++) {
		if ((*gf_list)[i].second->operation == RECOMPUTED) {
			// int layer_id = (*gt_list)[i].first;
			// tensor_t<value_type>* tensor = net_layer->get_f_out();
			tensor_t<value_type>* tensor = (*gf_list)[i].first;
			int layer_id = tensor->get_layer_id();
			base_network_layer_t<value_type>* net_layer = (base_network_layer_t<value_type>*) layers.find(layer_id)->second;
			if (IsRecomputingTensorLegal(tensor)) 
			{
				auto iter = std::find(recompute_tensors->begin(), recompute_tensors->end(), tensor);
				if (iter == recompute_tensors->end()) {  // Prevent duplicate addition
					recompute_tensors->push_back(tensor);
					tensor->set_position(RECOMPUTE_IN_BACKWARD);
					((base_layer_t<value_type>*)net_layer)->set_layer_position(RECOMPUTE_LAYER);
					mr += tensor->get_mem_size();
				}
				if (mr >= _MR) {
					for (size_t j = 0; j < recompute_tensors->size(); j++) {
						layer_id = (*recompute_tensors)[j]->get_layer_id();
						base_layer_t<value_type>* layer = (base_layer_t<value_type>*)layers.find(layer_id)->second;
						// printf("recompute tensor%d-layer%d-type%d\n", (*recompute_tensors)[j]->get_tensor_id(), layer_id, layer->get_layer_type());
					}
					printf("total recomputing size = %zd = %f\n", mr, BYTE_TO_MB(mr));
					break;
				}
			}
		}
	}
	*MR = mr;
	if (*MR >= _MR) {
		printf("after traversing gt_list, mr = %f >= MR = %f, SelectRecomputingTensor done!, update MR to %fMB\n", BYTE_TO_MB(mr), BYTE_TO_MB(_MR), BYTE_TO_MB(*MR));
		return true;
	}
	else {
		printf("after traversing gt_list, mr = %f < MR = %f, SelectRecomputingTensor fail!, update MR to %fMB\n", BYTE_TO_MB(mr), BYTE_TO_MB(_MR), BYTE_TO_MB(*MR));
		return false;
	}
}

template <class value_type>
bool ATPSearch<value_type>::SelectRecomputingTensor(size_t* MR, std::vector<std::pair<tensor_t<value_type>*, MIN_GENERATE*>>* gt_list, std::vector<tensor_t<value_type>*>* recompute_tensors) {
	std::map<int, void* > layers = reg->get_net_layers();
	size_t mr = 0;
	size_t _MR = *MR;
	for (auto iter = layers.begin(); iter != layers.end(); iter++) {
		// Reset tensor position into REMAIN_IN_GPU  
		base_layer_t<value_type>* layer = (base_layer_t<value_type>*)iter->second;
		tensor_t<value_type>* tensor;
		if (layer->get_layer_structure() == MIMO) {
			tensor = ((base_structure_t<value_type>*)layer)->get_outputs()[0];
			tensor->set_position(REMAIN_IN_GPU);
		}
		else {
			tensor = ((base_network_layer_t<value_type>*)layer)->get_f_out();
			tensor->set_position(REMAIN_IN_GPU);
		}
	}

	for (size_t i = 0; i < gt_list->size(); i++) {
		if ((*gt_list)[i].second->pre_operate == RECOMPUTED) {
			// int layer_id = (*gt_list)[i].first;
			// tensor_t<value_type>* tensor = net_layer->get_f_out();
			tensor_t<value_type>* tensor = (*gt_list)[i].first;
			int layer_id = tensor->get_layer_id();
			base_network_layer_t<value_type>* net_layer = (base_network_layer_t<value_type>*) layers.find(layer_id)->second;
			// if (IsRecomputingTensorLegal(tensor)) {
			{
				recompute_tensors->push_back(tensor);
				tensor->set_position(RECOMPUTE_IN_BACKWARD);
				((base_layer_t<value_type>*)net_layer)->set_layer_position(RECOMPUTE_LAYER);
				mr += tensor->get_mem_size();
				if (mr >= _MR) {
					for (size_t j = 0; j < recompute_tensors->size(); j++) {
						layer_id = (*recompute_tensors)[j]->get_layer_id();
						base_layer_t<value_type>* layer = (base_layer_t<value_type>*)layers.find(layer_id)->second;
						// printf("recompute tensor%d-layer%d-type%d\n", (*recompute_tensors)[j]->get_tensor_id(), layer_id, layer->get_layer_type());
					}
					// printf("total recomputing size = %zd = %f, MR = %f\n", mr, BYTE_TO_MB(mr), BYTE_TO_MB(MR));
					return true;
				}
			}
		}
	}
	*MR = mr;
	printf("after traversing gt_list, mr = %f < MR = %f, SelectRecomputingTensor fail!, update MR to %fMB\n", BYTE_TO_MB(mr), BYTE_TO_MB(_MR), BYTE_TO_MB(*MR));
	return false;
}

template <class value_type>
void ATPSearch<value_type>::GetAlternativeTensors(std::vector<tensor_t<value_type>*>* alternative_tensors) {
	std::map<int, void* > layers = reg->get_net_layers();
	std::vector<std::pair<int, net_comp> > net_route = reg->get_net_comp_route();
	for (int i = 0; i < net_route.size(); i++) {
		if (net_route[i].second == FORWARD) {
			base_layer_t<value_type>* layer = (base_layer_t<value_type>*) layers.find(net_route[i].first)->second;
			tensor_t<value_type>* tensor;
			std::vector<tensor_t<value_type>*> tensors;
			if (layer->get_layer_structure() == MIMO) {
				if (layer->get_layer_type() != FORK_L) {
					tensors = ((base_structure_t<value_type>*)layer)->get_outputs();
					for (int j = 0; j < tensors.size(); j++) {
						auto iter = std::find(alternative_tensors->begin(), alternative_tensors->end(), tensors[j]);
						if (iter == alternative_tensors->end()) {  // Prevent duplicate addition
							alternative_tensors->push_back(tensors[j]);
						}
					}
				}
				
			}
			else {
				if (layer->get_layer_type() != SOFTMAX && layer->get_layer_type() != DATA_L) {
					// HOME doesn't consider RNN layer
					// if (layer->get_layer_type() == RNN || layer->get_layer_type() == SATTN) {  
					// 	// tensor = ((rnn_layer_t<value_type>*)layer)->get_rnn_reserve_buff();
					// 	tensor = ((base_network_layer_t<value_type>*)layer)->get_reserve_buff();
					// 	alternative_tensors->push_back(tensor);
					// }
					tensor = ((base_network_layer_t<value_type>*)layer)->get_f_out();
					auto iter = std::find(alternative_tensors->begin(), alternative_tensors->end(), tensor);
					if (iter == alternative_tensors->end()) {
						alternative_tensors->push_back(tensor);
					}
				}
			}
		}
	}
	#ifdef DEBUG
		for (int i = 0; i < alternative_tensors->size(); i++) {
			printf("alternative_tensors%d-type%d layer%d-type%d\n", (*alternative_tensors)[i]->get_tensor_id(), (*alternative_tensors)[i]->get_type(), 
				(*alternative_tensors)[i]->get_layer_id(), ((base_layer_t<value_type>*)layers.find((*alternative_tensors)[i]->get_layer_id())->second)->get_layer_type());
		}
	#endif
}

// #define DEBUG
template <class value_type>
void ATPSearch<value_type>::GetAlternativeSwappingTensors(std::vector<std::pair<tensor_t<value_type>*, GENERATE_EFFICIENCY*>>* gf_list, 
													std::vector<tensor_t<value_type>*>* recompute_tensors, 
													std::vector<tensor_t<value_type>*>* alternative_swap_tensors) 
{
	std::map<int, void* > layers = reg->get_net_layers();
	std::vector<std::pair<int, net_comp> > net_route = reg->get_net_comp_route();
	// for (auto iter = gf_list->begin(); iter != gf_list->end(); iter++) {
	// 	tensor_t<value_type>* tensor = iter->first;
	// 	if (tensor->get_position() != RECOMPUTE_IN_BACKWARD) {
	// 		alternative_swap_tensors->push_back(tensor);
	// 	}
	// }
	for (int i = 0; i < net_route.size(); i++) {
		if (net_route[i].second == FORWARD) {
			base_layer_t<value_type>* layer = (base_layer_t<value_type>*) layers.find(net_route[i].first)->second;
			tensor_t<value_type>* tensor;
			std::vector<tensor_t<value_type>*> tensors;
			if (layer->get_layer_structure() == MIMO) {
				tensors = ((base_structure_t<value_type>*)layer)->get_outputs();
				for (int j = 0; j < tensors.size(); j++) {
					if (tensors[j]->get_position() != RECOMPUTE_IN_BACKWARD) {
						auto iter = std::find(alternative_swap_tensors->begin(), alternative_swap_tensors->end(), tensors[j]);
						if (iter == alternative_swap_tensors->end()) {  // Prevent duplicate addition
							alternative_swap_tensors->push_back(tensors[j]);
						}
					}
				}
			}
			else if (layer->get_layer_type() == SOFTMAX || layer->get_layer_type() == DATA_L) {
				continue;
			}
			else {
				if (layer->get_layer_type() == RNN || layer->get_layer_type() == SATTN) {
					// tensor = ((rnn_layer_t<value_type>*)layer)->get_rnn_reserve_buff();
					tensor = ((base_network_layer_t<value_type>*)layer)->get_reserve_buff();
					alternative_swap_tensors->push_back(tensor);
				}
				tensor = ((base_network_layer_t<value_type>*)layer)->get_f_out();
				if (tensor->get_position() != RECOMPUTE_IN_BACKWARD) {
					alternative_swap_tensors->push_back(tensor);
				}
			}
		}
	}
	#ifdef DEBUG
		for (int i = 0; i < alternative_swap_tensors->size(); i++) {
			printf("alternative_swap_tensor%d-type%d layer%d-type%d\n", (*alternative_swap_tensors)[i]->get_tensor_id(), (*alternative_swap_tensors)[i]->get_type(), 
				(*alternative_swap_tensors)[i]->get_layer_id(), ((base_layer_t<value_type>*)layers.find((*alternative_swap_tensors)[i]->get_layer_id())->second)->get_layer_type());
		}
	#endif
}
// #undef DEBUG

template <class value_type>
void ATPSearch<value_type>::set_output_tensor_id() {
	auto all_outputs = reg->get_all_outputs();
	int id = 0;
	for (auto it = all_outputs->begin(); it != all_outputs->end(); it++) {
		tensor_t<value_type>* t = it->second;
		t->set_tensor_id(id);
		id++;
	}
	for (auto it = all_outputs->begin(); it != all_outputs->end(); it++) {
		tensor_t<value_type>* t = it->second;
		printf("tensor%d ", t->get_tensor_id());
	}
	printf("num = %d\n", id);
}

template <class value_type>
void ATPSearch<value_type>::set_excessive_size_by_batchsize(int batchsize) {
	size_t total_mem = total_men_by_batchsize[batchsize];
	printf("total_mem = %f, gpu_total_mem = %f, pool_size = %f, ideal_offload_size = %f\n", 
		BYTE_TO_MB(total_mem), BYTE_TO_MB(this->gpu_total_mem), BYTE_TO_MB(this->pool_size), BYTE_TO_MB(ideal_offload_size_by_batchsize[batchsize]));
	if (total_mem > (this->gpu_total_mem)) {
		excessive_size_by_batchsize[batchsize] = total_mem - this->gpu_total_mem;  // Estimated pool size 11000000
	}
	else {
		excessive_size_by_batchsize[batchsize] = 0;
	}
}

template <class value_type>
void ATPSearch<value_type>::set_ideal_offload_time_by_batchsize(int batchsize) {
	size_t offload_size = ideal_offload_size_by_batchsize[batchsize];
	ideal_offload_time_by_batchsize[batchsize] = (double)offload_size / (double)this->pcie_bandwith;
}

template <class value_type>
void ATPSearch<value_type>::set_ideal_iter_time_by_batchsize(int batchsize) {
	// forward time with offload
	double ft = iter_forward_time_by_batchsize[batchsize];
	double bt = iter_backward_time_by_batchsize[batchsize];
	double ct = ideal_offload_time_by_batchsize[batchsize];
	// double update_time = iter_computing_time_by_batchsize[batchsize] - ft - bt;
	double update_time = 0;
	double fo_time = ft > ct ? ft : ct ;
	double bp_time = bt > ct ? bt : ct ;
	ideal_iter_time_by_batchsize[batchsize] = fo_time + bp_time + update_time;
	// printf("\nft=%lf, btt=%lf, ut=%lf, ct=%lf, fo=%lf, bp=%lf, ideal_iter_time_by_batchsize[%d] = %lf", ft, bt, update_time, ct, fo_time, bp_time, batchsize, ideal_iter_time_by_batchsize[batchsize]);
	// while(1);
}

template <class value_type>
void ATPSearch<value_type>::get_best_config(double* max_throughput, size_t* batchsize, size_t* min_offloadsize) {
	*max_throughput = 0;
	for (auto it = ideal_throughput_by_batchsize.begin(); it != ideal_throughput_by_batchsize.end(); it++) {
		if (*max_throughput < it->second) {
			*max_throughput = it->second;
			*batchsize = it->first;
			*min_offloadsize = ideal_offload_size_by_batchsize[it->first];
			printf("throughput = %f, batchsize = %d, min_offloadsize = %zd\n", 
				it->second, it->first, ideal_offload_size_by_batchsize[it->first]);
		}
		else {
			printf("throughput = %f, batchsize = %d, min_offloadsize = %zd\n", 
				it->second, it->first, ideal_offload_size_by_batchsize[it->first]);
		}
	}
	this->best_batchsize = *batchsize;
	this->min_offloadsize = *min_offloadsize;
}

template <class value_type>
void ATPSearch<value_type>::print_swap_layers() {
	std::map<int, void* > net_layers  = reg->get_net_layers();
    // auto route = reg->get_net_comp_route();
	printf("\n***********************SWAP_LAYERS*************************%d\n", 0);
	for (size_t i = 0; i < this->swap_num; i++) {
		base_layer_t<value_type>* layer = (base_layer_t<value_type>*)((net_layers.find(this->swap_layers[i]))->second);
		LAYER layer_type = layer->get_layer_type();
		std::string type_name;
		if (layer_type == CONV) type_name.assign("CONV");
		else if (layer_type == POOL) type_name.assign("POOL");
		else if (layer_type == ACT) type_name.assign("ACT");
		else if (layer_type == BN) type_name.assign("BN");
		else if (layer_type == FC) type_name.assign("FC");
		else if (layer_type == LRN) type_name.assign("LRN");
		else if (layer_type == PADDING) type_name.assign("PADDING");
		else if (layer_type == DROPOUT) type_name.assign("DROPOUT");
		else if (layer_type == SOFTMAX) type_name.assign("SOFTMAX");
		else if (layer_type == CONCAT) type_name.assign("CONCAT");
		else if (layer_type == FORK_L) type_name.assign("FORK_L");
		else if (layer_type == JOIN_L) type_name.assign("JOIN_L");
		else type_name.assign("555"); 
		printf("%d, ", layer->get_base_id(), type_name.c_str());
    }
	printf("swap_layer_num = %d\n", swap_num);
}

template <class value_type>
double ATPSearch<value_type>::invalid_time_v2(bool* comp_route_swap, int size) {
	std::map<int, void* > net_layers  = reg->get_net_layers();
	std::vector<std::pair<int, net_comp> > net_comp_route = reg->get_net_comp_route();
	// printf("net_comp_route.size = %d, comp_route_swap.size = %d\n", net_comp_route.size(), size);
	double invalid_time = 0.0;
	double temp_invalid_time = 0.0;
	double temp_valid_time = 0.0;
	double temp_occupy_time = 0.0;
	for (size_t i = 1; i < net_comp_route.size(); i++) {
		if (net_comp_route[i].second == FORWARD) {
			int layer_id = net_comp_route[i].first;
			if (comp_route_swap[i] == true) {
				temp_invalid_time = scalar_relu(temp_occupy_time - temp_valid_time);
				invalid_time += temp_invalid_time;
				temp_occupy_time = 0;
				temp_valid_time = layers_ft[layer_id] + (double)layers_output_size[layer_id] / pcie_bandwith;
				temp_occupy_time += layers_ft[layer_id];
			}
			else {
				temp_occupy_time += layers_ft[layer_id];
			}
		}
	}
	return invalid_time;
}

template <class value_type>
double ATPSearch<value_type>::invalid_time() {
	std::map<int, void* > net_layers  = reg->get_net_layers();
	std::vector<std::pair<int, net_comp> > net_comp_route = reg->get_net_comp_route();
	double invalid_time = 0.0;
	double temp_invalid_time = 0.0;
	double temp_valid_time = 0.0;
	double temp_occupy_time = 0.0;
	for (size_t i = 1; i < net_comp_route.size(); i++) {
		if (net_comp_route[i].second == FORWARD) {
			int layer_id = net_comp_route[i].first;
			if (this->comp_route_swap[i] == true) {
				temp_invalid_time = scalar_relu(temp_occupy_time - temp_valid_time);
				invalid_time += temp_invalid_time;
				temp_occupy_time = 0;
				temp_valid_time = layers_ft[layer_id] + (double)layers_output_size[layer_id] / pcie_bandwith;
				temp_occupy_time += layers_ft[layer_id];
			}
			else {
				temp_occupy_time += layers_ft[layer_id];
			}
		}
	}
	return invalid_time;
}

template <class value_type>
size_t ATPSearch<value_type>::get_pool_size() {
	std::map<int, void* > net_layers  = reg->get_net_layers();
	std::vector<std::pair<int, net_comp> > net_comp_route = reg->get_net_comp_route();
	int layer_id = net_comp_route[1].first;  // 先把第一层放到swap_layer
	base_layer_t<value_type>* layer = (base_layer_t<value_type>*) net_layers.find(layer_id)->second;
	this->pool_size = (((base_network_layer_t<value_type>*)layer)->get_f_out())->get_mem_size();
	return this->pool_size;
}

template <class value_type>
size_t ATPSearch<value_type>::offload_size_by_swap_layers_pool_malloc_mode(bool* comp_route_swap, int size) {
	std::map<int, void* >* net_layers  = reg->get_net_layers_ptr();
	std::vector<std::pair<int, net_comp> >* net_comp_route = reg->get_net_comp_route_ptr();
	int max_layer_id = net->get_max_layer_id();
	size_t offload_size = 0;
	size_t temp_offload_size;
	size_t temp_size;
	for (size_t i = 0; i < net_comp_route->size(); i++) {
		if ((*net_comp_route)[i].second == FORWARD) {
			if (comp_route_swap[i] == true) {
				int layer_id = (*net_comp_route)[i].first;
				base_layer_t<value_type>* layer = (base_layer_t<value_type>*) net_layers->find(layer_id)->second;
				temp_size = layers_output_size[layer_id];
				offload_size += temp_size;
			}
		}
	}
	return offload_size;
}

template <class value_type>
size_t ATPSearch<value_type>::offload_size_by_swap_layers(bool* comp_route_swap, int size) {
	std::map<int, void* > net_layers  = reg->get_net_layers();
	std::vector<std::pair<int, net_comp> > net_comp_route = reg->get_net_comp_route();
	// printf("net_comp_route.size = %d, comp_route_swap.size = %d\n", net_comp_route.size(), size);
	int max_layer_id = net->get_max_layer_id();
	size_t cuda_mem_block = 2097152;
	size_t offload_size = 0;
	size_t temp_offload_size;
	size_t temp_size;
	size_t residual_mem = 0;
	int block_num = 0;

	for (size_t i = 0; i < net_comp_route.size(); i++) {
		if (net_comp_route[i].second == FORWARD) {
			if (comp_route_swap[i] == true) {
				int layer_id = net_comp_route[i].first;
				base_layer_t<value_type>* layer = (base_layer_t<value_type>*) net_layers.find(layer_id)->second;
				std::vector<base_layer_t<value_type>*> next_layers = layer->get_next();
				temp_size = layers_output_size[layer_id];	
				if (temp_size < residual_mem) {
            		residual_mem -= temp_size;
        		}
				else {
					while(true) {
						block_num++;  // 申请一个显存块
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
	}
	offload_size = cuda_mem_block * block_num;
	return offload_size;
}

template <class value_type>
double ATPSearch<value_type>::offload_size_error(bool comp_route_swap[], int size) {
	size_t offload_size = offload_size_by_swap_layers_pool_malloc_mode(comp_route_swap, size);
	size_t offload_requirement = get_offload_requirement();
	size_t error;
	if (offload_size > offload_requirement) {
		error = offload_size - offload_requirement;
	}
	else {
		error = 0;
	}
	// size_t error = offload_size - ideal_offload_size_by_batchsize[best_batchsize];
	double error2 = (double)error / (double)ideal_offload_size_by_batchsize[best_batchsize];
	return error2; //(error2 > 1 ? 1 : error2);
}

// template <class value_type>
// void ATPSearch<value_type>::clear_swap_tensors( ) {
// 	while (this->swap_tensors.empty()) {

// 	} 
// 	this->swap_tensors
// }

template <class value_type>
size_t ATPSearch<value_type>::set_swap_tensors(bool* comp_route_swap) {
    std::vector<std::pair<int, net_comp> >* net_comp_route = reg->get_net_comp_route_ptr();
    std::map<int, void* >* net_layers  = reg->get_net_layers_ptr();
	base_structure_t<value_type>* structure_layer;
	base_network_layer_t<value_type>* net_layer;
	size_t sum = 0;
    // forward swap_tensors init
	this->swap_tensors.clear();
	this->prefetch_tensors.clear();
	for (size_t i = 0; i < net_comp_route->size(); i++) {  // reset
		if ((*net_comp_route)[i].second == FORWARD) {
				int layer_id = (*net_comp_route)[i].first;
				base_layer_t<value_type>* layer = (base_layer_t<value_type>*) net_layers->find(layer_id)->second;
				layer->set_layer_position(REMAIN_LAYER);
				if (layer->get_layer_structure() == MIMO) {
					structure_layer = (base_structure_t<value_type>* )layer;
					std::vector<tensor_t<value_type>* > tensors = structure_layer->get_outputs();
					tensors[0]->reset_all_state();

				}
				else {
					net_layer = (base_network_layer_t<value_type>* )layer;
					tensor_t<value_type>* tensor = net_layer->get_f_out();
					tensor->reset_all_state();
				}
		}
	}
    for (size_t i = 0; i < net_comp_route->size(); i++) {
        int layer_id = (*net_comp_route)[i].first;
        base_layer_t<value_type>* layer = (base_layer_t<value_type>*) net_layers->find(layer_id)->second;
		if ((*net_comp_route)[i].second == FORWARD) {
			// fill in the offload tensors' list
			if ( comp_route_swap[i] == true ) {
				layer->set_layer_position(SWAP_LAYER);
				if (layer->get_layer_structure() == MIMO) {
					structure_layer = (base_structure_t<value_type>* )layer;
					std::vector<tensor_t<value_type>* > tensors = structure_layer->get_outputs();
					tensors[0]->set_position(SHARED_GPU_POOL);
					this->swap_tensors.push_back(tensors[0]);
					// printf("set layer%d tensor%d is swap_tensor\n", layer_id, tensors[0]->get_tensor_id());
					sum += tensors[0]->get_mem_size();
				}
				else {  // layer->get_layer_structure() == SISO
					net_layer = (base_network_layer_t<value_type>* )layer;
					tensor_t<value_type>* tensor = net_layer->get_f_out();
					tensor->set_position(SHARED_GPU_POOL);
					this->swap_tensors.push_back(tensor);
					// printf("set layer%d tensor%d is swap_tensor\n", layer_id, tensor->get_tensor_id());
					sum += tensor->get_mem_size();
				}
			}
		}
		else {  // (*net_comp_route)[(*swap_layers)[i]].second == BACKWARD
			// fill in the prefetch tensors' list
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
	return sum;
}

// #define DEBUG
template <class value_type>
bool ATPSearch<value_type>::GetRecomputeTime(std::map<int, void* >* net_layers, tensor_t<value_type>* tensor, base_layer_t<value_type>* layer, net_comp dir, double* re_time) {
	static int re_counter = 0;
	static int re_counter2 = 0;
	re_counter2++;
	if (layer->get_layer_type() == CONCAT || layer->get_layer_type() == JOIN_L) {
		printf("It is not recommended to recompute the CONCAT & JOIN_L layer\n");
		return 0;
	}
	bool is_re_done = false;
	tensor_t<value_type>* t_in;
	if (layer->get_layer_type() == FORK_L) {
		t_in = ((base_structure_t<value_type>*)layer)->get_inputs()[0];
	}
	else {
		t_in = ((base_network_layer_t<value_type>*)layer)->get_f_in();
	}
	// printf("tensor%d recompute need layer%d=t_in%d-p%d-s%d\n", tensor->get_tensor_id(), t_in->get_layer_id(), t_in->get_tensor_id(), t_in->get_data_position(), t_in->get_data_state());
	if (t_in->get_data_position() == DELETED) {  // Need to recompute last layer
		base_layer_t<value_type>* last_layer = (base_layer_t<value_type>*)net_layers->find(t_in->get_layer_id())->second;
		if (t_in == tensor) {  // if fork layer, t_in == t_out, go to last_layer of this last_layer
			last_layer = layer->get_prev()[0];
			// t_in = ((base_network_layer_t<value_type>*)last_layer)->get_f_in();
			// int last_layer_id = t_in->get_layer_id();
			// last_layer = (base_layer_t<value_type>*)net_layers->find(last_layer_id)->second;
		}
		is_re_done = GetRecomputeTime(net_layers, t_in, last_layer, FORWARD, re_time);
		#ifdef DEBUG
		printf("after GetRecomputeTime t_in%d-p%d-s%d", t_in->get_tensor_id(), t_in->get_data_position(), t_in->get_data_state());
		#endif
	}
	else if (t_in->get_data_position() == IN_CPU) {  // Wait for t prefetch to complete
		is_re_done = false;
	}
	if (t_in->get_data_position() == IN_GPU || t_in->get_data_position() == IN_CPU_GPU) {
	// else {  // t_in is in GPU, run recomputation of layer, and record recomputation time
		*re_time = *re_time + layers_ft[layer->get_base_id()];
		re_counter++;
		// printf("re_counter = %d\n", re_counter);
		// printf("re_counter2 = %d\n", re_counter2);
		is_re_done = true;
		tensor_t<value_type>* t_out = ((base_network_layer_t<value_type>*)layer)->get_f_out();
		t_out->set_data_position(IN_GPU);
		t_out->set_data_state(STILL_USED);
		#ifdef DEBUG
			printf("complete layer%d-type%d-tensor%d recomputation\n", layer->get_base_id(), layer->get_layer_type(), ((base_network_layer_t<value_type>*)layer)->get_f_out()->get_tensor_id());
		#endif
	}
	return is_re_done;
}
// #undef DEBUG

// #define DEBUG
template <class value_type>
bool ATPSearch<value_type>::IterationTimeEvaluator(std::vector<tensor_t<value_type>* >* recompute_tensors,
		std::vector<tensor_t<value_type>* >* swap_tensors, double* forward, double* sync_time_f, double* offload_time,
		std::vector<tensor_t<value_type>* >* prefetch_tensors, double* backward, double* sync_time_b, double* fetch_time) 
{
	std::map<int, void* >* net_layers = reg->get_net_layers_ptr();
	std::vector<std::pair<int, net_comp> >* net_comp_route = reg->get_net_comp_route_ptr();
	base_layer_t<value_type>* layer;
	base_layer_t<value_type>* re_layer;
	double cur_recomp_time = 0.0;
	double T_offload = 0.0;
	double T_fetch = 0.0;
	double sync = 0.0;
	double re_tc_time = 0.0;
	double re_tf_time = 0.0;
	double re_tp_time = 0.0;
	double re_tb_time = 0.0;
	bool still_swap_flag = false;
	double T_f = 0.0;
	double T_c = 0.0;
	double T_b = 0.0;
	double T_p = 0.0;
	int bplus_counter = 0;
	double re_time = 0.0;
	tensor_t<value_type>* t_out;
	tensor_t<value_type>* t_in;
	std::vector<tensor_t<value_type>*> t_outs;
	std::vector<tensor_t<value_type>*> t_ins;
	std::vector<tensor_t<value_type>*> swapped_tensors;
	bool is_layer_done = false;
	bool is_re_done = false;
	int layer_id;
	int block_id_j;
	int block_id;
	tensor_t<value_type>* block_tensor;
	for ( int k = 0; k < net_comp_route->size(); k++ ) {  // reset tensor state
		if ((*net_comp_route)[k].second == FORWARD) {
			layer_id = (*net_comp_route)[k].first;
			layer = (base_layer_t<value_type>*)(net_layers->find(layer_id)->second);
			if (layer->get_layer_structure() == SISO) {
				t_out = ((base_network_layer_t<value_type>*)layer)->get_f_out();
				t_out->reset_all_data_state();
				if (layer->get_layer_type() == RNN || layer->get_layer_type() == SATTN) {
					// tensor_t<value_type>* reserve_buff = ((rnn_layer_t<value_type>*)layer)->get_rnn_reserve_buff();
					tensor_t<value_type>* reserve_buff = ((base_network_layer_t<value_type>*)layer)->get_reserve_buff();
					reserve_buff->reset_all_data_state();
				}
			}
			else {
				t_outs = ((base_structure_t<value_type>*)layer)->get_outputs();
				for (int j = 0; j < t_outs.size(); j++) {
					t_outs[j]->reset_all_data_state();
				}
			}
		}
	}

	size_t i = 0, j = 0;
	T_f +=  layers_ft[(*net_comp_route)[i].first];
	i++;
	mem_controller->reset_swap_block();
	for (int k = 0; k < SWAP_BLOCK_NUM; k++) {  // the last three swap_tensors don't need to be offloaded, and the first three prefetch_tensors don't need to be fetched 
		swap_tensors->pop_back();
		// prefetch_tensors->erase(prefetch_tensors->begin());
	}
	#ifdef DEBUG
	printf("IterationTimeEvaluator: reset tensor state done, T_f = %lf, T_b = %lf, swap_tensors.size = %d, prefetch_tensors.size = %d\n", 
		T_f, T_b, swap_tensors->size(), prefetch_tensors->size());
	#endif
	// while(1);

	std::vector<std::pair<int, net_comp> > net_route = reg->get_net_comp_route();
	std::map<int, void* > layers = reg->get_net_layers();
	// printf("CODE size IterationTimeEvaluator = %d: ", layers.size());
	#ifdef DEBUG
	for (size_t q = 0; q < layers.size(); q++) {  // set rs_code_route
		if (net_route[q].second == FORWARD) {
			base_layer_t<value_type>* layer = (base_layer_t<value_type>*) layers.find(net_route[q].first)->second;
			if (layer->get_layer_structure() == MIMO) {
				t_out = ((base_structure_t<value_type>*)layer)->get_outputs()[0];
				if (t_out->get_position() == SHARED_GPU_POOL) {
					// rs_code_route[i] = 1;
					printf("x%d-t%d-l%d-t%d-s[%d]-p[%d]--SWAP\n", t_out->get_tensor_id(), t_out->get_type(), layer->get_base_id(), layer->get_layer_type(), 1, t_out->get_swap_block_id(), t_out->get_prefetch_block_id());
				} 
				else if (t_out->get_position() == RECOMPUTE_IN_BACKWARD) {
					printf("x%d-t%d-l%d-t%d--RECOMPUTE\n", t_out->get_tensor_id(), t_out->get_type(), layer->get_base_id(), layer->get_layer_type());
				}
				else {
					// rs_code_route[i] = 0;
					printf("x%d-t%d-l%d-t%d--STAY\n", t_out->get_tensor_id(), t_out->get_type(), layer->get_base_id(), layer->get_layer_type(), 0);
				}
			}
			else {
				if (layer->get_layer_type() == RNN || layer->get_layer_type() == SATTN) {
					// tensor_t<value_type>* reserve_buff = ((rnn_layer_t<value_type>*)layer)->get_rnn_reserve_buff();
					tensor_t<value_type>* reserve_buff = ((base_network_layer_t<value_type>*)layer)->get_reserve_buff();
					if (reserve_buff->get_position() == SHARED_GPU_POOL) {
						// rs_code_route[i] = 1;
						printf("r%d-t%d-l%d-t%d-s[%d]-p[%d]--SWAP\n", reserve_buff->get_tensor_id(), reserve_buff->get_type(), layer->get_base_id(), layer->get_layer_type(), 1, reserve_buff->get_swap_block_id(), reserve_buff->get_prefetch_block_id());
					} 
					else if (reserve_buff->get_position() == RECOMPUTE_IN_BACKWARD) {
						printf("r%d-t%d-l%d-t%d--RECOMPUTE\n", reserve_buff->get_tensor_id(), reserve_buff->get_type(), layer->get_base_id(), layer->get_layer_type(), 2);
					}
					else {
						// rs_code_route[i] = 0;
						printf("r%d-t%d-l%d-t%d--STAY\n", reserve_buff->get_tensor_id(), reserve_buff->get_type(), layer->get_base_id(), layer->get_layer_type(), 0);
					}
				}
				t_out = ((base_network_layer_t<value_type>*)layer)->get_f_out();
				if (t_out->get_position() == SHARED_GPU_POOL) {
					// rs_code_route[i] = 1;
					printf("x%d-t%d-l%d-t%d-s[%d]-p[%d]--SWAP\n", t_out->get_tensor_id(), t_out->get_type(), layer->get_base_id(), layer->get_layer_type(), 1, t_out->get_swap_block_id(), t_out->get_prefetch_block_id());
				} 
				else if (t_out->get_position() == RECOMPUTE_IN_BACKWARD) {
					printf("x%d-t%d-l%d-t%d--RECOMPUTE\n", t_out->get_tensor_id(), t_out->get_type(), layer->get_base_id(), layer->get_layer_type(), 2);
				}
				else {
					// rs_code_route[i] = 0;
					printf("x%d-t%d-l%d-t%d--STAY\n", t_out->get_tensor_id(), t_out->get_type(), layer->get_base_id(), layer->get_layer_type(), 0);
				}
			}
		}
	}
	printf("\n");
	#endif
	bool reserve_buff_ready = false;
	bool is_rnn_layer = false;
	mem_controller->reset_recompute_pool();
	mem_controller->reset_swap_block();
	for ( ; i < net_comp_route->size(); ) {
		layer_id = (*net_comp_route)[i].first;
		layer = (base_layer_t<value_type>*)(net_layers->find(layer_id)->second);
		if ((*net_comp_route)[i].second == FORWARD) {
			#ifdef DEBUG
				printf("start layer%d forward\n", layer_id);
			#endif
			if (layer->get_layer_structure() == SISO) {
				// bool reserve_buff_ready = false;
				// bool is_rnn_layer = false;
				#ifdef DEBUG
					printf("layer%d is SISO type%d\n", layer_id, layer->get_layer_type());
				#endif
				if ((layer->get_layer_type() == RNN || layer->get_layer_type() == SATTN) && reserve_buff_ready == false) {
					#ifdef DEBUG
						printf("layer%d is RNN\n", layer_id);
					#endif
					is_rnn_layer = true;
					// tensor_t<value_type>* reserve_buff = ((rnn_layer_t<value_type>*)layer)->get_rnn_reserve_buff();
					tensor_t<value_type>* reserve_buff = ((base_network_layer_t<value_type>*)layer)->get_reserve_buff();
					block_id = reserve_buff->get_swap_block_id();
					if (reserve_buff->get_position() == SHARED_GPU_POOL) {
						block_tensor = mem_controller->get_swap_block_tensor(block_id);
						if (block_tensor != NULL) {
						#ifdef DEBUG
							printf("layer%d-reserve_buff%d should use block[%d]-swapstate%d orig-tensor%d-type%d-layer%d-state%d-position%d\n", 
								layer->get_base_id(), reserve_buff->get_tensor_id(), block_id, mem_controller->get_swap_block_state(block_id), block_tensor->get_tensor_id(), block_tensor->get_type(), block_tensor->get_layer_id(), block_tensor->get_data_state(), block_tensor->get_data_position());
						#endif	
						}
						if (block_tensor == reserve_buff) {
							#ifdef DEBUG
								printf("layer%d-reserve_buff%d has been allocated\n", layer_id, reserve_buff->get_tensor_id());
							#endif
							reserve_buff_ready = true;
						}
						else {
							if (mem_controller->is_swap_block_occupied(block_id) == false || mem_controller->get_swap_block_state(block_id) == DONE) {
								#ifdef DEBUG
									printf("layer%d-reserve_buff%d is alloc to block[%d]\n", layer_id, reserve_buff->get_tensor_id(), reserve_buff->get_swap_block_id());
								#endif
								block_id = reserve_buff->get_swap_block_id();
								block_tensor = mem_controller->get_swap_block_tensor(block_id);
								if (block_tensor != NULL) {  // set block's original tensor data_position
									block_tensor->set_data_position(IN_CPU);
								}
								// swap_block[block_id] is ready, do layer's forward computation
								mem_controller->set_swap_block(block_id, reserve_buff);
								reserve_buff->set_data_state(STILL_USED);
								reserve_buff_ready = true;
							}
							else {
								reserve_buff_ready = false;
								is_layer_done = false;
							}
						}
					}
					else {
						reserve_buff_ready = true;
					}
				}
				if (reserve_buff_ready == true || is_rnn_layer == false) {
					reserve_buff_ready = false;
					is_rnn_layer = false;
					t_out = ((base_network_layer_t<value_type>*)layer)->get_f_out();
					#ifdef DEBUG
						printf("layer%d out tensor%d-position%d-dp%d\n", layer_id, t_out->get_tensor_id(), t_out->get_position(), t_out->get_data_position());
					#endif
					if (t_out->get_position() == SHARED_GPU_POOL) {
						block_id = t_out->get_swap_block_id();
						block_tensor = mem_controller->get_swap_block_tensor(block_id); // get the last tensor in this buffer
						if (block_tensor == t_out) {
							is_layer_done = true;
						}
						else {
							// #ifdef DEBUG
							// 	printf("layer%d-out%d should use block[%d]-swapstate%d orig-tensor%d-type%d-layer%d-state%d-position%d\n", 
							// 		layer_id, t_out->get_tensor_id(), block_id, mem_controller->get_swap_block_state(block_id), block_tensor->get_tensor_id(), block_tensor->get_type(), block_tensor->get_layer_id(), t_out->get_data_state(), t_out->get_data_position());
							// #endif 
							if (block_tensor != NULL) {
							#ifdef DEBUG
								printf("layer%d-out%d should use block[%d]-swapstate%d orig-tensor%d-type%d-layer%d-state%d-position%d\n", 
									layer_id, t_out->get_tensor_id(), block_id, mem_controller->get_swap_block_state(block_id), block_tensor->get_tensor_id(), block_tensor->get_type(), block_tensor->get_layer_id(), t_out->get_data_state(), t_out->get_data_position());
								mem_controller->printSWAPBLOCK("allocate tensor");
							#endif
							}
							if (mem_controller->is_swap_block_occupied(block_id) == false 
								|| mem_controller->get_swap_block_state(block_id) == DONE) {
							#ifdef DEBUG
								printf("layer%d-tensor%d is alloc to block[%d]\n", layer_id, t_out->get_tensor_id(), t_out->get_swap_block_id());
							#endif
								block_id = t_out->get_swap_block_id();
								block_tensor = mem_controller->get_swap_block_tensor(block_id);
								if (block_tensor != NULL) {  // set block's original tensor data_position
									block_tensor->set_data_position(IN_CPU);
								}
								// swap_block[block_id] is ready, do layer's forward computation
								mem_controller->set_swap_block(block_id, t_out);
								t_out->set_data_state(STILL_USED);
								is_layer_done = true;
							}
							else {	
								is_layer_done = false;
							}
						}	
					}
					else if (t_out->get_position() == RECOMPUTE_IN_BACKWARD) {  // recomputing tensor
						if (t_out->get_data_position() == IN_GPU) {
                            is_layer_done = true;
                        }
						else {
							block_id = t_out->get_recompute_pool_id();
							#ifdef DEBUG
								printf("layer%d-type%d out tensor%d find recompute block\n", layer_id, layer->get_layer_type(), t_out->get_tensor_id());
								mem_controller->printRecomputePool("forward before find_free_recompute_block");
							#endif
							if (mem_controller->find_free_recompute_block(FORWARD, &block_id, t_out, false) == false) {
								printf("forward there is no recompute_pool for tensor%d-layer%d, all blocks is still using\n", t_out->get_tensor_id(), t_out->get_layer_id());
								is_layer_done = false;
								exit(0);
							}
							else {
								block_tensor = mem_controller->get_recompute_block_tensor(block_id);
								if (block_tensor != NULL) {
									#ifdef DEBUG
									printf("layer%d-type%d out tensor%d need re_block[%d], original block_tensor%d-ds%d-dp%d\n", 
										layer_id, layer->get_layer_type(), t_out->get_tensor_id(), block_id, block_tensor->get_tensor_id(), block_tensor->get_data_state(), block_tensor->get_data_position());
									#endif
									block_tensor->set_data_position(DELETED);
								}
								mem_controller->set_recompute_block(block_id, t_out);
								is_layer_done = true;
								#ifdef DEBUG
									mem_controller->printRecomputePool("forward after set recompute_block");
								#endif
							}
						}
					}
					else {  // common tensor
						is_layer_done = true;
					}
				}
				else {

				}
			}
			else {  // MIMO LAYER
				t_outs = ((base_structure_t<value_type>*)layer)->get_outputs();
				for (int j = 0; j < t_outs.size(); j++) {
					if (t_outs[j]->get_position() == SHARED_GPU_POOL) {
						// t_out = t_outs[j];
						block_id = t_outs[j]->get_swap_block_id();
						block_tensor = mem_controller->get_swap_block_tensor(block_id); // get the last tensor in this buffer
						if (block_tensor == t_outs[j]) {
							is_layer_done = true;
						}
						else {
							#ifdef DEBUG
								printf("start layer%d forward t_outs[%d]%d-data_position%d-set_data_state%d-block[%d]\n", 
									layer_id, j, t_outs[j]->get_tensor_id(), t_outs[j]->get_data_position(), t_outs[j]->get_data_state(), t_outs[j]->get_swap_block_id());
							#endif
							if (mem_controller->is_swap_block_occupied(block_id) == false 
								|| mem_controller->get_swap_block_state(block_id) == DONE) 
							{
							#ifdef DEBUG
								printf("layer%d-tensor%d is alloc to block[%d]\n", layer_id, t_outs[j]->get_tensor_id(), t_outs[j]->get_swap_block_id());
							#endif
								block_id = t_outs[j]->get_swap_block_id();
								block_tensor = mem_controller->get_swap_block_tensor(block_id);
								if (block_tensor != NULL) {  // set block's original tensor data_position
									block_tensor->set_data_position(IN_CPU);
								}
								mem_controller->set_swap_block(block_id, t_outs[j]);
								t_outs[j]->set_data_state(STILL_USED);
								is_layer_done = true;
							}
							else {
								is_layer_done = false;
								break;
							}
						}
					}
					else if (t_outs[j]->get_position() == RECOMPUTE_IN_BACKWARD) {  // recomputing tensor
						if (t_outs[j]->get_data_position() == IN_GPU) {
                            is_layer_done = true;
                        }
						else {
							block_id = t_outs[j]->get_recompute_pool_id();
							if (mem_controller->find_free_recompute_block(FORWARD, &block_id, t_outs[j], false) == false) {
								printf("forward there is no recompute_pool for tensor%d-layer%d, all blocks is still using\n", t_outs[j]->get_tensor_id(), t_outs[j]->get_layer_id());
								is_layer_done = false;
								exit(0);
							}
							else {
								block_tensor = mem_controller->get_recompute_block_tensor(block_id);
								if (block_tensor != NULL) {
									block_tensor->set_data_position(DELETED);
								}
								mem_controller->set_recompute_block(block_id, t_outs[j]);
								is_layer_done = true;
							}
						}
					}
					else {
						is_layer_done = true;
					}
				}  
			}
			if (is_layer_done == false) {  // common tensor
			#ifdef DEBUG
				printf("locked layer%d forward\n", layer_id);
			#endif
				// if swap_block[block_id] is occupied, Wait for the previous swap_tensor offloading to complete, to free swap_block[block_id]
				if (still_swap_flag == false) {  // no swap_tensor is offloading now
					block_tensor = mem_controller->get_swap_block_tensor((*swap_tensors)[j]->get_swap_block_id());
				#ifdef DEBUG
					printf("swap_tensors%d-position%d\n", (*swap_tensors)[j]->get_tensor_id(), (*swap_tensors)[j]->get_data_position());
				#endif
					if (j > swap_tensors->size()) j = swap_tensors->size();
					if ((*swap_tensors)[j]->get_data_position() == IN_GPU) {
						if ((*swap_tensors)[j]->get_use_counter(BACKWARD) == 0 && (*swap_tensors)[j]->get_type() != RNN_RESERVE) {  
						// if (false) {
							// if (*swap_tensors)[j] is useless in backward, Just update the state without actually unloading
							block_id_j = (*swap_tensors)[j]->get_swap_block_id();
							mem_controller->set_swap_block_state(block_id_j, DONE);  // swap_tensor_j is offloaded
							(*swap_tensors)[j]->set_data_position(IN_CPU_GPU);
							j++;  // go to next swap_tensor
							still_swap_flag = false;  // before swap_tensor_j offloaded, trainning is waiting, so still_swap_flag is false
						}
						else {
							// re_tc_time = (*swap_tensors)[j]->get_comm_time(); 
							re_tc_time = (*swap_tensors)[j]->get_swap_time(OFFLOAD);
							T_offload += re_tc_time;
						#ifdef DEBUG
							// printf("start offload tensor%d-layer%d-block[%d]-size%zd-comm_time=%f\n", 
							// 		(*swap_tensors)[j]->get_tensor_id(), (*swap_tensors)[j]->get_layer_id(), (*swap_tensors)[j]->get_swap_block_id(), (*swap_tensors)[j]->get_mem_size(), (*swap_tensors)[j]->get_comm_time()); 
							printf("start offload tensor%d-layer%d-block[%d]-size%zd-offload_time=%f\n", 
									(*swap_tensors)[j]->get_tensor_id(), (*swap_tensors)[j]->get_layer_id(), (*swap_tensors)[j]->get_swap_block_id(), (*swap_tensors)[j]->get_mem_size(), (*swap_tensors)[j]->get_swap_time(OFFLOAD)); 
							printf("re_tc_time = %f", re_tc_time);
						#endif
							T_f += re_tc_time;
							T_c += re_tc_time;
							sync += re_tc_time;
							block_id_j = (*swap_tensors)[j]->get_swap_block_id();
							mem_controller->set_swap_block_state(block_id_j, DONE);  // swap_tensor_j is offloaded
							(*swap_tensors)[j]->set_data_position(IN_CPU_GPU);
						#ifdef DEBUG
							printf("complete offload tensor%d-layer%d-block[%d]\n", 
									(*swap_tensors)[j]->get_tensor_id(), (*swap_tensors)[j]->get_layer_id(), (*swap_tensors)[j]->get_swap_block_id());
						#endif
							j++;  // go to next swap_tensor
							still_swap_flag = false;  // before swap_tensor_j offloaded, trainning is waiting, so still_swap_flag is false
						}
					}
					else {  // if layer_i can not execute and swap_tensor_j can not offloading, training is locked.
					#ifdef DEBUG
						printf("can not swap tensor%d-state%d-posistion%d in block[%d], training is locked in layer%d\n", 
								block_tensor->get_tensor_id(), block_tensor->get_data_state(), block_tensor->get_data_position(), block_tensor->get_swap_block_id(), layer_id);
						mem_controller->printSWAPBLOCK("can not swap tensor");
						while(1);
					#endif
						return false;
					}
				}
				else {  // still_swap_flag == true, re_tc_time > 0, last swap_tensor is still offloading
					// wait for last swap_tensor offloading done
					T_f += re_tc_time;
					T_c += re_tc_time;
					sync += re_tc_time;
					re_tc_time = 0.0;
				#ifdef DEBUG
					printf("re_tc_time = %f, ", re_tc_time);
				#endif
					still_swap_flag = false;
					(*swap_tensors)[j]->set_data_position(IN_CPU_GPU);
					block_id_j = (*swap_tensors)[j]->get_swap_block_id();
					mem_controller->set_swap_block_state(block_id_j, DONE);
				#ifdef DEBUG
					printf("complete offload tensor%d-layer%d-block[%d]\n", 
								(*swap_tensors)[j]->get_tensor_id(), (*swap_tensors)[j]->get_layer_id(), (*swap_tensors)[j]->get_swap_block_id());
				#endif
					j++;
				}	
			}
			else {  // if layer_i is complete, check what happen in swap_list during layer_i
			#ifdef DEBUG
				printf("complete layer%d forwardtime = %f\n", layer_id, layers_ft[layer_id]);
			#endif
				T_f += layers_ft[layer_id];  // recode T_f
				T_c += layers_ft[layer_id];  // recode T_c
				re_tf_time = layers_ft[layer_id];  // 
			#ifdef DEBUG	
				printf("re_tf_time = %f\n", re_tf_time);
			#endif
				while (true) {  
					// check what happen during layer_i
					// if (j >= prefetch_tensors->size()) break;
					if (j >= swap_tensors->size()) break;
					if (still_swap_flag == true) {
						if (re_tc_time > re_tf_time) {  // after layer_i done, last re_tc_time is still offloading
							re_tc_time = re_tc_time - re_tf_time;
						#ifdef DEBUG
							printf("re_tc_time = re_tc_time - re_tf_time = %f\n", re_tc_time);
						#endif
							still_swap_flag = true;
							break;
						} 
						else {
						#ifdef DEBUG
							printf("complete offload tensor%d-layer%d-block[%d]\n", 
								(*swap_tensors)[j]->get_tensor_id(), (*swap_tensors)[j]->get_layer_id(), (*swap_tensors)[j]->get_swap_block_id());
						#endif
							block_id_j = (*swap_tensors)[j]->get_swap_block_id();
							mem_controller->set_swap_block_state(block_id_j, DONE);
							(*swap_tensors)[j]->set_data_position(IN_CPU_GPU);
							j++;
							re_tf_time = re_tf_time - re_tc_time;
							re_tc_time = 0.0;
						#ifdef DEBUG
							printf("re_tf_time = re_tf_time - re_tc_time =  %f\n", re_tf_time);
						#endif
							still_swap_flag = false;
						}
					}
					else {  // when layer_i is executing, last re_tc_time complete offloading,
					#ifdef DEBUG
						printf("ssssss%d\n", 666);
						printf("swap_tensor%d-layer%d-block[%d]-data_position%d\n", 
							(*swap_tensors)[j]->get_tensor_id(), (*swap_tensors)[j]->get_layer_id(), (*swap_tensors)[j]->get_swap_block_id(), (*swap_tensors)[j]->get_data_position()); 
					#endif
						if ((*swap_tensors)[j]->get_data_position() == IN_GPU) {  // swap_tensor_j has been worked out, start offloading next swap_tensor
							if ((*swap_tensors)[j]->get_use_counter(BACKWARD) == 0 && (*swap_tensors)[j]->get_type() != RNN_RESERVE) {  // Just update the state without actually unloading 
							// if (false) {
								re_tc_time = 0.0;
								T_offload += re_tc_time;
								#ifdef DEBUG
									printf("start offload tensor%d-layer%d-block[%d]-size%zd-offload_time=%f useless in backward\n", 
										(*swap_tensors)[j]->get_tensor_id(), (*swap_tensors)[j]->get_layer_id(), (*swap_tensors)[j]->get_swap_block_id(), (*swap_tensors)[j]->get_mem_size(), 0.0);
									printf("re_tc_time = %f\n", 0.0);
								#endif
								(*swap_tensors)[j]->set_data_position(IN_CPU_GPU);
								block_id = (*swap_tensors)[j]->get_swap_block_id();
								mem_controller->set_swap_block_state(block_id, DONE);
								still_swap_flag = false;
								j++;
								#ifdef DEBUG
									mem_controller->printSWAPBLOCK("start offload useless tensor");
									printf("now swap_tensor%d-layer%d\n", (*swap_tensors)[j]->get_tensor_id(), (*swap_tensors)[j]->get_layer_id());
								#endif
								// block_id_j = (*swap_tensors)[j]->get_swap_block_id();

							}
							else {  // get_backward_useful == true
								// re_tc_time = (*swap_tensors)[j]->get_comm_time();
								re_tc_time = (*swap_tensors)[j]->get_swap_time(OFFLOAD);
								T_offload += re_tc_time;
								#ifdef DEBUG
									printf("start offload tensor%d-layer%d-block[%d]-size%zd-offload_time=%f\n", 
										(*swap_tensors)[j]->get_tensor_id(), (*swap_tensors)[j]->get_layer_id(), (*swap_tensors)[j]->get_swap_block_id(), (*swap_tensors)[j]->get_mem_size(), (*swap_tensors)[j]->get_swap_time(OFFLOAD));
									printf("re_tc_time = %f\n", re_tc_time);
								#endif
								still_swap_flag = true;
								block_id_j = (*swap_tensors)[j]->get_swap_block_id();
								// mem_controller->set_swap_block_state(block_id_j, DONE);  // swap_tensor_j is offloaded
								// j++;  // go to next swap_tensor
							}
						} 
						else {  // swap_tensor_j hasn't been worked out, can not start offloading swap_tensor_j+1
							re_tc_time = 0.0;
						#ifdef DEBUG
							printf("re_tc_time = %f\n", re_tc_time);
						#endif
							still_swap_flag = false;
							break;
						}
					}
				}
				// update t_out and t_in state
				// layer->increase_input_cur_use_counter(FORWARD);
				layer->fake_run(FORWARD, reg);
            	layer->update_input_state(FORWARD);
				if (layer->get_layer_structure() == SISO) {  
					t_out = ((base_network_layer_t<value_type>*)layer)->get_f_out();
					// do layer's forward computation
					if (t_out->get_position() == SHARED_GPU_POOL) {
						block_id = t_out->get_swap_block_id();
						block_tensor = mem_controller->get_swap_block_tensor(block_id);
						// printf("layer%d-type%d-tensor%d is swap, block_tensor%d\n", layer->get_base_id(), layer->get_layer_type(), t_out->get_tensor_id(), (block_tensor != NULL ? block_tensor->get_tensor_id() : -1));
						if (block_tensor != NULL) {  // set block's original tensor data_position
							// block_tensor->set_data_position(IN_CPU);
						}
						// swap_block[block_id] is ready, do layer's forward computation
						// mem_controller->set_swap_block(block_id, t_out);
					}
					t_out->set_data_position(IN_GPU);  // update t_out and t_in state
					t_out->set_data_state(STILL_USED);
					if (layer->get_layer_type() == RNN || layer->get_layer_type() == SATTN) {
						// tensor_t<value_type>* reserve_buff = ((rnn_layer_t<value_type>*)layer)->get_rnn_reserve_buff();
						tensor_t<value_type>* reserve_buff = ((base_network_layer_t<value_type>*)layer)->get_reserve_buff();
						reserve_buff->set_data_position(IN_GPU);
						reserve_buff->set_data_state(FORWARD_DELETE_OK);
					}
					// t_in = ((base_network_layer_t<value_type>*)layer)->get_f_in();
					// t_in->decrease_use_count();
					// if (t_in->is_cur_use_count_zero()) {
					// 	t_in->set_data_state(FORWARD_DELETE_OK);
					// }
				} 
				else {  // MIMO layer, update t_outs state
					t_outs = ((base_structure_t<value_type>*)layer)->get_outputs();
					// if (t_outs[0]->get_position() == SHARED_GPU_POOL) {
					// 	block_id = t_outs[0]->get_swap_block_id();
					// 	block_tensor = mem_controller->get_swap_block_tensor(block_id);
					// 	if (block_tensor != NULL) {  // set block's original tensor data_position
					// 		block_tensor->set_data_position(IN_CPU);
					// 	}
					// 	// swap_block[block_id] is ready, do layer's forward computation
					// 	// mem_controller->set_swap_block(block_id, t_outs[0]);
					// }
					// do layer's forward computation
					for (int k = 0; k < t_outs.size(); k++) {
						block_id = t_outs[k]->get_swap_block_id();
						if (t_outs[k]->get_position() == SHARED_GPU_POOL) {
							block_tensor = mem_controller->get_swap_block_tensor(block_id);
							// printf("layer%d-type%d-tensor%d is swap, block_tensor%d\n", layer->get_base_id(), layer->get_layer_type(), t_outs[k]->get_tensor_id(), (block_tensor != NULL ? block_tensor->get_tensor_id() : -1));
							if (block_tensor != NULL) {  // set block's original tensor data_position
								// block_tensor->set_data_position(IN_CPU);
							}
						}
						t_outs[k]->set_data_position(IN_GPU);  // update t_out and t_in state
						t_outs[k]->set_data_state(STILL_USED);
					}
					// t_ins = ((base_structure_t<value_type>*)layer)->get_inputs();
					// for (int k = 0; k < t_ins.size(); k++) {
					// 	t_ins[k]->decrease_use_count();
					// 	if (t_ins[k]->is_cur_use_count_zero()) {
					// 		t_ins[k]->set_data_state(FORWARD_DELETE_OK);
					// 	}
					// }
				}
			#ifdef DEBUG
				printf("complete updata layer%d\n", layer_id);
				mem_controller->printSWAPBLOCK("forward done");
				mem_controller->printRecomputePool("forward done");
			#endif
			
				i++;  // since layer_i is complete, go to the layer_i+1
				if ((*net_comp_route)[i].second == BACKWARD) {  // Forward propagation done
					*forward = T_f;
					*offload_time = T_offload;
					*sync_time_f = sync;
					sync = 0.0;
					// mem_controller->reset_swap_block();
					// j = 0 + SWAP_BLOCK_NUM - 1;  // reset the index of swap_tensor, the first SWAP_BLOCK_NUM prefetch_tensors is not need to fetech since they are still in block
					j = 0;
					still_swap_flag = false;
					// (*prefetch_tensors)[j]->set_data_position(IN_GPU);
					// for (int p = 0; p < prefetch_tensors->size(); p++) {
					// 	tensor_t<value_type>* tttt = (*prefetch_tensors)[p];
					// 	base_layer_t<value_type>* layer = (base_layer_t<value_type>*) layers.find(tttt->get_layer_id())->second;
					// 	printf("tensor%d-layer%d-type%d bu%d tp%d dp%d\n", tttt->get_tensor_id(), tttt->get_layer_id(), layer->get_layer_type(), tttt->get_use_counter(BACKWARD), tttt->get_position(), tttt->get_data_position());
					// }
					// while(1);
				} 
			}
		}
		else {  // Backward propagation
			// printf("Forward Simulation done, T_f = %lf\n", T_f);
			#ifdef DEBUG
				printf("start layer%d backward\n", layer_id);
			#endif
			if (layer->get_layer_type() == DATA_L) {  // Backward done, DATA_L is always the last layer in Backward
				*backward = T_b;
				// printf("bplus_counter = %d, T_b = %f, re_time = %f, \n", bplus_counter, T_b, re_time);
				double sum_time = 0;
				double re_sum = 0;
				int rc = 0;
				int rc2 = 0;
				int bpc = 0;
				int busecount = 0;
				// std::vector<tensor_t<value_type>* >* all_tensors = reg->get_vector();
				// for (size_t i = 0; i < all_tensors->size(); i++) {
				// 	t_out = (*all_tensors)[i];
				// 	if (t_out->get_position() == RECOMPUTE_IN_BACKWARD) {
				// 		// sum_time += layers_ft[l];
				// 		rc2++;
				// 		if (t_out->get_use_counter(BACKWARD) > 0) {
				// 			busecount++;
				// 		}
				// 	}
				// }
				// for (int l = 1; l < net_layers->size(); l++) {
				// 	sum_time += layers_bt[l];
				// 	bpc++;
				// 	layer = (base_layer_t<value_type>*)(net_layers->find(l)->second);
				// 	// printf("%d-layer%d-type%d, ", l, layer->get_base_id(), layer->get_layer_type());
				// 	if (layer->get_layer_structure() == SISO && layer->get_layer_type() != DATA_L && layer->get_layer_type() != SOFTMAX) {
				// 		t_out = ((base_network_layer_t<value_type>*)layer)->get_f_out();
				// 		if (t_out->get_position() == RECOMPUTE_IN_BACKWARD) {
				// 			// sum_time += layers_ft[l];
				// 			re_sum += layers_ft[l];
				// 			rc++;
				// 		}
				// 	}
					
				// }
				// printf("bpc = %d, rc = %d, b_sum_time = %f, re_sum = %f, rc2 = %d, busecount = %d\n", bpc, rc, sum_time, re_sum, rc2, busecount);
				// *sync_time_b = sync;
				// *fetch_time = T_fetch;
				// mem_controller->reset_swap_block();
				// while(1);
				break;
			}
			if (layer->get_layer_structure() == SISO) {
				reserve_buff_ready = true;
				if (layer->get_layer_type() == RNN || layer->get_layer_type() == SATTN) {
					// tensor_t<value_type>* reserve_buff = ((rnn_layer_t<value_type>*)layer)->get_rnn_reserve_buff();
					tensor_t<value_type>* reserve_buff = ((base_network_layer_t<value_type>*)layer)->get_reserve_buff();
					block_id = reserve_buff->get_prefetch_block_id();
					if (reserve_buff->get_position() == SHARED_GPU_POOL) {
						if (reserve_buff->get_data_position() == IN_CPU_GPU || reserve_buff->get_data_position() == IN_GPU) {  // if t_in has been fetched
						#ifdef DEBUG
							printf("layer%d's reserve_buff%d has been fetched into block[%d]\n", layer_id, reserve_buff->get_tensor_id(), reserve_buff->get_prefetch_block_id());
						#endif
							reserve_buff_ready = true;
						}
						else {  // if reserve_buff has not been fetched
						#ifdef DEBUG
							printf("layer%d's reserve_buff-tensor%d has not been fetched into block[%d]\n", layer_id, reserve_buff->get_tensor_id(), reserve_buff->get_prefetch_block_id());
						#endif
							reserve_buff_ready = false;
						}
					}
				}
				if (reserve_buff_ready) {
					t_in = ((base_network_layer_t<value_type>*)layer)->get_f_in();
					if (t_in->get_position() == SHARED_GPU_POOL) {
						if (t_in->get_data_position() == IN_CPU_GPU || t_in->get_data_position() == IN_GPU) {  // if t_in has been fetched
						#ifdef DEBUG
							printf("layer%d's t_in%d has been fetched into block[%d]\n", layer_id, t_in->get_tensor_id(), t_in->get_prefetch_block_id());
						#endif
							// block_id_j = (*prefetch_tensors)[j]->get_swap_block_id();
							t_in->set_data_position(IN_GPU);
							t_in->set_data_state(STILL_USED);
							if (t_in == (*prefetch_tensors)[j]) {
								mem_controller->set_swap_block(block_id_j, (*prefetch_tensors)[j]);
								mem_controller->set_swap_block_state(block_id_j, DONE);  // swap_tensor_j is offloaded
								j++;
							}
							else {
								// printf("prefetch_list error! t_in%d-layer%d should not in this order\n", t_in->get_tensor_id(), t_in->get_layer_id());
							}
							// (*prefetch_tensors)[j]->set_data_position(IN_GPU);
							is_layer_done = true;
						}
						else {  // if t_in has not been fetched
						#ifdef DEBUG
							printf("layer%d's t_in-tensor%d has not been fetched into block[%d]\n", layer_id, t_in->get_tensor_id(), t_in->get_prefetch_block_id());
						#endif
							is_layer_done = false;
						}
					}
					else if (t_in->get_data_position() == DELETED) {
						#ifdef DEBUG
							printf("layer%d's t_in%d has been deleted which need to be recomputed(layer%d)\n", layer_id, t_in->get_tensor_id(), t_in->get_layer_id());
						#endif
						re_layer = (base_layer_t<value_type>*)(net_layers->find(t_in->get_layer_id())->second);
						cur_recomp_time = 0;
						is_re_done = false;
						is_re_done = GetRecomputeTime(net_layers, t_in, re_layer, (*net_comp_route)[i].second, &cur_recomp_time);
						#ifdef DEBUG
							printf("is_re_done=%d, layer%d's t_in%d-layer%d-p%d-s%d, cur_recomp_time=%f\n", is_re_done, layer_id, t_in->get_tensor_id(), t_in->get_layer_id(), t_in->get_data_position(), t_in->get_data_state(), cur_recomp_time);
						#endif
						T_b += cur_recomp_time;
						re_time += cur_recomp_time;
						is_layer_done = false;  // layer's backward has not been executed, but t_in's recomputation may be completed.
					}
					else {
						is_layer_done = true;
					}
				}
				else {
					is_layer_done = false;
				}
			}
			else {  // MIMO LAYER
				t_ins = ((base_structure_t<value_type>*)layer)->get_inputs();
				if (layer->get_layer_type() == JOIN_L || layer->get_layer_type() == FORK_L || layer->get_layer_type() == CONCAT ) {
					is_layer_done = true;
				}
				else {
					for (int k = t_ins.size()-1; k >= 0; k--) {
						if (t_ins[k]->get_position() == SHARED_GPU_POOL) {
							if (t_ins[k]->get_data_position() == IN_CPU_GPU || t_ins[k]->get_data_position() == IN_GPU) {
							#ifdef DEBUG
								printf("layer%d's t_ins[%d]-tensor%d has been fetched into block[%d]\n", layer_id, k, t_ins[k]->get_tensor_id(), t_ins[k]->get_swap_block_id());
							#endif
								t_ins[k]->set_data_position(IN_GPU);
								t_ins[k]->set_data_state(STILL_USED);
								if (t_ins[k] == (*prefetch_tensors)[j]) {
									mem_controller->set_swap_block(block_id_j, (*prefetch_tensors)[j]);
									mem_controller->set_swap_block_state(block_id_j, DONE);  // swap_tensor_j is offloaded
									j++;
								}
								else {
									printf("prefetch_list error! t_ins[%d]%d-layer%d should not in this order\n", k, t_ins[k]->get_tensor_id(), t_ins[k]->get_layer_id());
								}
								is_layer_done = true;
							}
							else {
							#ifdef DEBUG
								printf("layer%d's t_ins[%d]-tensor%d has not been fetched into block[%d]\n", layer_id, k, t_ins[k]->get_tensor_id(), t_ins[k]->get_swap_block_id());
							#endif
								is_layer_done = false;
								break;
							}
						}
						else if (t_ins[k]->get_data_position() == DELETED) {
							re_layer = (base_layer_t<value_type>*)(net_layers->find(t_ins[k]->get_layer_id())->second);
							cur_recomp_time = 0;
							is_re_done = false;
							is_re_done = GetRecomputeTime(net_layers, t_ins[k], re_layer, (*net_comp_route)[i].second, &cur_recomp_time);
							re_time += cur_recomp_time;
							T_b += cur_recomp_time;
							is_layer_done = false; 
						}
						else {
							is_layer_done = true;
						}
					}
				}
			}
			if (is_layer_done == false) {  // layer_i backward false, t_in has not been prefetched
				if (is_re_done == false) {  // neither backward nor recomputation done, wait for prefetching
				#ifdef DEBUG
					printf("locked layer%d backward\n", layer_id);
				#endif
					// if swap_block[block_id] is not prefetch done, Wait for the previous prefetch_tensor fetch to complete
					if (still_swap_flag == false) {  // no tensor is fetching now
						if ((*prefetch_tensors)[j]->get_use_counter(BACKWARD) == 0) {
							j++;
						}
						// if (j >= prefetch_tensors->size()) break;
						if (j > prefetch_tensors->size()) j = prefetch_tensors->size();
						if ((*prefetch_tensors)[j]->get_data_position() == IN_CPU_GPU 
							|| (*prefetch_tensors)[j]->get_data_position() == IN_CPU_GPU 
							|| (*prefetch_tensors)[j]->get_data_state() == NO_COMPUTED) {
							j++;  // if prefetch_tensors[j] need not used in backward, pass
						}
						// if (j >= prefetch_tensors->size()) break;
						if (j > prefetch_tensors->size()) j = prefetch_tensors->size();

						block_tensor = mem_controller->get_swap_block_tensor((*prefetch_tensors)[j]->get_prefetch_block_id());
					#ifdef DEBUG
						printf("prefetch_tensors%d-position%d\n", (*prefetch_tensors)[j]->get_tensor_id(), (*prefetch_tensors)[j]->get_data_position());
					#endif
						if (block_tensor->get_data_state() == NO_COMPUTED) {
							block_id_j = (*prefetch_tensors)[j]->get_swap_block_id();
							mem_controller->set_swap_block(block_id_j, (*prefetch_tensors)[j]);
						#ifdef DEBUG
							printf("start fetch tensor%d-layer%d-block[%d]-fetch_time=%f\n", 
									(*prefetch_tensors)[j]->get_tensor_id(), (*prefetch_tensors)[j]->get_layer_id(), (*prefetch_tensors)[j]->get_prefetch_block_id(), (*prefetch_tensors)[j]->get_swap_time(FETCH));
						#endif	
							// re_tp_time = (*prefetch_tensors)[j]->get_comm_time(); 
							(*prefetch_tensors)[j]->set_data_state(STILL_USED);
							re_tp_time = (*prefetch_tensors)[j]->get_swap_time(FETCH);
							T_fetch += re_tp_time;
							T_b += re_tp_time;			
							T_p += re_tp_time;
							sync += re_tp_time;
							mem_controller->set_swap_block_state(block_id_j, DONE);  // swap_tensor_j is offloaded
							(*prefetch_tensors)[j]->set_data_position(IN_GPU);
						#ifdef DEBUG
							printf("re_tp_time = %f", re_tp_time);
							printf("a complete fetch tensor%d-layer%d-block[%d]\n", 
									(*prefetch_tensors)[j]->get_tensor_id(), (*prefetch_tensors)[j]->get_layer_id(), (*prefetch_tensors)[j]->get_prefetch_block_id());
						#endif
							j++;  // go to next swap_tensor
							still_swap_flag = false;  // before swap_tensor_j offloaded, trainning is waiting, so still_swap_flag is false
						}
						else {  // if layer_i can not execute and swap_tensor_j can not offloading, training is locked.
						#ifdef DEBUG
							printf("can not fetch tensor%d, original_tensor%d-state%d-posistion%d in block[%d], training is locked in back layer%d\n", 
									(*prefetch_tensors)[j]->get_tensor_id(), block_tensor->get_tensor_id(), block_tensor->get_data_state(), block_tensor->get_data_position(), block_tensor->get_prefetch_block_id(), layer_id);
							while(1);
						#endif
							return false;
						}
					}
					else {  // still_swap_flag == true, re_tp_time > 0, last prefetch_tensor is still fetching
						// wait for last prefetch_tensor fetching done
						T_b += re_tp_time;
						T_p += re_tp_time;
						sync += re_tp_time;
						re_tp_time = 0.0;
						still_swap_flag = false;
						(*prefetch_tensors)[j]->set_data_position(IN_GPU);
						block_id_j = (*prefetch_tensors)[j]->get_prefetch_block_id();
						mem_controller->set_swap_block_state(block_id_j, DONE);
					#ifdef DEBUG
						printf("re_tp_time = %f", re_tp_time);
						printf("b complete fetch tensor%d-layer%d-block[%d]\n", 
								(*prefetch_tensors)[j]->get_tensor_id(), (*prefetch_tensors)[j]->get_layer_id(), (*prefetch_tensors)[j]->get_prefetch_block_id());
					#endif
						j++;
					}
				}
				else {  // is_re_done=true, backward stop but recomputation done, check the how much data has been prefetching during t_in's recomputing
					// is_re_done = GetRecomputeTime(net_layers, re_layer, (*net_comp_route)[i].second, &cur_recomp_time);
					// is_layer_done == false;  // layer's backward has not been executed, but t_in's recomputation may be completed.
					is_re_done = false;
					#ifdef DEBUG
						printf("complete layer%d recomputation time = %f\n", re_layer->get_base_id(), layers_ft[re_layer->get_base_id()]);
					#endif
					// T_b += cur_recomp_time;  // recode recomputing time
					while (true) {
						if (j >= prefetch_tensors->size()) {
							*fetch_time = T_fetch;
							break;  //  complete fetch all tensors
						}
						if (still_swap_flag == true) {
							if (re_tp_time > cur_recomp_time) {  // after re_layer recomputation done, last re_tp_time is still fetching
								re_tp_time = re_tp_time - cur_recomp_time;
								#ifdef DEBUG
									printf("re_tp_time = re_tp_time - cur_recomp_time = %f\n", re_tp_time);
								#endif
								break;
							}
							else {  // after re_layer recomputation done, last prefetching tensor has been prefetch.
								#ifdef DEBUG
									printf("backward complete fetch tensor%d-layer%d-block[%d]\n", 
										(*prefetch_tensors)[j]->get_tensor_id(), (*prefetch_tensors)[j]->get_layer_id(), (*prefetch_tensors)[j]->get_prefetch_block_id());
								#endif
								block_id_j = (*prefetch_tensors)[j]->get_prefetch_block_id();
								mem_controller->set_swap_block_state(block_id_j, DONE);
								(*prefetch_tensors)[j]->set_data_position(IN_GPU);
								cur_recomp_time = cur_recomp_time - re_tp_time;  // update Remaining recomputation time
								re_tp_time = 0.0;
								j++;
								#ifdef DEBUG
									printf("cur_recomp_time = cur_recomp_time - re_tp_time =  %f\n", cur_recomp_time);
									mem_controller->printSWAPBLOCK("prefetch done");
								#endif
								still_swap_flag = false;								
							}
						}
						else {  // still_swap_flag = false, after re_layer is recomputed, last prefetching tensor has been prefetch, check how many new tensors have been prefetching
							if ((*prefetch_tensors)[j]->get_use_counter(BACKWARD) == 0 && (*prefetch_tensors)[j]->get_type() != RNN_RESERVE) {
								j++;
							}
							if (j >= prefetch_tensors->size()) break;
							// if (false) {
							if ((*prefetch_tensors)[j]->get_data_position() == IN_CPU_GPU 
								|| (*prefetch_tensors)[j]->get_data_position() == IN_CPU_GPU 
								|| (*prefetch_tensors)[j]->get_data_state() == NO_COMPUTED) {
								j++;  // if prefetch_tensors[j] need not used in backward, pass
							}
							if (j >= prefetch_tensors->size()) break;
							#ifdef DEBUG
								printf("prefetch_tensors%d-layer%d-block[%d]-data_position%d\n", 
									(*prefetch_tensors)[j]->get_tensor_id(), (*prefetch_tensors)[j]->get_layer_id(), (*prefetch_tensors)[j]->get_prefetch_block_id(), (*prefetch_tensors)[j]->get_data_position()); 
							#endif
							block_tensor = mem_controller->get_swap_block_tensor((*prefetch_tensors)[j]->get_prefetch_block_id());
							block_id_j = (*prefetch_tensors)[j]->get_prefetch_block_id();
							if (block_tensor->get_data_state() == NO_COMPUTED) {  // original tensor in block is useless
								mem_controller->set_swap_block(block_id_j, (*prefetch_tensors)[j]);
							#ifdef DEBUG
								printf("start fetch tensor%d-layer%d-block[%d]-fetch_time=%f\n", 
									(*prefetch_tensors)[j]->get_tensor_id(), (*prefetch_tensors)[j]->get_layer_id(), (*prefetch_tensors)[j]->get_prefetch_block_id(), (*prefetch_tensors)[j]->get_swap_time(FETCH));
								// printf("start fetch tensor%d-layer%d-block[%d]-comm_time=%f\n", 
								// 	(*prefetch_tensors)[j]->get_tensor_id(), (*prefetch_tensors)[j]->get_layer_id(), (*prefetch_tensors)[j]->get_prefetch_block_id(), (*prefetch_tensors)[j]->get_comm_time());
								printf("re_tp_time = %f\n", re_tp_time);
							#endif
								(*prefetch_tensors)[j]->set_data_state(STILL_USED);
								mem_controller->set_swap_block_state(block_id_j, READY);
								re_tp_time = (*prefetch_tensors)[j]->get_swap_time(FETCH);  
								T_fetch += re_tp_time;
								still_swap_flag = true;
							}
							else { 
								re_tp_time = 0.0;
							#ifdef DEBUG
								printf("can not fetch tensor%d-block[%d] in layer%d backward , wait for original tensor%d-layer%d become useless\n", 
									(*prefetch_tensors)[j]->get_tensor_id(), (*prefetch_tensors)[j]->get_prefetch_block_id(), layer_id, block_tensor->get_tensor_id(), block_tensor->get_layer_id());
								printf("re_tp_time = %f\n", re_tp_time);
							#endif
								still_swap_flag = false;
								break;
							}
						}
					}
				}
			}
			else {  // layer_i backward done, check the how much data has been prefetching during layer_i backward
			#ifdef DEBUG
				printf("complete layer%d backward time = %f\n", layer_id, layers_bt[layer_id]);
				printf("re_tb_time = %f\n", layers_bt[layer_id]);
			#endif
				// printf("complete layer%d backward time = %f\n", layer_id, layers_bt[layer_id]);
				// if layer_i is complete, check what happen in swap_list during layer_i
				T_b += layers_bt[layer_id];  // recode T_f
				bplus_counter++;
				T_p += layers_bt[layer_id];  // recode T_p mean comm stream work for T_p time
				re_tb_time = layers_bt[layer_id];  // 
				while (true) {  
					// check what happen during layer_i backward
					if (j >= prefetch_tensors->size()) {
						*fetch_time = T_fetch;
						break;  //  complete fetch all tensors 
					}
					
					if (still_swap_flag == true) {
						if (re_tp_time > re_tb_time) {  // after layer_i done, last re_tp_time is still fetching
							re_tp_time = re_tp_time - re_tb_time;
						#ifdef DEBUG
							printf("re_tp_time = re_tp_time - re_tb_time = %f\n", re_tp_time);
						#endif
							break;
						} 
						else {
						#ifdef DEBUG
							printf("backward complete fetch tensor%d-layer%d-block[%d]\n", 
								(*prefetch_tensors)[j]->get_tensor_id(), (*prefetch_tensors)[j]->get_layer_id(), (*prefetch_tensors)[j]->get_prefetch_block_id());
						#endif
							block_id_j = (*prefetch_tensors)[j]->get_prefetch_block_id();
							mem_controller->set_swap_block_state(block_id_j, DONE);
							(*prefetch_tensors)[j]->set_data_position(IN_GPU);
							j++;
							re_tb_time = re_tb_time - re_tp_time;
							re_tp_time = 0.0;
						#ifdef DEBUG
							printf("re_tb_time = re_tb_time - re_tp_time =  %f\n", re_tb_time);
						#endif
							still_swap_flag = false;
						}
					}
					else {  // when layer_i is executing, last re_tp_time complete fetching,
						if ((*prefetch_tensors)[j]->get_data_position() == IN_CPU_GPU 
							|| (*prefetch_tensors)[j]->get_data_position() == IN_GPU 
							|| (*prefetch_tensors)[j]->get_data_state() == NO_COMPUTED) 
						{
							j++;  // (*prefetch_tensors)[j] is in GPU, go to next prefetch_tensor, it usually happen in the last few tensors

						}
						else {  // (*prefetch_tensors)[j] is not in GPU, so it need to be fetched
						#ifdef DEBUG
							printf("prefetch_tensors%d-layer%d-backusecount(%d)-block[%d]-data_position%d\n", 
								(*prefetch_tensors)[j]->get_tensor_id(), (*prefetch_tensors)[j]->get_layer_id(), (*prefetch_tensors)[j]->get_use_counter(BACKWARD), (*prefetch_tensors)[j]->get_prefetch_block_id(), (*prefetch_tensors)[j]->get_data_position()); 
						#endif
							block_tensor = mem_controller->get_swap_block_tensor((*prefetch_tensors)[j]->get_prefetch_block_id());
							block_id_j = (*prefetch_tensors)[j]->get_prefetch_block_id();
							if (block_tensor->get_data_state() == NO_COMPUTED) {  // original tensor in block is useless
								mem_controller->set_swap_block(block_id_j, (*prefetch_tensors)[j]);
							#ifdef DEBUG
								printf("start fetch tensor%d-layer%d-block[%d]-fetch_time=%f\n", 
									(*prefetch_tensors)[j]->get_tensor_id(), (*prefetch_tensors)[j]->get_layer_id(), (*prefetch_tensors)[j]->get_prefetch_block_id(), (*prefetch_tensors)[j]->get_swap_time(FETCH));
								printf("re_tp_time = %f\n", re_tp_time);
							#endif
								// re_tp_time = (*prefetch_tensors)[j]->get_comm_time();  
								(*prefetch_tensors)[j]->set_data_state(STILL_USED);
								mem_controller->set_swap_block_state(block_id_j, READY);
								re_tp_time = (*prefetch_tensors)[j]->get_swap_time(FETCH);  
								T_fetch += re_tp_time;
								// T_b += re_tp_time;
								// T_p += re_tp_time;
								still_swap_flag = true;
							}
							else { 
								// printf("layer%d\n", layer_id);
								re_tp_time = 0.0;
							#ifdef DEBUG
								printf("can not fetch tensor%d-block[%d] in layer%d backward , wait for original tensor%d-layer%d become useless\n", 
									(*prefetch_tensors)[j]->get_tensor_id(), (*prefetch_tensors)[j]->get_prefetch_block_id(), layer_id, block_tensor->get_tensor_id(), block_tensor->get_layer_id());
								printf("re_tp_time = %f\n", re_tp_time);
							#endif
								still_swap_flag = false;
								break;
							}
						}
					}
				}
				// update tensor state
				// layer->increase_input_cur_use_counter(BACKWARD);
				// mem_controller->printSWAPBLOCK("backward done");
				layer->fake_run(BACKWARD, reg);
            	layer->update_input_state(BACKWARD);
				layer->update_output_state(BACKWARD);
				if (layer->get_layer_type() == RNN || layer->get_layer_type() == SATTN) {
					// tensor_t<value_type>* reserve_buff = ((rnn_layer_t<value_type>*)layer)->get_rnn_reserve_buff();
					tensor_t<value_type>* reserve_buff = ((base_network_layer_t<value_type>*)layer)->get_reserve_buff();
					reserve_buff->set_data_state(NO_COMPUTED);
                	reserve_buff->set_data_position(NO_DATA);
				}
				#ifdef DEBUG
					mem_controller->printSWAPBLOCK("backward done");
					printf("Cumulative backpropagation time = %f\n", T_b);
				#endif
					// printf("Cumulative backpropagation time = %f\n", T_b);
				i++;
			#ifdef DEBUG
				printf("complete updata layer%d\n", layer_id);
			#endif
			}
		}
	}
	*forward = T_f;
	*backward = T_b;
	// while(1);
	return true;
}
// #undef DEBUG

// #define DEBUG
template <class value_type>
size_t ATPSearch<value_type>::SetSwappingTensors(bool* swapping_code, std::vector<tensor_t<value_type>*>* alternative_swap_tensors, 
											std::vector<tensor_t<value_type>*>* swap_tensors, std::vector<tensor_t<value_type>*>* prefetch_tensors) {
	std::map<int, void* >* net_layers  = reg->get_net_layers_ptr();
	std::vector<std::pair<int, net_comp> >* net_comp_route = reg->get_net_comp_route_ptr();
	base_layer_t<value_type>* layer;
	base_structure_t<value_type>* structure_layer;
	base_network_layer_t<value_type>* net_layer;
	
	size_t swapping_size = 0;
	int swap_layer_num = 0;

	for (size_t i = 0; i < net_comp_route->size(); i++) {  // reset data state
		if ((*net_comp_route)[i].second == FORWARD) {
			int layer_id = (*net_comp_route)[i].first;
			base_layer_t<value_type>* layer = (base_layer_t<value_type>*) net_layers->find(layer_id)->second;
			if (layer->get_layer_structure() == MIMO) {
				structure_layer = (base_structure_t<value_type>* )layer;
				// printf("layer%d-type%d, inputs_size = %d, outputs_size = %d\n", layer->get_base_id(), layer->get_layer_type(), structure_layer->get_inputs().size(), structure_layer->get_outputs().size());
				std::vector<tensor_t<value_type>* > tensors = structure_layer->get_outputs();
				for (int j = 0; j < tensors.size(); j++) {
					tensors[j]->reset_all_data_state();
				}
			}
			else {
				if (layer->get_layer_type() == RNN || layer->get_layer_type() == SATTN) {
					// tensor_t<value_type>* tensor = ((rnn_layer_t<value_type>*)layer)->get_rnn_reserve_buff();
					tensor_t<value_type>* tensor = ((base_network_layer_t<value_type>*)layer)->get_reserve_buff();
					tensor->reset_all_data_state();
				}
				net_layer = (base_network_layer_t<value_type>* )layer;
				tensor_t<value_type>* tensor = net_layer->get_f_out();
				tensor->reset_all_data_state();
			}
		}
	}
	// printf("SetSwappingTensors swapping_code: ");
	for (int i = 0; i < alternative_swap_tensors->size(); i++) {
		// printf("%d,", swapping_code[i]);
	}
	// printf("\n");
	// printf("alternative_swap_tensors.size() = %d: ", alternative_swap_tensors->size());
	for (size_t i = 0; i < alternative_swap_tensors->size(); i++) {  // reset alternative_swap_tensors position
		(*alternative_swap_tensors)[i]->set_position(REMAIN_IN_GPU);
		// layer = (base_layer_t<value_type>*) net_layers->find((*alternative_swap_tensors)[i]->get_layer_id())->second;
		// layer->set_layer_position(REMAIN_LAYER);
		// printf("s%dlayer%d-type%d-p", swapping_code[i], layer->get_base_id(), layer->get_layer_type());
		if (swapping_code[i] == true) {
			(*alternative_swap_tensors)[i]->set_position(SHARED_GPU_POOL);
			// layer->set_layer_position(SWAP_LAYER);
			swap_layer_num++;
			// swap_tensors->push_back((*alternative_swap_tensors)[i]);
			swapping_size += (*alternative_swap_tensors)[i]->get_mem_size();
			// printf("%d", 1);
		}
		// printf(", ");
	}
	// printf("\n");
	std::map<int, void* > layers = reg->get_net_layers();
	std::vector<std::pair<int, net_comp> > net_route = reg->get_net_comp_route();
	// printf("SetSwappingTensors CODE size = %d:\n", layers.size());
	for (size_t i = 0; i < layers.size(); i++) {  // set rs_code_route
		if (net_route[i].second == FORWARD) {
			base_layer_t<value_type>* layer = (base_layer_t<value_type>*) layers.find(net_route[i].first)->second;
			if (layer->get_layer_position() == SWAP_LAYER) {
				// rs_code_route[i] = 1;
				// printf("%d,", 1);
			} 
			else if (layer->get_layer_position() == RECOMPUTE_LAYER) {
				// printf("%d,", 2);
			}
			else {
				// rs_code_route[i] = 0;
				// printf("%d,", 0);
			}
			
		}
	}
	swap_tensors->clear();
	prefetch_tensors->clear();
	tensor_t<value_type>* tensor; 
	std::vector<tensor_t<value_type>*> tensors; 
	for (size_t i = 0; i < net_comp_route->size(); i++) {  // reset tensor state
		if ((*net_comp_route)[i].second == FORWARD) {
			size_t layer_id = (*net_comp_route)[i].first;
			base_layer_t<value_type>* layer = (base_layer_t<value_type>*) net_layers->find(layer_id)->second;
			if (layer->get_layer_structure() == MIMO) {
				structure_layer = (base_structure_t<value_type>* )layer;
				tensors = (structure_layer->get_outputs());
				for (int j = 0; j < tensors.size(); j++) {
					if (tensors[j]->get_position() == SHARED_GPU_POOL) {
						// if (tensors[j]->get_backward_useful() == true) {
						// if (tensors[j]->get_use_counter(BACKWARD) > 0) 
						{
							auto iter = std::find(swap_tensors->begin(), swap_tensors->end(), tensors[j]);
							if (iter == swap_tensors->end()) {  // Prevent duplicate addition
								swap_tensors->push_back(tensors[j]);
							}
						}
					}
				}					
			}
			else {
				if (layer->get_layer_type() != DATA_L) {
					if (layer->get_layer_type() == RNN || layer->get_layer_type() == SATTN) {
						// tensor = ((rnn_layer_t<value_type>*)layer)->get_rnn_reserve_buff();
						tensor = ((base_network_layer_t<value_type>*)layer)->get_reserve_buff();
						if (tensor->get_position() == SHARED_GPU_POOL) {
							swap_tensors->push_back(tensor);
						}
					}
					net_layer = (base_network_layer_t<value_type>* )layer;
					tensor = net_layer->get_f_out();	
					if (tensor->get_position() == SHARED_GPU_POOL) {
						swap_tensors->push_back(tensor);
					}
				}
			}
		}
		else {  // (*net_comp_route)[(*swap_layers)[i]].second == BACKWARD
			size_t layer_id = (*net_comp_route)[i].first;
            base_layer_t<value_type>* layer = (base_layer_t<value_type>*) net_layers->find(layer_id)->second;
            if (layer->get_layer_structure() == MIMO) {
                std::vector<tensor_t<value_type>*> t_ins = ((base_structure_t<value_type>*)layer)->get_inputs();
                for (int j = t_ins.size()-1; j >= 0; j--) {
                    if (t_ins[j]->get_position() == SHARED_GPU_POOL) {
						// if (t_ins[j]->get_backward_useful() == true) {
						if (t_ins[j]->get_use_counter(BACKWARD) > 0 || t_ins[j]->get_type() == RNN_RESERVE) {
							auto iter = std::find(prefetch_tensors->begin(), prefetch_tensors->end(), t_ins[j]);
							if (iter == prefetch_tensors->end()) {  // Prevent duplicate addition
								prefetch_tensors->push_back(t_ins[j]);
							}
						}
                    }
                }
            }
            else {  // SISO Layer
                if (layer->get_layer_type() != DATA_L) {
					if (layer->get_layer_type() == RNN || layer->get_layer_type() == SATTN) {
						// tensor_t<value_type>* reserve_buff = ((rnn_layer_t<value_type>*)layer)->get_rnn_reserve_buff();
						tensor_t<value_type>* reserve_buff = ((base_network_layer_t<value_type>*)layer)->get_reserve_buff();
						if (reserve_buff->get_position() == SHARED_GPU_POOL) {
                        	prefetch_tensors->push_back(reserve_buff);
                    	}
					}
                    tensor_t<value_type>* t_in = ((base_network_layer_t<value_type>*)layer)->get_f_in();
                    if (t_in->get_position() == SHARED_GPU_POOL) {
						auto iter = std::find(prefetch_tensors->begin(), prefetch_tensors->end(), t_in);
						if (iter == prefetch_tensors->end()) {  // Prevent duplicate addition
							prefetch_tensors->push_back(t_in);
						}
                    }
                }
            }
		}
	}
	
	// printf("\n");
	#ifdef DEBUG
		printf("set swap_tensors and prefetch_tensors done, swap_tensors_size = %f, swap_tensors.size=%d, prefetch_tensors.size=%d\n", 
			BYTE_TO_MB(swapping_size), swap_tensors->size(), prefetch_tensors->size());
	#endif
	return swapping_size;
}
// #undef DEBUG

// #define DEBUG
template <class value_type>
bool ATPSearch<value_type>::IsSwappingLegal(std::vector<tensor_t<value_type>*>* swap_tensors, std::vector<tensor_t<value_type>*>* prefetch_tensors) {
#ifdef DEBUG
	for (int i = 0; i < swap_tensors->size(); i++) {
		printf("before preallocate swap_tensors%d-layer%d-block[%d] %d/%d\n", (*swap_tensors)[i]->get_tensor_id(), (*swap_tensors)[i]->get_layer_id(), (*swap_tensors)[i]->get_swap_block_id(), i, swap_tensors->size());
	}
#endif
	if (!mem_controller->PreAllocateRecompteSwapBlock(&this->swap_tensors, &this->prefetch_tensors)) {
		return false;
	}
#ifdef DEBUG
	for (int i = 0; i < swap_tensors->size(); i++) {
		printf("after preallocate swap_tensors%d-layer%d-block[%d]\n", (*swap_tensors)[i]->get_tensor_id(), (*swap_tensors)[i]->get_layer_id(), (*swap_tensors)[i]->get_swap_block_id());
	}
	for (int i = 0; i < prefetch_tensors->size(); i++) {
		printf("after preallocate prefetch_tensors%d-layer%d-block[%d]\n", (*prefetch_tensors)[i]->get_tensor_id(), (*prefetch_tensors)[i]->get_layer_id(), (*prefetch_tensors)[i]->get_prefetch_block_id());
	}
	mem_controller->printSWAPBLOCK("lalalala");
#endif	
	return true;
}
// #undef DEBUG

template <class value_type>
void ATPSearch<value_type>::test_code() {
	// size_t code1_size = 1886;
	// size_t code2_size = 1666;
	int code1[2056] = {0,1,1,1,0,0,0,1,0,0,0,0,0,0,2,0,0,0,2,2,2,0,0,2,0,1,2,1,0,0,2,2,2,0,1,2,0,0,2,0,0,0,2,2,2,0,1,0,0,0,0,0,0,0,0,0,2,2,2,1,0,0,0,1,0,0,0,0,2,2,2,0,0,1,0,0,1,0,0,0,2,2,2,1,0,0,1,0,0,0,0,0,2,2,2,0,1,0,0,0,1,0,0,0,2,2,2,0,0,0,1,0,0,0,0,0,2,2,2,0,0,0,0,0,0,0,0,0,2,2,2,0,0,0,0,0,0,0,0,0,2,2,2,0,1,0,0,1,0,0,0,0,0,0,2,2,2,0,1,0,0,1,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,1,2,2,2,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,1,0,0,1,0,0,0,0,0,1,0,0,0,0,0,0,1,2,2,2,0,0,1,1,0,0,1,0,0,0,0,0,1,0,0,0,0,0,0,1,0,2,2,2,0,0,0,0,0,1,0,0,1,0,0,0,0,0,1,0,0,1,0,0,0,2,2,2,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,1,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,1,1,0,0,1,0,0,0,2,2,2,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,1,0,0,0,0,2,2,2,0,0,0,0,0,1,0,0,0,1,1,1,1,0,1,0,0,0,1,0,0,0,0,0,0,0,0,1,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,1,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,1,0,0,0,0,0,1,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,1,1,1,0,0,0,1,1,0,1,0,0,0,0,0,0,0,0,0,1,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,1,0,0,0,0,0,0,0,2,2,2,0,0,0,1,0,0,0,0,0,2,2,2,0,0,0,0,0,0,0,0,1,2,2,2,0,0,1,0,0,0,0,0,0,1,1,1,0,0,0,1,0,0,0,0,0,2,2,2,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,2,2,2,1,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,1,0,0,1,0,2,2,2,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,1,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,0,0,0,0,0,0,0,0,0,1,1,1,0,0,0,0,0,0,0,0,0,2,2,2,0,0,0,0,1,0,0,0,0,0,0,0,0,1,0,0,0,1,0,0,1,2,2,2,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,2,2,2,0,0,0,0,0,0,0,0,0,1,1,1,0,0,0,0,0,0,1,0,0,2,2,2,0,1,0,0,0,0,0,0,1,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,2,2,2,1,0,0,0,0,0,0,0,1,0,0,0,0,1,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,1,0,0,0,1,0,0,0,0,1,1,0,0,1,0,0,0,1,0,0,0,0,1,0,0,1,1,0,0,0,2,2,2,0,0,0,1,0,0,1,0,0,0,0,0,1,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,2,2,2,0,1,1,0,0,1,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,1,0,0,1,0,0,0,1,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,2,2,2,0,0,0,1,0,0,1,1,0,2,2,2,0,0,0,0,1,1,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,1,0,0,2,2,2,1,0,0,0,1,0,0,1,1,0,0,0,0,1,0,0,1,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,2,2,2,0,0,0,0,1,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,1,0,0,0,0,1,0,0,0,0,0,0,0,1,1,1,0,0,1,0,0,0,1,0,0,2,2,2,1,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,1,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0,1,0,0,0,0,0,0,0,0,1,0,0,1,0,0,0,0,0,1,0,0,1,1,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,1,0,0,1,0,0,0,0,0,0,0,0,1,0,0,0,1,0,0,0,0,0,0,0,0,1,0,0,1,1,0,0,0,0,0,0,0,1,1,0,0,0,0,0,1,1,1,1,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,1,0,0,1,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,1,1,1,0,0,1,0,0,0,0,0,0,0,0,0,1,0,0,1,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,1,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,1,0,0,1,0,0,0,0,0,2,2,2,0,0,0,0,0,1,0,1,0,1,1,1,0,0,1,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,1,0,0,0,0,0,0,0,0,1,0,0,0,0,0,2,2,2,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,1,0,0,1,0,0,0,1,2,2,2,1,0,1,0,0,0,1,0,0,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,1,1,0,0,1,0,0,0,0,1,1,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,1,0,0,1,0,0,1,2,2,2,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,1,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,1,1,1,0,0,0,1,0,0,1,0,0,0,0,0,1,0,0,1,0,0,0,1,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,1,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,1,0,0,1,0,0,0,2,2,2,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,1,0,2,2,2,0,1,0,0,0,1,0,0,1,0,0,0,0,0,0,0,0,1,0,0,0,2,2,2,0,0,0,1,0,0,0,1,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,1,1,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,1,1,1,1,0};
	bool code2[1666] = {1,1,1,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,1,0,0,0,0,0,0,1,0,0,1,0,0,0,0,0,0,1,0,0,0,1,0,0,1,0,0,0,1,0,0,1,0,0,0,1,0,0,1,0,0,0,0,0,0,0,1,0,0,1,0,0,0,0,0,0,0,0,0,0,1,0,0,1,0,0,0,1,0,0,1,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,1,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,1,0,0,0,0,0,0,1,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,1,0,0,0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,1,0,0,0,0,0,0,1,0,0,1,0,0,0,1,0,0,1,0,0,0,1,0,0,0,0,0,0,1,0,0,0,0,0,0,1,0,0,1,0,0,0,1,0,0,1,0,0,0,1,0,0,0,0,0,0,1,0,0,0,1,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,1,0,0,0,1,0,0,0,1,0,0,1,0,0,0,0,0,0,0,0,0,0,1,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,1,0,0,1,0,0,0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,1,0,0,0,1,0,0,1,0,0,0,1,0,0,1,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,1,0,0,1,0,0,0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,1,0,0,1,0,0,0,0,0,0,0,0,0,0,1,0,0,1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,1,0,0,0,0,0,0,1,0,0,1,0,0,0,1,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,1,0,0,0,0,0,0,1,0,0,1,0,0,0,1,0,0,1,0,0,0,0,1,0,1,1,0,0,0,0,0,0,1,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,1,0,0,0,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,1,0,0,0,1,0,0,0,0,0,0,1,0,0,1,0,1,0,0,0,0,0,0,0,0,1,0,0,1,0,0,0,1,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,1,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,1,0,0,0,1,0,0,1,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,1,0,0,0,1,0,0,0,0,0,0,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,0,0,0,1,0,0,0,0,0,0,1,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,1,0,0,0,1,0,0,1,1,0,0,0,0,0,0,1,0,0,0,1,1,0,0,1,0,0,1,0,0,0,0,0,0,0,0,0,0,1,0,0,1,1,0,0,1,0,0,0,1,0,0,0,0,0,0,1,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,1,0,0,0,0,1,0,0,0,0,0,0,1,0,0,0,1,0,0,0,0,0,1,1,0,1,0,0,0,0,1,0,0,1,0,0,0,1,0,0,1,0,0,0,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,0,0,1,1,0,0,1,0,0,1,0,0,0,0,0,0,1,0,0,0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,1,0,0,0,1,0,0,1,0,0,0,1,0,0,1,0,0,0,0,0,0,1,0,0,0,1,0,0,1,0,0,0,0,0,0,1,1,0,0,0,0,0,1,1,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,1,0,0,1,0,0,0,1,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,1,1,0,0,0,1,0,0,1,0,0,0,1,0,0,1,0,0,0,1,0,0,0,1,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,1,0,0,0,0,0,0,1,0,0,0,0,0,0,1,0,0,0,0,0,0,1,0,0,0,1,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,1,0,0,0,0,0,0,1,0,0,0,1,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,1,0,0,1,0,0,0,1,0,0,1,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,0,1,0,0,1,1,0,0,0,0,0,0,0,0,0,1,0,0,0,1,0,0,0,0,0,0,1,0,0,0,0,0,0,1,0,0,1,0,0,0,0,0,0,1,0,0,0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,1,0,0,0,0,0,0,1,0,0,0,0,0,0,1,0,0,0,1,0,0,1,0,0,0,0,0,0,1,0,0,0,1,0,0,0,0,0,0,1,0,0,1,0,0,0,1,0,0,1,0,0,0,1,0,0,1,0,0,0,0,0,0,1,1,0,0,0,0,0,0,1,0,0,1,0,0,0,0,0,0,1,0,0,0,1,0,0,0,0,0,0,1,0,0,1,0,0,0,0,0,0,1,0,0,0,1,0,0,1,0,0,0,1,0,0,1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,1,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,1,0,0,0,0,0,0,1,0,0,1,0,0,0,1,0,0,1,0,0,0,0,0,0,1,0,0,0,1,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,1,0,0,1,0,0,0,1,0,0,1,0,0,0,1,0,0,0,0,0,0,0,0,0,1,0,0,0,1,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,1,0,0,1,0,0,0,1,0,0,1,0,0,0,1,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,1,0,0,0,1,0,0,1,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,1,0,0,1,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1};
	std::map<int, void* > net_layers  = reg->get_net_layers();
	for (auto it = net_layers.begin(); it != net_layers.end(); it++) {
		base_layer_t<value_type>* layer = (base_layer_t<value_type>*) it->second;
		if (layer->get_layer_structure() == MIMO) {
			std::vector<tensor_t<value_type>*> tensors = ((base_structure_t<value_type>*)layer)->get_outputs();
			for (int j = 0; j < tensors.size(); j++) {
				tensors[j]->set_position(REMAIN_IN_GPU);
			}	
		}
		else {
			tensor_t<value_type>* tensor = ((base_network_layer_t<value_type>*)layer)->get_f_out();
			tensor->set_position(REMAIN_IN_GPU);
		}
	}
	net->SetRecomputeSwapTensorsbyRoute(code1, 2056);
	double iter_time = 0.0;
	double sync_time = 0.0;
	double offload_size_error = 0;
	bool flag = 0;
	flag = GetIterationTimeGASwappingCode(code2, &iter_time, &sync_time, &offload_size_error, true);
	printf("flag = %d\n", flag);
}

template <class value_type>
bool ATPSearch<value_type>::GetIterationTimeGASwappingCode(bool* swapping_code, double* iter_time, double* sync_time, double* swapping_size_error, bool display) {
	double sync_time_f = 0.0;
	double sync_time_b = 0.0;
	double tf = 0.0;
	double tb = 0.0;
	double offload_time = 0.0;
	double fetch_time = 0.0;
	bool pass_flag;
	size_t swapping_size;
	size_t swapping_size_error_int;
	std::vector<tensor_t<value_type>*> swap_tensors;
	std::vector<tensor_t<value_type>*> prefetch_tensors;
	swapping_size = SetSwappingTensors(swapping_code, &(this->alternative_swap_tensors), &swap_tensors, &prefetch_tensors);
	
	// printf("SetSwappingTensors done, swapping_size = %zd=%f, %d-swap_tensors, %d-prefetch_tensors\n", swapping_size, BYTE_TO_MB(swapping_size), swap_tensors.size(), prefetch_tensors.size());
	if (this->min_swapping_size > swapping_size) {
		if (display) printf("SetSwappingTensors fail, swapping_size = %zd=%f < MS = %zd=%f\n", swapping_size, BYTE_TO_MB(swapping_size), min_swapping_size, BYTE_TO_MB(min_swapping_size));
		return false;
	} 	
	else {
		swapping_size_error_int = swapping_size - this->min_swapping_size;
		*swapping_size_error = (double)swapping_size_error_int / (double)min_swapping_size; 
		if (display) printf("SetSwappingTensors success, swapping_size = %f, *swapping_size_error = %lf\n", BYTE_TO_MB(swapping_size), *swapping_size_error);
	}															
	pass_flag = IsSwappingLegal(&swap_tensors, &prefetch_tensors);
	if (display) printf("pass_flag = %s\n", pass_flag ? "true" : "false");
	if (pass_flag == false) {
		printf("Swapping Tensors is illegal %d\n", 555);
		return false;
	}
	pass_flag = IterationTimeEvaluator(&this->recompute_tensors, &swap_tensors, &tf, &sync_time_f, &offload_time, &prefetch_tensors, &tb, &sync_time_b, &fetch_time);
	if (display) {
		printf("IterationTimeEvaluator done, pass_flag = %d:\n iter = %lf, tf = %lf, tb = %lf, sf = %lf, sb = %lf, offload = %lf, fetch = %lf, swapping_size = %fMB, min_size = %fMB, size_error = %lf\n", 
		pass_flag, tf + tb, tf, tb, sync_time_f, sync_time_b, offload_time, fetch_time, BYTE_TO_MB(swapping_size), BYTE_TO_MB(min_swapping_size), *swapping_size_error);
	}
	
	if (pass_flag) {
		*iter_time = tf + tb;
		*sync_time = sync_time_f + sync_time_b;
		return true;
	}
	else {
		return false;
	}
}

template <class value_type>
bool ATPSearch<value_type>::GetIterationTimePSOCode012(int* pso012_code, double* iter_time, double* sync_time, double* swapping_size_error, bool display) {
	double sync_time_f = 0.0;
	double sync_time_b = 0.0;
	double tf = 0.0;
	double tb = 0.0;
	double offload_time = 0.0;
	double fetch_time = 0.0;
	bool pass_flag;
	size_t swapping_size;
	size_t swapping_size_error_int;
	size_t excessive_size = excessive_size_by_batchsize[this->batch_size];
	std::vector<tensor_t<value_type>*> swap_tensors;
	std::vector<tensor_t<value_type>*> prefetch_tensors;
	swapping_size = SetTensorPolicy(pso012_code, &swap_tensors, &prefetch_tensors);
	if (excessive_size > swapping_size) {
		if (display) printf("SetTensorPolicy fail, swapping_size = %zd=%f < excessive_size = %zd=%f\n", swapping_size, BYTE_TO_MB(swapping_size), excessive_size, BYTE_TO_MB(excessive_size));
		return false;
	} 	
	else {
		swapping_size_error_int = swapping_size - excessive_size;
		*swapping_size_error = (double)swapping_size_error_int / (double)excessive_size; 
		if (display) printf("SetTensorPolicy success, swapping_size = %f, *swapping_size_error = %lf\n", BYTE_TO_MB(excessive_size), *swapping_size_error);
	}															
	pass_flag = IsPSOcode012Legal(&swap_tensors, &prefetch_tensors);
	// printf("pass_flag = %s\n", pass_flag ? "true" : "false");
	pass_flag = IterationTimeEvaluator(&this->recompute_tensors, &swap_tensors, &tf, &sync_time_f, &offload_time, &prefetch_tensors, &tb, &sync_time_b, &fetch_time);
	if (display) {
		printf("IterationTimeEvaluator done, pass_flag = %d:\n iter = %lf, tf = %lf, tb = %lf, sf = %lf, sb = %lf, offload = %lf, fetch = %lf, swapping_size = %fMB, min_size = %fMB, size_error = %lf\n", 
		pass_flag, tf + tb, tf, tb, sync_time_f, sync_time_b, offload_time, fetch_time, BYTE_TO_MB(swapping_size), BYTE_TO_MB(excessive_size), *swapping_size_error);
	}
	if (pass_flag) {
		*iter_time = tf + tb;
		*sync_time = sync_time_f + sync_time_b;
		return true;
	}
	else {
		return false;
	}
}

template <class value_type>
size_t ATPSearch<value_type>::SetTensorPolicy(int* pso012_code, std::vector<tensor_t<value_type>*>* swap_tensors, std::vector<tensor_t<value_type>*>* prefetch_tensors) {
	std::map<int, void* >* net_layers  = reg->get_net_layers_ptr();
	std::vector<std::pair<int, net_comp> >* net_comp_route = reg->get_net_comp_route_ptr();
	base_layer_t<value_type>* layer;
	base_structure_t<value_type>* structure_layer;
	base_network_layer_t<value_type>* net_layer;
	
	size_t swapping_size = 0;
	int swap_layer_num = 0;												
	int k = 0;
	for (size_t i = 0; i < this->alternative_tensors.size(); i++) {
		tensor_t<value_type>* tensor = this->alternative_tensors[i];
		tensor->reset_all_state();
		if (pso012_code[k] == 1) {
			tensor->set_position(SHARED_GPU_POOL);
			swapping_size += tensor->get_mem_size();
		}
		else if (pso012_code[k] == 2) {
			base_layer_t<value_type>* layer = (base_layer_t<value_type>*) net_layers->find(tensor->get_layer_id())->second;
			if (layer->get_layer_structure() == MIMO) {  // MIMO output can not be recomputed
				tensor->set_position(SHARED_GPU_POOL);
				pso012_code[k] = 1;
			}
			else {
				tensor->set_position(RECOMPUTE_IN_BACKWARD);
			}
			swapping_size += tensor->get_mem_size();
		}
		else {
			tensor->set_position(REMAIN_IN_GPU);
		}
		k++;
	}
	swap_tensors->clear();
	prefetch_tensors->clear();
	tensor_t<value_type>* tensor; 
	std::vector<tensor_t<value_type>*> tensors; 
	for (size_t i = 0; i < net_comp_route->size(); i++) {  // reset tensor state
		if ((*net_comp_route)[i].second == FORWARD) {
			size_t layer_id = (*net_comp_route)[i].first;
			base_layer_t<value_type>* layer = (base_layer_t<value_type>*) net_layers->find(layer_id)->second;
			if (layer->get_layer_structure() == MIMO) {
				structure_layer = (base_structure_t<value_type>* )layer;
				tensors = (structure_layer->get_outputs());
				for (int j = 0; j < tensors.size(); j++) {
					if (tensors[j]->get_position() == SHARED_GPU_POOL) {
						if (tensors[j]->get_backward_useful() == true) {
							auto iter = std::find(swap_tensors->begin(), swap_tensors->end(), tensors[j]);
							if (iter == swap_tensors->end()) {  // Prevent duplicate addition
								swap_tensors->push_back(tensors[j]);
							}
						}
					}
				}					
			}
			else {
				if (layer->get_layer_type() == RNN || layer->get_layer_type() == SATTN) {
					// tensor = ((rnn_layer_t<value_type>*)layer)->get_rnn_reserve_buff();
					tensor = ((base_network_layer_t<value_type>*)layer)->get_reserve_buff();
					if (tensor->get_position() == SHARED_GPU_POOL) {
						swap_tensors->push_back(tensor);
					}
				}
				net_layer = (base_network_layer_t<value_type>* )layer;
				tensor = net_layer->get_f_out();	
				if (tensor->get_position() == SHARED_GPU_POOL) {
					swap_tensors->push_back(tensor);
				}
			}
		}
		else {  // (*net_comp_route)[(*swap_layers)[i]].second == BACKWARD
			size_t layer_id = (*net_comp_route)[i].first;
            base_layer_t<value_type>* layer = (base_layer_t<value_type>*) net_layers->find(layer_id)->second;
            if (layer->get_layer_structure() == MIMO) {
                std::vector<tensor_t<value_type>*> t_ins = ((base_structure_t<value_type>*)layer)->get_inputs();
                for (int j = t_ins.size()-1; j >= 0; j--) {
                    if (t_ins[j]->get_position() == SHARED_GPU_POOL) {
						if (t_ins[j]->get_use_counter(BACKWARD) > true) {
							auto iter = std::find(prefetch_tensors->begin(), prefetch_tensors->end(), t_ins[j]);
							if (iter == prefetch_tensors->end()) {  // Prevent duplicate addition
								prefetch_tensors->push_back(t_ins[j]);
							}
						}
                    }
                }
            }
            else {  // SISO Layer
                if (layer->get_layer_type() != DATA_L) {
					if (layer->get_layer_type() == RNN || layer->get_layer_type() == SATTN) {
						// tensor_t<value_type>* reserve_buff = ((rnn_layer_t<value_type>*)layer)->get_rnn_reserve_buff();
						tensor_t<value_type>* reserve_buff = ((base_network_layer_t<value_type>*)layer)->get_reserve_buff();
						if (reserve_buff->get_position() == SHARED_GPU_POOL) {
                        	prefetch_tensors->push_back(reserve_buff);
                    	}
					}
                    tensor_t<value_type>* t_in = ((base_network_layer_t<value_type>*)layer)->get_f_in();
                    if (t_in->get_position() == SHARED_GPU_POOL) {
						auto iter = std::find(prefetch_tensors->begin(), prefetch_tensors->end(), t_in);
						if (iter == prefetch_tensors->end()) {  // Prevent duplicate addition
							prefetch_tensors->push_back(t_in);
						}
                    }
                }
            }
		}
	}
	
	// printf("\n");
	// printf("set swap_tensors and prefetch_tensors done, swap_tensors->size = %d, prefetch_tensors->size = %d\n", swap_tensors->size(), prefetch_tensors->size());
	return swapping_size;
}

// #define DEBUG
template <class value_type>
bool ATPSearch<value_type>::IsPSOcode012Legal(std::vector<tensor_t<value_type>*>* swap_tensors, std::vector<tensor_t<value_type>*>* prefetch_tensors) {
#ifdef DEBUG
	for (int i = 0; i < swap_tensors->size(); i++) {
		printf("before preallocate swap_tensors%d-layer%d-block[%d] %d/%d\n", (*swap_tensors)[i]->get_tensor_id(), (*swap_tensors)[i]->get_layer_id(), (*swap_tensors)[i]->get_swap_block_id(), i, swap_tensors->size());
	}
#endif
	if (!mem_controller->PreAllocateRecompteSwapBlock(&this->swap_tensors, &this->prefetch_tensors)) {
		return false;
	}
#ifdef DEBUG
	for (int i = 0; i < swap_tensors->size(); i++) {
		printf("after preallocate swap_tensors%d-layer%d-block[%d]\n", (*swap_tensors)[i]->get_tensor_id(), (*swap_tensors)[i]->get_layer_id(), (*swap_tensors)[i]->get_swap_block_id());
	}
	for (int i = 0; i < prefetch_tensors->size(); i++) {
		printf("after preallocate prefetch_tensors%d-layer%d-block[%d]\n", (*prefetch_tensors)[i]->get_tensor_id(), (*prefetch_tensors)[i]->get_layer_id(), (*prefetch_tensors)[i]->get_prefetch_block_id());
	}
	mem_controller->printSWAPBLOCK("lalalala");
#endif	
	return true;
}
// #undef DEBUG

template <class value_type>
void ATPSearch<value_type>::set_simulator_trainning(
		preprocessor<value_type>* processor, parallel_reader_t<value_type> *reader, network_t<value_type> *net, 
		size_t batch_size, base_layer_t<value_type> *first_layer, base_layer_t<value_type> *last_layer,
		const char *train_image_bin, const char *train_label_bin, const char *train_mean_file) {
	
	// size_t batch_size = shape[0]; 
	// size_t C = shape[1];
	// size_t H = shape[2];
	// size_t W = shape[3]; 
	
	this->net->init_network_trainning(
		batch_size, 0, 
		processor, reader,
		first_layer, last_layer, 
		train_image_bin, train_label_bin, train_mean_file);
	printf("after init_network_trainning%d\n", 666);	
	this->batch_size = batch_size;

	this->mem_controller = (this->net)->get_mem_controller();
	// this->liveness = (this->mem_controller)->get_liveness_analysis_t();
	this->reg = net->get_registry();
	// this->total_men_by_batchsize[batch_size] = liveness->get_total_size();

	size_t net_size;
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
	size_t max_fragment_size;
	size_t max_tensor_size;	
	size_t max_workspace_size;
	size_t max_layer_size;
	printf("before get_size_info%d\n", 666);	
	reg->get_size_info(&net_size, &gpu_tensors_size, &gpu_pool_size, &swap_tensors_size, &swap_pool_size, &recompute_tensor_size, &recompute_pool_size, 
		&max_fragment_size, &max_grad_size, &max_tensor_size, &max_layer_size, &max_data_size, &b_data_pool_size, &QKV_buffer_size, &max_workspace_size);
	this->pool_size = swap_pool_size;
	// this->total_men_by_batchsize[batch_size] = net_size + this->inherent_size + swap_pool_size + recompute_pool_size;
	this->total_men_by_batchsize[batch_size] = net_size + this->inherent_size;
	this->set_excessive_size_by_batchsize(batch_size);

	printf("\nwhen batchsize = %d\nnet total_mem = %f, inheren size = %f\n", batch_size, BYTE_TO_MB(total_men_by_batchsize[batch_size]), BYTE_TO_MB(this->inherent_size));
	printf("net_total_men_size without swapping pool or recomputing pool = %f\n", BYTE_TO_MB(net_size + this->inherent_size));
	printf("gpu_tensors_size = %f\n", BYTE_TO_MB(gpu_tensors_size));
	printf("gpu_pool_size = %f\n", BYTE_TO_MB(gpu_pool_size));
	printf("excessive_size = %f\n", BYTE_TO_MB(this->excessive_size_by_batchsize[batch_size]));
	printf("recompte_pool_size = %f\n", BYTE_TO_MB(recompute_pool_size));
	printf("swap_pool_size = %f\n", BYTE_TO_MB(swap_pool_size));
	printf("max_grad_size = %f\n", BYTE_TO_MB(max_grad_size));
	printf("max_workspace_size = %f\n", BYTE_TO_MB(max_workspace_size));
	//printf("when batchsize = %d, idea net total_mem = %f\n", batch_size, BYTE_TO_MB(reg->get_total_size_pool_malloc_mode()+reg->get_inherent_size()));
	// tensor_t<value_type>* temp_tensor  = new tensor_t<value_type>( 1, 1, 1, 1, reg->get_vector(), DATA, 0);
	// printf("tensor counter = %d\n\n", temp_tensor->get_tensor_base_id());
	std::map<int, void* > net_layers  = reg->get_net_layers();
	for (auto it = net_layers.begin(); it != net_layers.end(); it++) {
		base_layer_t<value_type>* layer = (base_layer_t<value_type>*) it->second;
		if (layer->get_layer_structure() == SISO) {
			tensor_t<value_type>* t = ((base_network_layer_t<value_type>*)layer)->get_f_out();
			this->layers_output_size[layer->get_base_id()] = t->get_mem_size();
		}
		else { // layer->get_layer_structure() == MIMO
			tensor_t<value_type>* t = (((base_structure_t<value_type>*)layer)->get_outputs())[0];
			this->layers_output_size[layer->get_base_id()] = t->get_mem_size();
		}
	}
	mem_controller->init_swap_block();
	mem_controller->printSWAPBLOCK("set_simulator_trainning");
}

template <class value_type>
double ATPSearch<value_type>::GetThroughput(size_t batch_size) {
	return this->ideal_throughput_by_batchsize[batch_size];
}

// template <class value_type>
// bool ATPSearch<value_type>::UpdateThroughputPeak(size_t batch_size, int no_update_win) {
// 	if (this->ideal_throughput_by_batchsize[batch_size] > this->max_tp) {
// 		this->max_tp = this->ideal_throughput_by_batchsize[batch_size];
// 		this->best_batch_size = batch_size;
// 		// this->no_update_times = 0;
// 		return true;
// 	}
// 	else {
// 		return true;
// 		// this->no_update_times++;
// 	}
// 	if (this->no_update_times >= no_update_win) {
// 		return false;
// 	}
// 	else {
// 		return true;
// 	}
// }

template <class value_type>
bool ATPSearch<value_type>::simulate_trainning(network_t<value_type> *net, int batchsize, int iter) {
	assert(is_forward_setup == true);
    assert(is_testing_ready == true);
    assert(is_backward_setup == true);

    size_t curt_mem = query_used_mem();
    // printf("after setup the memory used:%f\n", BYTE_TO_MB(curt_mem));
	bool flag = false;
    value_type loss = 0;
    value_type running_std     = 0;
    value_type running_average = 0;
    value_type threshold       = 0;
    std::deque<value_type> loss_queue;
    double start = get_cur_time();
	double iter_start = 0;
	double end = 0;
	double aver_computing_time = 0;
	double throughput;
	
	layers_ft.clear();
	layers_bt.clear();

	this->swap_tensors.clear();
	this->prefetch_tensors.clear();
	this->alternative_swap_tensors.clear();
	this->alternative_tensors.clear();
	this->recompute_tensors.clear();

	start = get_cur_time();
	net->simulated_train(iter, &this->layers_ft, &this->layers_bt);
	end = get_cur_time();
	aver_computing_time = (end - start) / (double)iter;
	
	if (this->layers_ft.size() != this->layers_bt.size()) {
		printf("layer number is wrong, layers_ft.size = %d, layers_bt.size = %d\n", this->layers_ft.size(), this->layers_bt.size());
		exit(1);
	}
	
	double tf = 0; 
	double tb = 0;
	for (auto it = layers_ft.begin(); it != layers_ft.end(); it++) {
		tf += it->second;;  // layers_ft[i];
	}
	for (auto it = layers_bt.begin(); it != layers_bt.end(); it++) {
		tb += it->second;
	}
	this->forward_computing_time_by_batchsize[batchsize] = tf;
	this->backward_computing_time_by_batchsize[batchsize] = tb;
	// throughput = (double)batchsize / this->ideal_iter_time_by_batchsize[batchsize];
	size_t MR, MS;  // total swapping size, total 	
	// std::vector<std::pair<tensor_t<value_type>*, MIN_GENERATE*>> gt_list;	
	std::vector<std::pair<tensor_t<value_type>*, GENERATE_EFFICIENCY*>> gf_list;	
	ResetTensorPosition();	
	
#ifdef SWAP_ONLY											
	flag = DistributeMSSwapOnly(batchsize, &gf_list, &MR, &MS);
#else
	flag = DistributeMRMS_v2(batchsize, &gf_list, &MR, &MS);
#endif
	
	// while(1);
	double ideal_throughput;
	double iter_time, recomp_time, ideal_swap_time, f_time, b_time;
	printf("Start working out ThroughputUpperBound\n");
	// ThroughputUpperBound(batchsize, &MR, &MS, &gt_list, &(this->recompute_tensors), &(this->alternative_swap_tensors), &ideal_throughput, &iter_time, &recomp_time, &ideal_swap_time, &tf, &tb, &f_time, &b_time);
	ThroughputUpperBound_v2(batchsize, &MR, &MS, &gf_list, &(this->recompute_tensors), &(this->alternative_swap_tensors), &ideal_throughput, &iter_time, &recomp_time, &ideal_swap_time, &tf, &tb, &f_time, &b_time);
	ideal_recompute_size_by_batchsize[batchsize] = MR;
	ideal_offload_size_by_batchsize[batchsize] = MS;
	this->min_swapping_size = MS;
	ideal_offload_time_by_batchsize[batchsize] = (double)MS / pcie_bandwith;
	forward_computing_time_by_batchsize[batchsize] = tf;
	backward_computing_time_by_batchsize[batchsize] = tb;
	ideal_computing_time_by_batchsize[batchsize] = tf + tb;
	ideal_iter_time_by_batchsize[batchsize] = iter_time;
	iter_forward_time_by_batchsize[batchsize] = f_time;
	iter_backward_time_by_batchsize[batchsize] = b_time;
	ideal_recomputation_time_by_batchsize[batchsize] = recomp_time;
	ideal_swapping_time_by_batchsize[batchsize] = ideal_swap_time;
	ideal_throughput_by_batchsize[batchsize] = ideal_throughput;
	ideal_throughput_by_batchsize_without_swap[batchsize] = (double)batchsize / ideal_computing_time_by_batchsize[batchsize];
	// GetAlternativeSwappingTensors(&(this->recompute_tensors), &(this->alternative_swap_tensors));
	printf("batchsize = %d:\ntf = %lf, tb = %lf, t_comp = %lf\niter_time = %lf, f_time = %lf, b_time = %lf, recomp_time = %lf, ideal_swap_time = %lf, ideal_offload_time = %lf\nMR = %zd=%f, MS = %zd=%f\nThroughputUpperBound = %lf, throughput_without_recomputation_swapping = %lf\n\n", 
			batchsize, 
			forward_computing_time_by_batchsize[batchsize], backward_computing_time_by_batchsize[batchsize], ideal_computing_time_by_batchsize[batchsize], 
			ideal_iter_time_by_batchsize[batchsize], iter_forward_time_by_batchsize[batchsize], iter_backward_time_by_batchsize[batchsize], 
			ideal_recomputation_time_by_batchsize[batchsize], ideal_swapping_time_by_batchsize[batchsize], ideal_offload_time_by_batchsize[batchsize],
			ideal_recompute_size_by_batchsize[batchsize], BYTE_TO_MB(ideal_recompute_size_by_batchsize[batchsize]), ideal_offload_size_by_batchsize[batchsize], BYTE_TO_MB(ideal_offload_size_by_batchsize[batchsize]),
			ideal_throughput_by_batchsize[batchsize], ideal_throughput_by_batchsize_without_swap[batchsize]);
	if (flag == false) {
		ideal_throughput_by_batchsize[batchsize] = 0.0;
		printf("Excessive size is too large to train although swapping and recomputation are work.\n"); 
		// stream_singleton::destory_stream();
		// stream_singleton::get_compute_stream();
		return false;
	}
	// test_code();
	// while(1);

	// stream_singleton::destory_stream();
	// stream_singleton::get_compute_stream();
	// printSwapInfo("simulate_trainning");
	// while(1);
	return true;

	// while(1);
}

template <class value_type>
void ATPSearch<value_type>::get_swap_layers_from_ga_v2(bool* comp_route_swap, int size) {
	int len = 0;
	printf("swap_layers = ");
	for (size_t i = 0; i < size; i++) {
		if (comp_route_swap[i] == true) {
			printf("%d, ", i);
			len++;
		}
	}
	printf("total:%d\n", len);
}

template <class value_type>
void ATPSearch<value_type>::get_swap_layers_from_ga(bool* comp_route_swap, int size) {
	// int *swap_layers;
	int len = 0;
	for (size_t i = 0; i < size; i++) {
		if (comp_route_swap[i] == true) {
			len++;
		}
	}
	// swap_layers = new int(len);
	printf("swap_layers = ");
	std::vector<std::pair<int, net_comp> >* net_comp_route = reg->get_net_comp_route_ptr();
	int j = 0;
	for (size_t i = 0; i < net_comp_route->size(); i++) {
		if ((*net_comp_route)[i].second == FORWARD) {
			if (comp_route_swap[i] == true) {
				int layer_id = (*net_comp_route)[i].first;
				// swap_layers[j] = layer_id;
				j++;
				printf("%d, ", layer_id);
			}
		}
	}
	// printf("swap_layers = ");
	// for (int i = 0; i < len; i++) {
	// 	printf("%d, ", swap_layers[i]);
	// }
	printf("total:%d\n", len);
	// delete swap_layers;
	// return swap_layers;
}

//// VDNN
template <class value_type>
void ATPSearch<value_type>::GetVDNNSchemeCode() {
	std::map<int, void* > layers = reg->get_net_layers();
	std::vector<std::pair<int, net_comp> > net_route = reg->get_net_comp_route();
	printf("layers size = %d:\n", layers.size());
	size_t code_size = 0;
	for (size_t i = 0; i < layers.size(); i++) {  // set rs_code_route
		if (net_route[i].second == FORWARD) {
			base_layer_t<value_type>* layer = (base_layer_t<value_type>*) layers.find(net_route[i].first)->second;
			if (layer->get_layer_structure() == SISO) {
				if (layer->get_layer_type() == RNN || layer->get_layer_type() == SATTN) {
					// tensor_t<value_type>* reserve_buff = ((rnn_layer_t<value_type>*)layer)->get_rnn_reserve_buff();
					tensor_t<value_type>* reserve_buff = ((base_network_layer_t<value_type>*)layer)->get_reserve_buff();
					printf("%d,", 0);
					code_size++;
				}
				if (layer->get_layer_type() == CONV) {
					tensor_t<value_type>* t_out = ((base_network_layer_t<value_type>*)layer)->get_f_out();
					printf("%d,", 1);
				}
				else {
					printf("%d,", 0);
				}
				code_size++;
			}
			else {
				std::vector<tensor_t<value_type>*> t_outs = ((base_structure_t<value_type>*)layer)->get_outputs();
				for (int j = 0; j < t_outs.size(); j++) {
					if (t_outs[j]->get_position() == SHARED_GPU_POOL) {
						printf("%d,", 1);
					}
					else {
						printf("%d,", 0);
					}
					code_size++;
				}
			}
		}
	}
	printf(" CODE size = %d\n", code_size);
	size_t j = 0;
	printf("GetRecomputeSwapTensorGACode done %d\n", j);
}

template <class value_type>
void ATPSearch<value_type>::vdnn_conv(std::vector<int>* swap_layers, int* swap_num) {

	this->mem_controller = (this->net)->get_mem_controller();
	this->liveness = (this->mem_controller)->get_liveness_analysis_t();
	this->reg = net->get_registry();
	size_t offload_size = 0;
	printf("vdnn_conv select layer%d, type = %d\n", -1, -1);
	std::map<int, void* > net_layers  = reg->get_net_layers();
	std::vector<std::pair<int, net_comp> > net_comp_route = reg->get_net_comp_route();
	int layer_id = net_comp_route[1].first;  // 先把第一层放到swap_layer
	base_layer_t<value_type>* layer;
	layer = (base_layer_t<value_type>*) net_layers.find(layer_id)->second;
	this->swap_layers.push_back(layer_id);
	this->pool_size = (((base_network_layer_t<value_type>*)layer)->get_f_out())->get_mem_size();
	for (size_t i = 2; i < net_comp_route.size(); i++) {  	
		if (net_comp_route[i].second == FORWARD) {
			layer_id = net_comp_route[i].first;
			layer = (base_layer_t<value_type>*) net_layers.find(layer_id)->second;
			// if (can_layer_swap(layer) && layer->get_layer_type()==CONV) {
				printf("vdnn_conv select layer%d, type = %d\n", layer_id, layer->get_layer_type());
				this->swap_layers.push_back(layer_id);  // 加入pool列表
				offload_size += ((base_network_layer_t<value_type>*)layer)->get_f_out()->get_mem_size();
			// }
		}
	}
	this->swap_num = this->swap_layers.size();
	*swap_num = this->swap_num;
	for (int i = 0; i < this->swap_layers.size(); i++) {
		swap_layers->push_back((this->swap_layers)[i]);
	}
	printf("vDNN swap_tensor num = %d, offload size = %zd = %f\n", this->swap_num, offload_size, BYTE_TO_MB(offload_size));
}


//// MODNN
template <class value_type>
void ATPSearch<value_type>::modnn_batch_select(int start_batchsize, int end_batchsize) {
	double alpha = 0.15;
	size_t max_size = 0;
	size_t temp_size = 0;
	int batchsize = 0;
	for (size_t b = start_batchsize; b <= end_batchsize; b++) {

	}
	
}

template <class value_type>
size_t ATPSearch<value_type>::modnn_max_window_mem_by_batchsize(int batchsize) {
	double alpha = 0.15;
	size_t max_size = 0;
	size_t temp_size = 0;
	std::map<int, void* > net_layers  = reg->get_net_layers();
	std::vector<std::pair<int, net_comp> > net_comp_route = reg->get_net_comp_route();
	int layer_id;
	base_layer_t<value_type>* layer;
	std::vector<tensor_t<value_type>*> t_outs;
	tensor_t<value_type>* t_out;
	std::vector<tensor_t<value_type>*> t_ins;
	tensor_t<value_type>* t_in;
	std::vector<tensor_t<value_type>*> b_datas;
	tensor_t<value_type>* b_data;
	tensor_t<value_type>* t;
	size_t reuse_size;
	std::vector<tensor_t<value_type>* >* dep_tensors;

	int window_size = alpha * net_comp_route.size();
	int max_layer_id = net->get_max_layer_id();
	for (size_t i = 1; i < (net_comp_route.size() - window_size); i++) {  	
		temp_size = 0;
		for (size_t j = i; j < i + window_size; j++) {
			layer_id = net_comp_route[j].first;
			if (net_comp_route[j].second == FORWARD) {
				// printf("layer%d dep_tensors\n", layer_id);
				layer = (base_layer_t<value_type>*) net_layers.find(layer_id)->second;
				dep_tensors = reg->get_forward_dependency(layer_id);
				// printf("layer%d dep_tensors_size = %d \n", layer_id, dep_tensors->size());
				for (size_t k = 0; k < dep_tensors->size(); k++) {
					// printf("layer%d dep_tensors[%d] ", layer_id, k);
					t = dep_tensors->operator[](k);
					temp_size += t->get_mem_size();
					// printf("size = %d\n", t->get_mem_size());
				}
				// printf("layer%d dep_tensors_size = %d \n", layer_id, dep_tensors->size());
				reuse_size = 0;
				if (layer->get_layer_structure() == MIMO) {
					// printf("layer%d type = %d \n", layer_id, layer->get_layer_type());
					t_ins = ((base_structure_t<value_type>*)layer)->get_inputs();
					for (size_t k = 0; k < t_ins.size(); k++) {
						// printf("layer%d type = %d t_outs[%d] = %zd\n", layer_id, layer->get_layer_type(), k, (t_ins[k])->get_mem_size());
						reuse_size += (t_ins[k])->get_mem_size();
					}
				}
				else {
					// printf("layer%d type = %d \n", layer_id, layer->get_layer_type());
					t_in = ((base_network_layer_t<value_type>*)layer)->get_f_in();
					reuse_size += t_in->get_mem_size();
				}
				temp_size -= reuse_size;  
			}
			else {  // backward
				dep_tensors = reg->get_backward_dependency(layer_id);
				for (size_t k = 0; k < dep_tensors->size(); k++) {
					t = dep_tensors->operator[](k);
					temp_size += t->get_mem_size();
				}
				reuse_size = 0;
				if (layer->get_layer_structure() == MIMO) {
					b_datas = ((base_structure_t<value_type>*)layer)->get_b_data();
					for (size_t k = 0; k < t_ins.size(); k++) {
						// printf("layer%d type = %d t_outs[%d] = %zd\n", layer_id, layer->get_layer_type(), k, (t_outs[k])->get_mem_size());
						reuse_size += (b_datas[k])->get_mem_size();
					}
				}
				else {
					b_data = ((base_network_layer_t<value_type>*)layer)->get_b_data();
					reuse_size += b_data->get_mem_size();
				}
				temp_size -= reuse_size; 
			}
		}
		// while(1);
		if (max_size < temp_size) {
			max_size = temp_size;
		}
		// printf("modnn_max_window_mem_by_batchsize temp_size = %zd\n", temp_size);
	} 
	printf("MoDNN: sub-batchsize = %d, max_window_size = %zd = %f\n", batchsize, max_size, BYTE_TO_MB(max_size));
	return max_size;
}

template <class value_type>
size_t ATPSearch<value_type>::modnn_offload_size_by_batchsize(int batchsize) {
	size_t offload_size = 0;
	size_t net_size = this->total_men_by_batchsize[batchsize];
	size_t max_window_size = this->modnn_max_window_mem_by_batchsize(batchsize);
	printf("when batchsize = %d, modnn_offload_size_by_batchsize = %f\n", batchsize, BYTE_TO_MB(net_size - max_window_size));
	return net_size - max_window_size;
}

template <class value_type>
void ATPSearch<value_type>::printTimeList() {
	std::map<int, void* >* net_layers  = reg->get_net_layers_ptr();
	std::vector<std::pair<int, net_comp> >* net_comp_route = reg->get_net_comp_route_ptr();
	int layer_id;
	printf("******swap printTimeList*****net_comp_route->size = %d\n", net_comp_route->size());
	for (int i = 0; i < net_comp_route->size(); i++) {
		layer_id = (*net_comp_route)[i].first;
		if ((*net_comp_route)[i].second == FORWARD) {
			printf("layers_ft[%d] = %f\n", layer_id, layers_ft[layer_id]);
		}
		else {
			printf("layers_bt[%d] = %f\n", layer_id, layers_bt[layer_id]);
		}
	}
}

template <class value_type>
void ATPSearch<value_type>::printSwapInfo(const char* str) {
	std::map<int, void* > net_layers  = reg->get_net_layers();
	std::vector<std::pair<int, net_comp> > net_comp_route = reg->get_net_comp_route();
	tensor_t<value_type>* tensor;
	std::vector<tensor_t<value_type>* > tensors;
	printf("\n******%s******\n", str);
	for (int i = 0; i < net_comp_route.size(); i++) {
		int layer_id = net_comp_route[i].first;
		base_layer_t<value_type>* layer = (base_layer_t<value_type>*) net_layers.find(layer_id)->second;
		if (net_comp_route[i].second == FORWARD) {
			if (layer->get_layer_type() == SOFTMAX) continue;
			if (layer->get_layer_structure() == SISO) {
				tensor = ((base_network_layer_t<value_type>*)layer)->get_f_out();
				printf("layers_ft[%d]=%f, tensor%d-mem=%zd-offload_time=%f\n", 
					layer_id, layers_ft[layer_id], tensor->get_tensor_id(), tensor->get_mem_size(), tensor->get_swap_time(OFFLOAD));
			}
			else {
				tensors = ((base_structure_t<value_type>*)layer)->get_outputs();
				printf("layers_ft[%d]=%f, tensor%d-mem=%zd-offload_time=%f\n", 
					layer_id, layers_ft[layer_id], tensors[0]->get_tensor_id(), tensors[0]->get_mem_size(), tensors[0]->get_swap_time(OFFLOAD));
			}
		}
		else {
			if (layer->get_layer_type() == DATA_L) continue;
			if (layer->get_layer_structure() == SISO) {
				tensor = ((base_network_layer_t<value_type>*)layer)->get_f_in();
				printf("layers_bt[%d]=%f, tensor%d-mem=%zd-fetch_time=%f\n", 
					layer_id, layers_bt[layer_id], tensor->get_tensor_id(), tensor->get_mem_size(), tensor->get_swap_time(FETCH));
			}
			else {
				tensors = ((base_structure_t<value_type>*)layer)->get_inputs();
				printf("layers_bt[%d]=%f, tensor%d-mem=%zd-fetch_time=%f\n", 
					layer_id, layers_bt[layer_id], tensors[0]->get_tensor_id(), tensors[0]->get_mem_size(), tensors[0]->get_swap_time(FETCH));
			}
		}
	}
	printf("\n************\n");
}

INSTANTIATE_CLASS(ATPSearch);
    
} //ATP namespace
