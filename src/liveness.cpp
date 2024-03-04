//
// Created by ay27 on 9/14/17.
//

#include <liveness.h>
#include <util/mem_util.h>

namespace ATP
{

template <class value_type>
std::vector<std::pair<int, net_comp>> &
liveness_analysis_t<value_type>::get_subsequent_layers(int curt_layer_id, net_comp dir)
{
    if (dir == FORWARD)
    {
        return subsequent_forward->operator[]((size_t)curt_layer_id);
    }
    else
    {
        return subsequent_backward->operator[]((size_t)curt_layer_id);
    }
}

// 检查tensor在层layer_id中是否被用到
template <class value_type>
bool liveness_analysis_t<value_type>::is_used_by_layer(int layer_id, net_comp dir, tensor_t<value_type> *t)
{
    std::vector<tensor_t<value_type> *> *tensors = NULL;
    // 取该层需要的tensor�?
    if (dir == FORWARD)
    {
        tensors = reg->get_forward_dependency(layer_id);
    }
    else if (dir == BACKWARD)
    {
        tensors = reg->get_backward_dependency(layer_id);
    }
    if (tensors == NULL)
        return false;
    // 逐个检�?
    for (size_t i = 0; i < tensors->size(); i++)
    {
        if (tensors->operator[](i) == t)
        {
            return true;
        }
    }
    return false;
}

template <class value_type>
bool liveness_analysis_t<value_type>::is_freeable_afterwards(int curt_layer_id, net_comp dir, tensor_t<value_type> *t)
{
    std::vector<std::pair<int, net_comp>> subsequent_layers = get_subsequent_layers(curt_layer_id, dir);
    // 遍历层curt_layer_id的子序列，看看t是否在子序列中被使用，如不在被使用则可以释放
    for (size_t i = 0; i < subsequent_layers.size(); i++)
    {
        std::pair<int, net_comp> layer = subsequent_layers[i];
        bool is_used = is_used_by_layer(layer.first, layer.second, t); // (层id，计算方向，当前tensor)
        if (is_used)
        {
            return false;
        }
    }
    return true;
}

// 参数（ins，前向或反向），只负责一个方?
template <class value_type>
void liveness_analysis_t<value_type>::set_ins(std::vector<std::vector<void *>> *ins, net_comp dir)
{
    ins->resize(max_layer_id + 1);

    auto all_layers = reg->get_net_layers(); // 获取net_layers
    auto nets = reg->get_net_comp_route();   // 获取计算顺序�?
    // 顺序记录前向过程中所有被用到的tensor（除了数据和CONV_BUFF），按层     的计算顺序记录ins[layer_id]={......}
    for (auto layer = nets.begin(); layer != nets.end(); ++layer)
    {
        if (dir != layer->second)
        {
            // 计算方向不同直接跳过这次循环
            continue;
        }

        int layer_id = layer->first;
        ins->operator[](layer_id).resize(0);

        // we don't care about the DATA layer，不管数据层
        auto tmp = all_layers.find(layer_id);
        if (tmp == all_layers.end() || ((base_layer_t<value_type> *)tmp->second)->get_layer_type() == DATA_L)
        {
            continue;
        }

        std::vector<tensor_t<value_type> *> *tensors = NULL;
        // 取得前向或反向传播中，该层需要的tensor，这些tensor放在一个vector�?
        if (dir == FORWARD)
        {
            tensors = reg->get_forward_dependency(layer_id);
        }
        else if (dir == BACKWARD)
        {
            tensors = reg->get_backward_dependency(layer_id);
        }
        if (tensors == NULL)
            return;
        for (size_t i = 0; i < tensors->size(); i++)
        {
            tensor_t<value_type> *t = tensors->operator[](i);
            if (t->get_type() != DATA && t->get_type() != CONV_BUFF)
            {
                // 如果既不是数据，也不是卷积层的tensor，跳过不记录
                continue;
            }
            auto r_it = regulated_tensors->find(t);
            // 顺序记录前向过程中所有被用到的tensor，按顺序记录，regulated_tensors�? 是所有DATA和CONV_BUFF，最后一个除�?            
			if (r_it != regulated_tensors->end()) {
				ins->operator[](layer_id).push_back((void *)t);
			}
        }
    }
}

template <class value_type>
void liveness_analysis_t<value_type>::set_outs(std::vector<std::vector<void *>> *outs, net_comp dir)
{
    outs->resize(max_layer_id + 1);

    // the free_list scan algorithm is quite simple and stupid, can we make it more elegant?

    auto all_layers = reg->get_net_layers();
    auto nets = reg->get_net_comp_route();
	/*
	for (auto layer = all_layers.begin(); layer != all_layers.end(); ++layer) {
		printf("layer_id:%d\n", layer->first);
	}
	for (auto it = nets.begin(); it != nets.end(); ++it) {
		printf("nets layer_id:%d\n", it->first);
	}
	*/
    for (auto layer = nets.begin(); layer != nets.end(); ++layer)
    {
        if (dir != layer->second)
        {
            continue;
        }
        // 初始化outs
        int layer_id = layer->first;
		// printf("layer_id:%d\n", layer->first);
        outs->operator[](layer_id).resize(0);

        // we don't care about DATA layer
        auto tmp = all_layers.find(layer_id);

        // 首层数据层和末尾层不处理
        if (tmp == all_layers.end() || ((base_layer_t<value_type> *)tmp->second)->get_layer_type() == DATA_L)
        {
            continue;
        }
		if (((base_layer_t<value_type> *)tmp->second)->get_layer_type() == SOFTMAX) {
			continue;
		}
        // 从层的regulated_tensors迭代
        for (auto it = regulated_tensors->begin(); it != regulated_tensors->end(); it++)
        {
            tensor_t<value_type> *t = (tensor_t<value_type> *)it->first;

            // ignore tensor in future，该tensor可能用在未来的层，但不考虑
            if (dir == FORWARD && t->get_layer_id() > layer_id)
            {
                continue;
            }
            if (dir == BACKWARD && t->get_layer_id() < layer_id)
            {
                continue;
            }

#ifdef RECOMPUTE_ON
            // 如果在前向阶段，取当前层对象，如果当前层的下一层不为空?
            if (dir == FORWARD)
            {
                base_layer_t<value_type> *l = (base_layer_t<value_type> *)(reg->get_net_layers().find(layer_id)->second);
                if (!l->get_next().empty())
                {
                    if (is_checkpoint(l) && (t == reg->get_reg_output(layer_id, l->get_next()[0]->get_base_id())))
                    {
                        // t是当前层的输出tensor的id，如果该层是checkpoint层，并且t是输出tensor而不是参数，
                        // 则跳过整个循环，将其作为checkpoint
                        continue;
                    }
                }
            }
#endif
            // layer_id的tenor t 能否释放？（t在是否在子序列中被使用）
            bool freeable = is_freeable_afterwards(layer_id, dir, t);

            if (freeable)
            {
                // 可以被释放的tensor记录在layer_id的outs列表
                outs->operator[](layer_id).push_back((void *)t);
				// printf("layer_id:%d\n", layer_id);
				// printf("layer_id:%d\n\n", t->get_layer_id());
            }
        }
    }

    // purify outs
    // 遍历net_comp_route计算序列
    for (auto layer = nets.begin(); layer != nets.end(); ++layer)
    {
        if (dir != layer->second)
        {
            continue;
        }

        int layer_id = layer->first; // 取layer_id

        // 遍历layer_id的outs列表（可释放tensors），去除重复�?
        for (auto it1 = outs->operator[](layer_id).begin(); it1 != outs->operator[](layer_id).end();)
        {

            bool should_delete = false;

            int start, end;
            if (dir == FORWARD)
            {
                start = 1;
                end = layer_id - 1; // 前向遍历�?~当前层上一�?
            }
            else
            {
                start = layer_id + 1;
                end = max_layer_id;
            }
            for (int l = start; l <= end; ++l)
            {
                // 遍历l层的outs列表
                for (auto it2 = outs->operator[](l).begin(); it2 != outs->operator[](l).end(); ++it2)
                {
                    if (*it1 == *it2)
                    {
                        // it1�?
                        should_delete = true;
                        break;
                    }
                }
                if (should_delete)
                {
                    break;
                }
            }

            if (should_delete)
            {
                it1 = outs->operator[](layer_id).erase(it1);
            }
            else
            {
                ++it1;
            }
        }
    }
}

template <class value_type>
size_t liveness_analysis_t<value_type>::get_total_size_v2() {
    // all tensor do cudamalloc according to layer sequence, except for data tensor
	std::vector<tensor_t<value_type>* >* all_tensors = reg->get_vector();
	size_t total_size = 0;
    size_t cuda_mem_block = 2097152;  // 2MB
    size_t block_num = 0;
    size_t tensor_num = 0;
    size_t residual_mem = 0;
    size_t temp_size;
	for (size_t i = 0; i < all_tensors->size(); i++) {
		// printf("(*all_tensors)[%d]->get_mem_size() = %d total_size = %zd\n", i, (*all_tensors)[i]->get_mem_size(), total_size);
        if ((*all_tensors)[i]->get_type() == DATA) {
            continue;  //  first cudamalloc unDATA tensors 
        }
        tensor_num++;
        temp_size = ((*all_tensors)[i])->get_mem_size();
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
    for (size_t i = 0; i < all_tensors->size(); i++) {
        // if ((((*all_tensors)[i])->get_type() != DATA) || ((*all_tensors)[i])->get_position() != REMAIN_IN_GPU) {
        if ( ((*all_tensors)[i])->get_type() != DATA ) {
            continue;   // second cudamalloc DATA tensors 
        }
        tensor_num++;
        temp_size = ((*all_tensors))[i]->get_mem_size();
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
    total_size = cuda_mem_block * block_num;
    printf("total_size = %f, block_num = %zd, tensor_num = %zd\n", BYTE_TO_MB(total_size), block_num, tensor_num);
	return total_size;
}

template <class value_type>
size_t liveness_analysis_t<value_type>::get_total_size() {
	std::vector<tensor_t<value_type>* >* all_tensors = reg->get_vector();
	size_t total_size = 0;
	for (size_t i = 0; i < all_tensors->size(); i++) {
		// printf("(*all_tensors)[%d]->get_mem_size() = %d total_size = %zd\n", i, (*all_tensors)[i]->get_mem_size(), total_size);
		total_size += (*all_tensors)[i]->get_mem_size();
	}
	return total_size;
}

template <class value_type>
void liveness_analysis_t<value_type>::find_bottleneck() {
    this->bottleneck_mem_size = 0;
	this->bottleneck_output_size = 0;
	this->bottleneck_data_size = 0;
    this->max_grad_size = 0;
    int bottleneck_layer_id = -1;
	int bottleneck_output_layer_id = -1;
	int temp_data_size;
	auto net_layers = reg->get_net_layers();
    std::vector<tensor_t<value_type>* >* all_tensors = reg->get_vector();
	int tt = 0;
    for (size_t i = 0; i < all_tensors->size(); i++) {
        if(((*all_tensors)[i])->get_type() == GRAD) {
            if (this->max_grad_size < ((*all_tensors)[i])->get_mem_size()) {
                this->max_grad_size = ((*all_tensors)[i])->get_mem_size();
            }
        }
    }
	for (auto it = net_layers.begin(); it != net_layers.end(); it++) {
		base_layer_t<value_type> *curt_l = (base_layer_t<value_type> *) it->second;
		if (curt_l->get_layer_type() == DATA_L) {
			continue;
		}
		auto ftensors = reg->get_forward_dependency(it->first);
		size_t tmp1 = 0;
		size_t tmp2 = 0;
		size_t temp_data_size1 = 0;
		size_t temp_data_size2 = 0;
		for (auto t = ftensors->begin(); t != ftensors->end(); ++t) {
			tmp1 += (*t)->get_mem_size();
			// if ((*t)->get_type() == DATA || (*t)->get_type() == B_DATA || (*t)->get_type() == CONV_BUFF || (*t)->get_type() == RNN_BUFF) {
            if ((*t)->get_type() == DATA || (*t)->get_type() == B_DATA || (*t)->get_type() == RNN_DATA 
                || (*t)->get_type() == RNN_B_DATA || (*t)->get_type() == RNN_RESERVE
                || (*t)->get_type() == CONV_BUFF || (*t)->get_type() == RNN_BUFF)
            {
				temp_data_size1 += (*t)->get_mem_size();
			}
			// printf("\nlayerid = %d, tmp1 += (*t)->get_mem_size();\n", it->first);			
		}
		f_mem_usage[it->first] = tmp1;
		if (bottleneck_data_size < temp_data_size1) {
			bottleneck_data_size = temp_data_size1;
		}
		std::vector<tensor_t<value_type>* >* btensors = reg->get_backward_dependency(it->first);
        // printf("\nlayer%d\n", it->first);
		for (auto t = btensors->begin(); t != btensors->end(); t++) {
			tmp2 += (*t)->get_mem_size();
            // printf("\ntensor%d\n", (*t)->get_tensor_id());
			if ((*t)->get_type() == DATA || (*t)->get_type() == B_DATA || (*t)->get_type() == RNN_DATA 
                || (*t)->get_type() == RNN_B_DATA || (*t)->get_type() == RNN_RESERVE
                || (*t)->get_type() == CONV_BUFF || (*t)->get_type() == RNN_BUFF) 
            {
				temp_data_size2 += (*t)->get_mem_size();
			}
		}
        // printf("\ntt = %d, bottleneck_mem_size = %d\n", bottleneck_mem_size, tt);
		b_mem_usage[it->first] = tmp2;
		if (bottleneck_data_size < temp_data_size2) {
			bottleneck_data_size = temp_data_size2;
		}
		// printf("\ntt = %d, bottleneck_mem_size = %d\n", bottleneck_mem_size, tt);
		if (tmp2 > bottleneck_mem_size) {
            bottleneck_mem_size = tmp2;
            bottleneck_layer_id = it->first;
        }
		tt++;
	}
	
	for (auto it = net_layers.begin(); it != net_layers.end(); it++) {
		base_layer_t<value_type> *curt_l = (base_layer_t<value_type> *) it->second;
		if (curt_l->get_layer_type() == DATA_L) {
			continue;
		}
		auto ftensors = reg->get_forward_dependency(it->first);
	}
	
	for (int layer_id = 1; layer_id < (int)f_stash_tensors.size(); ++layer_id) {
        // printf("layer : %d\n", layer_id);
		size_t tmp1o = 0;
        for (size_t i = 0; i < f_stash_tensors[layer_id].size(); ++i) {
            tensor_t<value_type>* t = (tensor_t<value_type>*)f_stash_tensors[layer_id][i];
			if ((t->get_layer_id() == layer_id)&&(t->get_type() == DATA)) {
				tmp1o += t->get_mem_size();
				// printf("tensor %p : layer %d, type %d, state %d\n", t, t->get_layer_id(), t->get_type(), t->get_state());
			}
        }
		f_ouput_usage[layer_id] = tmp1o;
		if (bottleneck_output_size < tmp1o) {
			bottleneck_output_size = tmp1o;
			bottleneck_output_layer_id = layer_id;
		}
    }
	
	for (int layer_id = 1; layer_id < (int)b_stash_tensors.size(); ++layer_id) {
        // printf("layer : %d\n", layer_id);
		size_t tmp2o = 0;
        for (size_t i = 0; i < b_stash_tensors[layer_id].size(); ++i) {
            tensor_t<value_type>* t = (tensor_t<value_type>*)b_stash_tensors[layer_id][i];
			if ((t->get_layer_id() == layer_id)&&(t->get_type() == DATA)) {
				tmp2o += t->get_mem_size();
				// printf("tensor %p : layer %d, type %d, state %d\n", t, t->get_layer_id(), t->get_type(), t->get_state());
			}
        }
		b_ouput_usage[layer_id] = tmp2o;
    }

	printf("bottleneck_layer_id: %d, fmem: %zu, bmem: %zu\n", 
		bottleneck_layer_id, f_mem_usage[bottleneck_layer_id], b_mem_usage[bottleneck_layer_id]);
	printf("output_bottleneck_layer_id: %d, fmem: %zu, bmem: %zu\n", 
		bottleneck_layer_id, f_ouput_usage[bottleneck_output_layer_id], f_ouput_usage[bottleneck_output_layer_id]);
#ifdef MEM_DEBUG
    printf("bottleneck_layer_id: %d, fmem: %zu, bmem: %zu\n", 
		bottleneck_layer_id, f_mem_usage[bottleneck_layer_id], b_mem_usage[bottleneck_layer_id]);
#endif
}

template <class value_type>
// 为layer_id所需的数据申请了空间，当GPU中没有空间或者有效数�?
void liveness_analysis_t<value_type>::stash(int layer_id, net_comp dir)
{
    //    std::vector<tensor_t<value_type>* >* tensors = NULL;
    //    if (dir == FORWARD) {
    //        tensors = reg->get_forward_dependency(layer_id);
    //    } else if(dir == BACKWARD) {
    //        tensors = reg->get_backward_dependency(layer_id);
    //    }
    //    if( tensors == NULL ) return;
    //    /*------------------------------------------*/
    //    //we get the tensors ready in the curt layers
    //    for( size_t i = 0; i < tensors->size(); i++ ) {
    //        tensor_t<value_type>* t = tensors->operator[](i);
    //        typename std::map<tensor_t<value_type>*, mem_mode>::iterator it = regulated_tensors.find(t);
    //        if (it != regulated_tensors.end()) {
    //            if(t->get_state() == VOID) {
    //                t->atomic_set_state( GPU );
    //                t->stash_gpu_space();
    //            }
    //        }
    //    }
    //    return;
    std::vector<std::vector<tensor_t<value_type> *>> *ins = NULL;
    // 取得ins列表
    if (dir == FORWARD)
    {
        ins = (std::vector<std::vector<tensor_t<value_type> *>> *)&f_stash_tensors;
    }
    else if (dir == BACKWARD)
    {
        ins = (std::vector<std::vector<tensor_t<value_type> *>> *)&b_stash_tensors;
    }

    // 遍历layer_id的ins
    for (auto it = ins->operator[](layer_id).begin(); it != ins->operator[](layer_id).end(); ++it)
    {
        tensor_t<value_type> *t = *it; // 取ins(layer_id)的一个tensor
#ifdef RECOMPUTE_ON
        // leave recompute tensor for recompute routine
        if (t->get_state() == RECOMPUTE)
        {
            // 如果是recompute的tensor，不做后续处�?            continue;
        }
#endif
        // 如果不是recompute的tensor
        if (t->get_type() == CONV_BUFF)
        {
            // 为CONV_BUFF申请空间
            t->stash_gpu_space();
        }
        else
        {
            t->CPUtoGPU();
        }
    }

#ifdef MEM_DEBUG
    printf("\n-------ins------\n");
    for (int i = 1; i < ins->size(); ++i)
    {
        printf("---layer %d\n", i);
        for (auto it = ins->operator[](i).begin(); it != ins->operator[](i).end(); ++it)
        {
            printf("%p ", (*it));
        }
        printf("\n");
    }
#endif
}

template <class value_type>
void liveness_analysis_t<value_type>::update(int layer_id, net_comp dir)
{
    //    // we only do free and offload to cpu here
    //    typename std::map<tensor_t<value_type>*, mem_mode>::iterator it = regulated_tensors.begin();
    //    for ( it = regulated_tensors.begin(); it != regulated_tensors.end(); it++ ) {
    //        // we update the table, and set tensors that no longer to be used to VOID to save memory
    //        // therefore, the operation is toward GPU tensors only
    //        tensor_t<value_type>* t = it->first;
    //        if( t->get_state() == VOID ) continue;
    //
    //        std::pair<bool, int> sta = is_freeable_afterwards(layer_id, dir, t);
    //
    //        if( t->get_state() == GPU ) {
    //            // to check if the tensor is freeable
    //            if(sta.first == true) {
    //                t->atomic_set_state( VOID );
    //#ifdef MEM_DEBUG
    //                printf("tensor:%p is ready to free\n", t);
    //#endif
    //                t->free_gpu_space();
    //            }
    //        }
    //    }
    //    return;

    std::vector<std::vector<tensor_t<value_type> *>> *outs;
    if (dir == FORWARD)
    {
        // 前向取f_free_tensors
        outs = (std::vector<std::vector<tensor_t<value_type> *>> *)&f_free_tensors;
    }
    else
    {
        // 反向取b_free_tensors
        outs = (std::vector<std::vector<tensor_t<value_type> *>> *)&b_free_tensors;
    }

    for (auto it = outs->operator[](layer_id).begin(); it != outs->operator[](layer_id).end(); ++it)
    {
        tensor_t<value_type> *t = *it;
        t->free_gpu_space(VOID);
    }
#ifdef MEM_DEBUG
    printf("\n-------outs------\n");
    for (int i = 1; i < outs->size(); ++i)
    {
        printf("---layer %d\n", i);
        for (auto it = outs->operator[](i).begin(); it != outs->operator[](i).end(); ++it)
        {
            printf("%p ", (*it));
        }
        printf("\n");
    }
#endif
}

INSTANTIATE_CLASS(liveness_analysis_t);

} // namespace ATP
