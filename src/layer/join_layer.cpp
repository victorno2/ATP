#include <layer/join_layer.h>

namespace ATP {
    
template <class value_type>
void join_layer_t<value_type>::forward_setup( registry_t<value_type>* reg, cudnnHandle_t* cudnn_h ) {
    // printf("======>setup the forward join layer@%d\n", this->get_id() );
    std::vector<std::pair<int, int> > input_keys = this->get_inputs_keys();
	this->clear_inputs();
    for (size_t i = 0; i < input_keys.size(); i++) {
        std::pair<int, int> input_key = input_keys[i];
        tensor_t<value_type>* input = reg->get_reg_output(input_key.first, input_key.second);
        assert(input != NULL);
        this->set_input( input );
		// input->increase_use_count_initial();
		// input->increase_use_count();
        // input->set_backward_useful(false);
        input->increase_use_counter(FORWARD);
    }
    // join layer only has 1 output
    // reduce all the inputs into input[0]

    // should not overlap tensor !!!
    tensor_t<value_type>* t = this->get_inputs()[0];
    tensor_t<value_type>* output = new tensor_t<value_type>(t->get_N(), t->get_C(), t->get_H(), t->get_W(), reg->get_vector(), DATA, this->get_id());
//    tensor_t<value_type>* output = (this->get_inputs())[0];
    std::pair<int, int> output_key = (this->get_outputs_keys())[0];
	this->clear_outputs();
    this->set_output( output, output_key, reg);
    // output->set_use_in_backward(false);
    // register the forward dependency
    std::vector<tensor_t<value_type>* > inputs = this->get_inputs();
    for (size_t i = 0; i < inputs.size(); i++) {
        reg->register_forward_dependency( this->get_id(), inputs[i] );
    }
    reg->register_forward_dependency(this->get_id(), output);
}

template <class value_type>
void join_layer_t<value_type>::backward_setup( registry_t<value_type>* reg, cudnnHandle_t* cudnn_h ) {
    //backward
    printf("======>setup the backward join layer@%d\n", this->get_id() );
    std::pair<int, int>    output_k  = (this->get_outputs_keys())[0];
    tensor_t<value_type>* dEdD_next = reg->get_reg_b_data(output_k.second, output_k.first);
    assert(dEdD_next != NULL);
    this->clear_dy();
    this->set_dy(dEdD_next);
    // std::vector<std::pair<int, int> > inputs  = this->get_inputs_keys();
    // this->clear_b_data();
    // this->set_b_data(dEdD_next, inputs[0], reg);
    // for(size_t i = 1; i < inputs.size(); i++) {
    //     tensor_t<value_type>* tmp = new tensor_t<value_type>( dEdD_next->get_N(), dEdD_next->get_C(), dEdD_next->get_H(), dEdD_next->get_W(), reg->get_vector(), B_DATA, this->get_id());
    //     this->set_b_data(tmp, inputs[i], reg);
    // }
    // while(1);
    // register the backward dependency
    // inputs  = this->get_inputs_keys();
    // tensor_t<value_type>* source = reg->get_reg_b_data(inputs[0].second, inputs[0].first);
    // source->increase_b_use_count();
    // for(size_t i = 1; i < inputs.size(); i++) {
    //     tensor_t<value_type>* target = reg->get_reg_b_data(inputs[i].second, inputs[i].first);
    //     reg->register_backward_dependency(this->get_id(), target );
    // }
    // reg->register_backward_dependency(this->get_id(), source );

    std::vector<std::pair<int, int> > inputs  = this->get_inputs_keys();

    // printf("inputs.size() = %d\n", inputs.size());
    this->clear_b_data();
    // for(size_t i = 0; i < inputs.size(); i++) {
    //     tensor_t<value_type>* tmp = new tensor_t<value_type>( dEdD_next->get_N(), dEdD_next->get_C(), dEdD_next->get_H(), dEdD_next->get_W(), reg->get_vector(), B_DATA, this->get_id());
    //     this->set_b_data(tmp, inputs[i], reg);
    // }
    for(size_t i = 0; i < inputs.size(); i++) {
        // tensor_t<value_type>* tmp = new tensor_t<value_type>( dEdD_next->get_N(), dEdD_next->get_C(), dEdD_next->get_H(), dEdD_next->get_W(), reg->get_vector(), B_DATA, this->get_id());
        this->set_b_data(dEdD_next, inputs[i], reg);

    }
    printf("b_datas.size() = %d\n", this->get_b_data().size());

    std::vector<tensor_t<value_type>*> b_datas = this->get_b_data();
    for(size_t i = 0; i < b_datas.size(); i++) {
        reg->register_backward_dependency(this->get_id(), b_datas[i] );
    }
    // inputs  = this->get_inputs_keys();
    // tensor_t<value_type>* source = reg->get_reg_b_data(inputs[0].second, inputs[0].first);
    dEdD_next->increase_b_use_count();
    reg->register_backward_dependency(this->get_id(), dEdD_next );

    std::vector<tensor_t<value_type>* > t_ins = this->get_inputs();
    for (int i = 0; i < t_ins.size(); i++) {
        // t_ins[i]->increase_use_counter(BACKWARD);
        reg->register_backward_dependency(this->get_id(), t_ins[i] );
        t_ins[i]->printTensorState("join layer input");
    }
}


template <class value_type>
std::vector<value_type> join_layer_t<value_type>::forward(network_stage stage, cublasHandle_t* cublas_h, cudnnHandle_t* cudnn_h, registry_t<value_type>* reg) {
    assert( cudnn_h != NULL );
    assert( reg     != NULL );
    //we use the input tensor of idx 0 as the output,
    //the subsequents should reduce to idx 0
    std::vector<tensor_t<value_type>* > inputs = this->get_inputs();
#ifdef DEBUG
    printf("@join layer forward:\n");
    printf("input 0 : %p, layer %d\n", inputs[0], inputs[0]->get_layer_id());
    inputs[0]->printTensor("@ join layer inputs");
#endif
    std::vector<tensor_t<value_type>* > outputs = this->get_outputs();
	outputs[0]->copy(inputs[0]);
	// inputs[0]->decrease_use_count();
    inputs[0]->increase_cur_use_counter(FORWARD);
    for (size_t i = 1; i < inputs.size(); i++) {
        outputs[0]->sum( inputs[i] );
		// inputs[i]->decrease_use_count();
        inputs[i]->increase_cur_use_counter(FORWARD);
    }
//     for (size_t i = 1; i < inputs.size(); i++) {
//         inputs[0]->sum( inputs[i] );
// 		inputs[i]->decrease_use_count();
// #ifdef DEBUG
//         inputs[i]->printTensor("@ join layer inputs");
//         printf("input %zu : %p, layer %d\n", i, inputs[i], inputs[i]->get_layer_id());
// #endif
//     }
//     this->get_outputs()[0]->copy(inputs[0]);

	//inputs[0]->decrease_use_count();
    //const value_type scale_factor = 1.0f/ (value_type) inputs.size();
    //inputs[0]->scale( scale_factor );
#ifdef DEBUG
    tensor_t<value_type>* output = (this->get_outputs())[0];
    output->printTensor("the output of join layer");
#endif
    return std::vector<value_type>();
}

template <class value_type>
void join_layer_t<value_type>::backward(network_stage stage, cublasHandle_t* cublas_h, cudnnHandle_t* cudnn_h, registry_t<value_type>* reg ) {
    assert( cudnn_h != NULL );
    assert( reg     != NULL );

#ifdef DEBUG
    printf("@join layer backward:\n");
#endif
    
    // std::vector<std::pair<int, int> > inputs  = this->get_inputs_keys();
    // tensor_t<value_type>* source = reg->get_reg_b_data(inputs[0].second, inputs[0].first);
    // for(size_t i = 1; i < inputs.size(); i++) {
    //     tensor_t<value_type>* target = reg->get_reg_b_data(inputs[i].second, inputs[i].first);
    //     target->copy(source);
    // }

    std::pair<int, int>    output_k  = (this->get_outputs_keys())[0];
    tensor_t<value_type>* dEdD_next = reg->get_reg_b_data(output_k.second, output_k.first);
    assert(dEdD_next != NULL);
    std::vector<tensor_t<value_type>*> b_datas  = this->get_b_data();
    // for(size_t i = 0; i < b_datas.size(); i++) {
    //     b_datas[i]->copy(dEdD_next);
    // }
    dEdD_next->decrease_cur_b_use_count();
    // printf("dEdD_next%d is_cur_b_use_count_zero=%d\n", dEdD_next->get_tensor_id(), dEdD_next->is_cur_b_use_count_zero());

	auto inputs2 = this->get_inputs();
	for (size_t i = 0; i < inputs2.size(); i++) {
		// inputs2[i]->increase_use_count();
        // inputs2[i]->increase_cur_use_counter(BACKWARD);
	}
#ifdef DEBUG
    source->printTensor("Source b_data @ join layer");
    for(size_t i = 0; i < inputs.size(); i++) {
        tensor_t<value_type>* target = reg->get_reg_b_data(inputs[i].second, inputs[i].first);
        target->printTensor("destination b_data@ join layer");
    }
#endif
}

template <class value_type>
void join_layer_t<value_type>::fake_run(net_comp dir, registry_t<value_type>* reg) {
    if (dir == FORWARD) {
	    auto inputs = this->get_inputs();
        for (size_t i = 0; i < inputs.size(); i++) {
            inputs[i]->increase_cur_use_counter(FORWARD);
        }
    }
    else {
        auto inputs = this->get_inputs();
        for (size_t i = 0; i < inputs.size(); i++) {
            // inputs[i]->increase_cur_use_counter(BACKWARD);
	    }
    }
}

INSTANTIATE_CLASS(join_layer_t);
    
} //ATP namespace
