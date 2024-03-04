#include <layer/fork_layer.h>

namespace ATP {
    
template <class value_type>
void fork_layer_t<value_type>::forward_setup( registry_t<value_type>* reg, cudnnHandle_t* cudnn_h ) {
    printf("======>setup the forward fork layer@%d\n", this->get_id() );
    std::pair<int, int> input_key = (this->get_inputs_keys())[0];
    tensor_t<value_type>* input = reg->get_reg_output(input_key.first, input_key.second);
    input->increase_use_counter(FORWARD);
	// input->increase_use_count_initial();
	// input->increase_use_count();
	this->clear_inputs();
    this->set_input( input );
    input->set_backward_useful(true);
    this->get_inputs();
    printf("next layer size = %d, prev layer size = %d\n", ((base_layer_t<value_type>*)this)->get_next().size(), ((base_layer_t<value_type>*)this)->get_prev().size());
    // it has to be hard copy to avoid multiple writes
    std::vector<std::pair<int, int> > output_keys = this->get_outputs_keys();

    // should not overlap tensor !!!!
    // this->set_output( input, output_keys[0], reg);
	this->clear_outputs();
    // for(size_t i = 0; i < output_keys.size(); i++) {
    //     tensor_t<value_type>* tmp = new tensor_t<value_type>(input->get_N(), input->get_C(), input->get_H(), input->get_W(), reg->get_vector(), DATA, this->get_id());
    //     // tmp->set_require_space(false);
    //     this->set_output(tmp, output_keys[i], reg);
    //     tmp->set_use_in_backward(false);
    // }
    for(size_t i = 0; i < output_keys.size(); i++) {
        this->set_output(input, output_keys[i], reg);
    }
    // input->set_use_in_backward(false);
    
    // register the forward dependency
    // please be noted the input is outputs[0]
    std::vector<tensor_t<value_type>* > outputs = this->get_outputs();
    for (size_t i = 0; i < outputs.size(); i++) {
        reg->register_forward_dependency( this->get_id(), outputs[i] );
    }
    reg->register_forward_dependency( this->get_id(), input);
    printf("======>End setup the forward fork layer@%d\n", this->get_id() );
}

template <class value_type>
void fork_layer_t<value_type>::backward_setup( registry_t<value_type>* reg, cudnnHandle_t* cudnn_h ) {
    //backward
    printf("======>setup the backward fork layer@%d\n", this->get_id() );
    //we use the first output_keys as the output b_data
    std::pair<int, int> b_data_key = (this->get_inputs_keys())[0];
    std::vector<std::pair<int, int> > dEdD_next_keys = this->get_outputs_keys();
    this->clear_dy();
    for(size_t i = 0; i < dEdD_next_keys.size(); i++) {
        tensor_t<value_type>* tmp = reg->get_reg_b_data(dEdD_next_keys[i].second, dEdD_next_keys[i].first);
        this->set_dy(tmp);
    }

    tensor_t<value_type>* dEdD = reg->get_reg_b_data( dEdD_next_keys[0].second, dEdD_next_keys[0].first );
    tensor_t<value_type>* b_data = new tensor_t<value_type>(dEdD->get_N(), dEdD->get_C(), dEdD->get_H(), dEdD->get_W(), reg->get_vector(), B_DATA, this->get_id());
    this->clear_b_data();
    this->set_b_data(b_data, b_data_key, reg);
	reg->register_backward_dependency(this->get_id(), b_data);
    for(size_t i = 0; i < dEdD_next_keys.size(); i++) {
        tensor_t<value_type>* tmp = reg->get_reg_b_data(dEdD_next_keys[i].second, dEdD_next_keys[i].first);
        tmp->increase_b_use_count();
		assert(tmp != NULL);
        assert( tmp->get_N() == b_data->get_N() );
        assert( tmp->get_C() == b_data->get_C() );
        assert( tmp->get_H() == b_data->get_H() );
        assert( tmp->get_W() == b_data->get_W() );
        reg->register_backward_dependency(this->get_id(), tmp );
    }
    reg->register_backward_dependency(this->get_id(), (this->get_inputs())[0] );
    printf("next layer size = %d, prev layer size = %d\n", ((base_layer_t<value_type>*)this)->get_next().size(), ((base_layer_t<value_type>*)this)->get_prev().size());
    printf("======>End setup the backward fork layer@%d\n", this->get_id() );
}
   

template <class value_type>
std::vector<value_type> fork_layer_t<value_type>::forward(network_stage stage, cublasHandle_t* cublas_h, cudnnHandle_t* cudnn_h, registry_t<value_type>* reg) {
    assert( cudnn_h != NULL );
    assert( reg     != NULL );
    //the input layer set the output tensor in the idx 0,
    //the subsequents should copy from idx0
    tensor_t<value_type>* input = this->get_inputs()[0];
    input->increase_cur_use_counter(FORWARD);
    std::vector<tensor_t<value_type>* > inputs = this->get_inputs();
    // printf("when fork layer forward, inputs_size = %d\n", inputs.size());
	// input->decrease_use_count();
    std::vector<tensor_t<value_type>* > outputs = this->get_outputs();
    // for (size_t i = 0; i < outputs.size(); i++) {
    //     outputs[i]->copy( input );
    //     // outputs[i]->set_gpu_ptr(input->get_gpu_ptr());
    // }

#ifdef DEBUG
    printf("f fork layer : %d\n", this->get_id());
    input->printTensor("input @ fork layer");
    for (size_t i=0; i<outputs.size(); i++) {
        printf("output %zu tensor %p, layer %d\n", i, outputs[i], outputs[i]->get_layer_id());
        outputs[i]->printTensor("output @ fork layer");
    }
#endif

    return std::vector<value_type>();
}

template <class value_type>
void fork_layer_t<value_type>::backward(network_stage stage, cublasHandle_t* cublas_h, cudnnHandle_t* cudnn_h, registry_t<value_type>* reg ) {
    assert( cudnn_h != NULL );
    assert( reg     != NULL );
    std::vector<std::pair<int, int> > dEdD_next_keys = this->get_outputs_keys();
    tensor_t<value_type>* b_data = (this->get_b_data())[0];
	tensor_t<value_type>* input = this->get_inputs()[0];
	// input->increase_use_count();
#ifdef DEBUG
    printf("b fork layer : %d\n", this->get_id());
    b_data->printTensor("backward data src @ fork layer");
#endif
    
    // for(size_t i = 1; i < dEdD_next_keys.size(); i++) {
    //     tensor_t<value_type>* tmp = reg->get_reg_b_data(dEdD_next_keys[i].second, dEdD_next_keys[i].first);
    //     tmp->decrease_cur_b_use_count();
    //     b_data->sum( tmp );
    // }
    //const value_type scale_factor = 1.0f/ (value_type)dEdD_next_keys.size();
    //b_data->scale( scale_factor );
	tensor_t<value_type>* tmp = reg->get_reg_b_data(dEdD_next_keys[0].second, dEdD_next_keys[0].first);
    b_data->copy(tmp);
	tmp->decrease_cur_b_use_count();
    for(size_t i = 1; i < dEdD_next_keys.size(); i++) {
        tensor_t<value_type>* tmp = reg->get_reg_b_data(dEdD_next_keys[i].second, dEdD_next_keys[i].first);
        tmp->decrease_cur_b_use_count();
        b_data->sum( tmp );
    }

#ifdef DEBUG
    printf("backward src 0, tensor %p layer %d\n", b_data, b_data->get_layer_id());
    for(size_t i = 1; i < dEdD_next_keys.size(); i++) {
        tensor_t<value_type>* tmp = reg->get_reg_b_data(dEdD_next_keys[i].second, dEdD_next_keys[i].first);
        tmp->printTensor("backward data src @ fork layer");
        printf("backward src %zu, tensor %p layer %d\n", i, tmp, tmp->get_layer_id());
    }

    b_data->printTensor("backward data dst @ fork layer");
#endif    
}

template <class value_type>
void fork_layer_t<value_type>::fake_run(net_comp dir, registry_t<value_type>* reg) {
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

INSTANTIATE_CLASS(fork_layer_t);
    
} //ATP namespace
