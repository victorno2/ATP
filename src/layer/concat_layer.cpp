//
// Created by ay27 on 17/6/20.
//

#include <layer/concat_layer.h>

namespace ATP {

template<class value_type>
void concat_layer_t<value_type>::forward_setup(registry_t<value_type> *reg, cudnnHandle_t *cudnn_h) {
    printf("======>setup the forward concat layer@%d\n", this->get_id());
    std::vector<std::pair<int, int> > input_keys = this->get_inputs_keys();
	this->clear_inputs();
    this->clear_outputs();
    for (size_t i = 0; i < input_keys.size(); i++) {
        std::pair<int, int> input_key = input_keys[i];
        tensor_t<value_type> *input = reg->get_reg_output(input_key.first, input_key.second);
		input->increase_use_count_initial();
		input->increase_use_count();
        assert(input != NULL);
        this->set_input(input);
        // input->set_backward_useful(false);
        input->increase_use_counter(FORWARD);
    }

    // concat layer only has 1 output
    size_t N = (this->get_inputs())[0]->get_N();
    // concat according C axis
    size_t C = 0;
    for (size_t i = 0; i < this->get_inputs().size(); ++i) {
        C += this->get_inputs()[i]->get_C();
    }
    size_t H = (this->get_inputs())[0]->get_H();
    size_t W = (this->get_inputs())[0]->get_W();

    tensor_t<value_type> *output = new tensor_t<value_type>(N, C, H, W, reg->get_vector(), DATA, this->get_id());
    this->set_output(output, (this->get_outputs_keys())[0], reg);
    printf("concat layer output%d desc: %d %d %d %d\n", 
        this->get_outputs()[0]->get_tensor_id(), this->get_outputs()[0]->get_N(), this->get_outputs()[0]->get_C(), this->get_outputs()[0]->get_H(), this->get_outputs()[0]->get_W());

    // register the forward dependency
    std::vector<tensor_t<value_type>* > inputs = this->get_inputs();
    for (size_t i = 0; i < inputs.size(); i++) {
        reg->register_forward_dependency( this->get_id(), inputs[i] );
    }
    reg->register_forward_dependency(this->get_id(), output);
    // output->set_use_in_backward(false);
    printf("forward_setup layer%d concat down\n", this->get_id());
}

template<class value_type>
void concat_layer_t<value_type>::backward_setup(registry_t<value_type> *reg, cudnnHandle_t *cudnn_h) {
    // the backward data should have the same size with input data ?

    std::pair<int, int> output_k = this->get_outputs_keys()[0];
    tensor_t<value_type> *source = reg->get_reg_b_data(output_k.second, output_k.first);
    source->increase_b_use_count();
    this->clear_dy();
    this->set_dy(source);

	this->clear_b_data();
    // for (size_t i = 0; i < this->get_inputs().size(); ++i) {

    //     size_t N = this->get_inputs()[i]->get_N();
    //     size_t C = this->get_inputs()[i]->get_C();
    //     size_t H = this->get_inputs()[i]->get_H();
    //     size_t W = this->get_inputs()[i]->get_W();

    //     tensor_t<value_type> *tmp = new tensor_t<value_type>(N, C, H, W, reg->get_vector(), B_DATA, this->get_id());
    //     this->set_b_data(tmp, this->get_inputs_keys()[i], reg);
    // }

    for (size_t i = 0; i < this->get_inputs().size(); ++i) {

        size_t N = this->get_inputs()[i]->get_N();
        size_t C = this->get_inputs()[i]->get_C();
        size_t H = this->get_inputs()[i]->get_H();
        size_t W = this->get_inputs()[i]->get_W();

        tensor_t<value_type> *tmp = new tensor_t<value_type>(N, C, H, W, reg->get_vector(), B_DATA, this->get_id());
        this->set_b_data(tmp, this->get_inputs_keys()[i], reg);
    }

    // register the backward dependency
    std::vector<std::pair<int, int> > inputs  = this->get_inputs_keys();
    for(size_t i = 0; i < inputs.size(); i++) {
        tensor_t<value_type>* target = reg->get_reg_b_data(inputs[i].second, inputs[i].first);
        reg->register_backward_dependency(this->get_id(), target );
    }
    reg->register_backward_dependency(this->get_id(), source );
}


template<class value_type>
std::vector<value_type>
concat_layer_t<value_type>::forward(network_stage stage, cublasHandle_t *cublas_h, cudnnHandle_t *cudnn_h,
                                  registry_t<value_type> *reg) {
    assert(cudnn_h != NULL);
    assert(reg != NULL);
    std::vector<tensor_t<value_type> *> inputs = this->get_inputs();
    tensor_t<value_type> *output = (this->get_outputs())[0];

    // stack output by inputs according to C axis
//    size_t one_input_size = inputs[0]->get_C() * inputs[0]->get_H() * inputs[0]->get_W();
    size_t offset = 0;
    for (size_t i = 0; i < inputs.size(); i++) {
        // output size is larger than input, so stack the input !
        output->copy(inputs[i], -1, -1, (int)offset, (int)(offset+inputs[i]->get_N()*inputs[i]->get_C()*inputs[i]->get_H()*inputs[i]->get_W()));

        offset += inputs[i]->get_N()*inputs[i]->get_C()*inputs[i]->get_H()*inputs[i]->get_W();
		// inputs[i]->decrease_use_count();
        inputs[i]->increase_cur_use_counter(FORWARD);

#ifdef DEBUG
        inputs[i]->printTensor("@ concat layer inputs");
#endif
    }
#ifdef DEBUG
    output->printTensor("the output of concat layer");
#endif
    return std::vector<value_type>();
}

template<class value_type>
void concat_layer_t<value_type>::backward(network_stage stage, cublasHandle_t *cublas_h, cudnnHandle_t *cudnn_h,
                                        registry_t<value_type> *reg) {
    assert(cudnn_h != NULL);
    assert(reg != NULL);

    std::vector<std::pair<int, int> > inputs = this->get_inputs_keys();
    std::pair<int, int> output_k = this->get_outputs_keys()[0];
    tensor_t<value_type> *source = reg->get_reg_b_data(output_k.second, output_k.first);
    source->decrease_cur_b_use_count();
    size_t offset = 0;

    for (size_t i = 0; i < inputs.size(); i++) {
        tensor_t<value_type> *target = reg->get_reg_b_data(inputs[i].second, inputs[i].first);
        // unfold output to input
        target->copy(source, (int)offset, (int)(offset + target->get_N()*target->get_C()*target->get_H()*target->get_W()), -1, -1);
        offset += target->get_N()*target->get_C()*target->get_H()*target->get_W();
    }

	std::vector<tensor_t<value_type> *> inputs2 = this->get_inputs();
	for (size_t i = 0; i < inputs2.size(); i++) {
		// inputs2[i]->increase_use_count();
	}
	
#ifdef DEBUG
    source->printTensor("Source b_data @ concat layer");
    for (size_t i = 0; i < inputs.size(); i++) {
        tensor_t<value_type> *target = reg->get_reg_b_data(inputs[i].second, inputs[i].first);
        target->printTensor("destination b_data@ concat layer");
    }

    printf("concat backward finish\n");
#endif
}

template <class value_type>
void concat_layer_t<value_type>::fake_run(net_comp dir, registry_t<value_type>* reg) {
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

INSTANTIATE_CLASS(concat_layer_t);

} //ATP namespace
