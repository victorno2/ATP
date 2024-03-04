
#include <layer/cudnn_activation_layer.h>

namespace ATP {
    
template <class value_type>
void act_layer_t<value_type>::forward_setup( registry_t<value_type>* reg, cudnnHandle_t* cudnn_h ) {
    // printf("======>setup the forward activation layer:%d\n", this->get_id());
    int curt_l  = this->get_id();
    int input_l = this->get_input_layer_id();
    tensor_t<value_type>* t_in = reg->get_reg_output(input_l, curt_l);
	t_in->increase_use_counter(FORWARD);
    assert( t_in != NULL );
    // t_in->increase_use_count_initial();
	// t_in->increase_use_count();
	this->set_f_in(t_in, reg);
	
    checkCUDNN( cudnnCreateActivationDescriptor( &(this->act_desc) ) );
    checkCUDNN( cudnnSetActivationDescriptor(this->act_desc,
                                 this->mode,
                                 this->p_nan, 0) );
    tensor_t<value_type>* t_out  = new tensor_t<value_type>(t_in->get_N(), t_in->get_C(), t_in->get_H(), t_in->get_W(), reg->get_vector(), DATA, this->get_id());

    this->set_f_out( t_out, reg );
    assert( this->get_f_out()  != NULL );
    
    //register the forward dependency
    t_in  = reg->get_reg_output(input_l, curt_l);
    t_out = this->get_f_out();
    reg->register_forward_dependency( this->get_id(), t_in );
    reg->register_forward_dependency( this->get_id(), t_out );

}
    
template <class value_type>
void act_layer_t<value_type>::backward_setup( registry_t<value_type>* reg, cudnnHandle_t* cudnn_h ) {
    //backward
    // printf("======>setup the backward activation layer:%d\n", this->get_id());
    int curt_l_id   = this->get_id();
    int input_l_id  = this->get_input_layer_id();
    int output_l_id = this->get_output_layer_id();
    tensor_t<value_type>* dEdD   = reg->get_reg_b_data(output_l_id, curt_l_id);
    this->set_dy(dEdD);
    dEdD->increase_b_use_count();
    // assert(dEdD != NULL);
    
    tensor_t<value_type> *b_data = new tensor_t<value_type>(dEdD->get_N(), dEdD->get_C(), dEdD->get_H(), dEdD->get_W(),
                                                             reg->get_vector(), B_DATA, this->get_id());
    this->set_b_data(b_data, reg);
    assert( this->get_b_data() != NULL );

    //register the backward dependency
    dEdD                        = reg->get_reg_b_data(output_l_id, curt_l_id);
    tensor_t<value_type>* t_out = this->get_f_out();
    tensor_t<value_type>* t_in  = this->get_f_in();  // reg->get_reg_output(input_l_id, curt_l_id);
    t_in->increase_use_counter(BACKWARD);
    t_out->increase_use_counter(BACKWARD);
    assert( dEdD  != NULL );
    assert( t_out != NULL );
    assert( t_in  != NULL );
    
    reg->register_backward_dependency(this->get_id(), b_data  );
    reg->register_backward_dependency(this->get_id(), t_out  );
    reg->register_backward_dependency(this->get_id(), t_in   );
    reg->register_backward_dependency(this->get_id(), dEdD   );

}


template <class value_type>
std::vector<value_type> act_layer_t<value_type>::forward(network_stage stage, cublasHandle_t* cublas_h, cudnnHandle_t* cudnn_h, registry_t<value_type>* reg) {
    assert( cudnn_h != NULL );
    assert( reg     != NULL );
    
    int input_l = this->get_input_layer_id();
    int curt_l  = this->get_id();
    tensor_t<value_type>* t_in  = reg->get_reg_output(input_l, curt_l);
    t_in->increase_cur_use_counter(FORWARD);
	// t_in->decrease_use_count();
    tensor_t<value_type>* t_out = this->get_f_out();
    
#ifdef DEBUG
    printf("input tensor from %d to %d\n", input_l, curt_l);
#endif
    // t_out->printTensor("");
    checkCUDNN( cudnnActivationForward(
                                        *(cudnn_h),
                                        this->act_desc,
                                        &(this->one),
                                        t_in->get_tensor_desc(),
                                        t_in->get_gpu_ptr(),
                                        &(this->zero),
                                        t_out->get_tensor_desc(),
                                        t_out->get_gpu_ptr() ) );
    
#ifdef DEBUG
    this->get_f_out()->printTensor("activation, forward output");
//    this->get_f_out()->GPUtoCPU();
#endif
    return std::vector<value_type>();
}
    
template <class value_type>
void act_layer_t<value_type>::backward(network_stage stage, cublasHandle_t* cublas_h, cudnnHandle_t* cudnn_h, registry_t<value_type>* reg ) {
    assert( cudnn_h != NULL );
    assert( reg     != NULL );
    
    int curt_l_id   = this->get_id();
    int input_l_id  = this->get_input_layer_id();
    int output_l_id = this->get_output_layer_id();
    
    tensor_t<value_type>* b_data = this->get_b_data();
    // printf("backward act layer%d b_data%d [%d %d %d %d]\n", this->get_id(), b_data->get_tensor_id(), b_data->get_N(), b_data->get_C(), b_data->get_H(), b_data->get_W());
    tensor_t<value_type>* t_out = this->get_f_out();
    // printf("backward act layer%d t_out%d [%d %d %d %d]\n", this->get_id(), t_out->get_tensor_id(), t_out->get_N(), t_out->get_C(), t_out->get_H(), t_out->get_W());
    tensor_t<value_type>* t_in  = this->get_f_in();
    t_in->increase_cur_use_counter(BACKWARD);
    t_out->increase_cur_use_counter(BACKWARD);
    // printf("backward act layer%d t_in%d [%d %d %d %d]\n", this->get_id(), t_in->get_tensor_id(), t_in->get_N(), t_in->get_C(), t_in->get_H(), t_in->get_W());
	// t_in->increase_use_count();
    tensor_t<value_type>* dEdD  = reg->get_reg_b_data(output_l_id, curt_l_id);
    // printf("backward act layer%d dEdD%d [%d %d %d %d]\n", this->get_id(), dEdD->get_tensor_id(), dEdD->get_N(), dEdD->get_C(), dEdD->get_H(), dEdD->get_W());
	// t_in->increase_use_count();
    // dEdD->decrease_cur_b_use_count();
    // t_in->printTensorState("act in backward");
    // t_out->printTensorState("act out backward");
    checkCUDNN( cudnnActivationBackward(
                        *(cudnn_h),
                        this->act_desc,
                        &(this->one),
                        t_out->get_tensor_desc(),
                        t_out->get_gpu_ptr(),
                        // t_in->get_gpu_ptr(),
                        dEdD->get_tensor_desc(),
                        dEdD->get_gpu_ptr(),
                        t_in->get_tensor_desc(),
                        t_in->get_gpu_ptr(),
                        &(this->zero),
                        b_data->get_tensor_desc(),
                        b_data->get_gpu_ptr()) );
                        // dEdD->get_tensor_desc(),
                        // dEdD->get_gpu_ptr()) );
#ifdef DEBUG
    dEdD->printTensor("Result of Backward Activation");
//    dEdD->GPUtoCPU();
#endif
}
    
template <class value_type>
void act_layer_t<value_type>::fake_run(net_comp dir, registry_t<value_type>* reg) {
    if (dir == FORWARD) {
        int input_l = this->get_input_layer_id();
        int curt_l  = this->get_id();
        tensor_t<value_type>* input = reg->get_reg_output(input_l, curt_l);
        input->increase_cur_use_counter(FORWARD);
    }
    else {
        int input_l = this->get_input_layer_id();
        int curt_l  = this->get_id();
        tensor_t<value_type>* input = reg->get_reg_output(input_l, curt_l);
        input->increase_cur_use_counter(BACKWARD);
        tensor_t<value_type>* output = this->get_f_out();
        output->increase_cur_use_counter(BACKWARD);
    }
}

INSTANTIATE_CLASS(act_layer_t);
    
} //ATP namespace
