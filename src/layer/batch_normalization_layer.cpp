#include <layer/batch_normalization_layer.h>

namespace ATP {
    
template <class value_type>
void batch_normalization_layer_t<value_type>::forward_setup(registry_t<value_type>* reg, cudnnHandle_t* cudnn_h) {
    printf("======>setup the forward batch normalization layer:%d\n", this->get_id());
    //hook the output of previous layer
    int input_l = this->get_input_layer_id();
    int curt_l  = this->get_id();
    tensor_t<value_type>* input = reg->get_reg_output(input_l, curt_l);
    input->increase_use_counter(FORWARD);
    assert( input != NULL );
	this->set_f_in(input, reg);
	// input->increase_use_count_initial();
    // input->increase_use_count();
	
    tensor_t<value_type>* f_out  = new tensor_t<value_type>( input->get_N(), input->get_C(), input->get_H(), input->get_W(), reg->get_vector(), DATA, this->get_id());
    //setup the output tensor
    this->set_f_out( f_out, reg );

    //gamma is the weight
    //beta  is the bias
    tensor_t<value_type>* gamma = NULL;
    tensor_t<value_type>* beta  = NULL;
    if(this->mode == CUDNN_BATCHNORM_SPATIAL) {
        //gamma, beta tensor dims are 1xCx1x1
        //(one value per C-dim normalized over Nx1xHxW subtensors)
        gamma                      = new tensor_t<value_type>(1, input->get_C(), 1, 1, reg->get_vector(), PARAM, this->get_id());
        beta                       = new tensor_t<value_type>(1, input->get_C(), 1, 1, reg->get_vector(), PARAM, this->get_id());
        this->resultRunningMean     = new tensor_t<value_type>(1, input->get_C(), 1, 1, reg->get_vector(), BN_MEAN_VAR, this->get_id()); //TO DO
        this->resultRunningVariance = new tensor_t<value_type>(1, input->get_C(), 1, 1, reg->get_vector(), BN_MEAN_VAR, this->get_id()); //TO DO
        beta->init(new constant_initializer_t<value_type>(0));
        gamma->init(new constant_initializer_t<value_type>(1));
//        beta->const_fill(0);
//        gamma->const_fill(1);

        resultRunningMean->init(new constant_initializer_t<value_type>(0));
        resultRunningVariance->init(new constant_initializer_t<value_type>(0));

//        resultRunningMean->const_fill(0);
//        resultRunningVariance->const_fill(0);
        
    } else if(this->mode == CUDNN_BATCHNORM_PER_ACTIVATION) {
        // gamma, beta tensor dims are 1xCxHxWx..
        // (one value per CHW...-slice, normalized over N slice)
        gamma                      = new tensor_t<value_type>(1, input->get_C(), input->get_H(), input->get_W(), reg->get_vector(), PARAM, this->get_id());
        beta                       = new tensor_t<value_type>(1, input->get_C(), input->get_H(), input->get_W(), reg->get_vector(), PARAM, this->get_id());
        this->resultRunningMean     = new tensor_t<value_type>(1, input->get_C(), input->get_H(), input->get_W(), reg->get_vector(), BN_MEAN_VAR, this->get_id());
        this->resultRunningVariance = new tensor_t<value_type>(1, input->get_C(), input->get_H(), input->get_W(), reg->get_vector(), BN_MEAN_VAR, this->get_id());
        // according to the default value in tensorflow, the beta should initialize as 0
        beta->init(new constant_initializer_t<value_type>(0));
        gamma->init(new constant_initializer_t<value_type>(1));
//        beta->const_fill(0);
//        gamma->const_fill(1);

        resultRunningMean->init(new constant_initializer_t<value_type>(0));
        resultRunningVariance->init(new constant_initializer_t<value_type>(0));
//        resultRunningMean->const_fill(0);
//        resultRunningVariance->const_fill(0);
    }
    assert(gamma != NULL);
    assert(beta  != NULL);
    tensor_t<value_type>* beta_prev  = new tensor_t<value_type>(gamma->get_N(), gamma->get_C(), gamma->get_H(), gamma->get_W(), reg->get_vector(), GRAD, this->get_id());
    tensor_t<value_type>* gamma_prev = new tensor_t<value_type>(gamma->get_N(), gamma->get_C(), gamma->get_H(), gamma->get_W(), reg->get_vector(), GRAD, this->get_id());
    beta_prev->init(new constant_initializer_t<value_type>(0));
    gamma_prev->init(new constant_initializer_t<value_type>(0));
//    beta_prev->const_fill( 0 );
//    gamma_prev->const_fill( 0 );
    //we treat gamma and beta as the layer params
    this->set_bias( beta, reg );
    this->set_weight( gamma, reg );
    this->set_bias_prev( beta_prev, reg );
    this->set_weight_prev( gamma_prev, reg );

    //forward hookup check
    assert( this->get_bias() != NULL );
    assert( this->get_f_out() != NULL );
    assert( this->get_weight() != NULL );
    assert( this->get_bias_prev() != NULL );
    assert( this->get_weight_prev() != NULL );
    
    //register the forward dependency
    tensor_t<value_type>* t_in   = reg->get_reg_output(input_l, curt_l);
    tensor_t<value_type>* t_out  = this->get_f_out();
    gamma  = this->get_weight();
    beta   = this->get_bias();
    
    reg->register_forward_dependency( this->get_id(), t_in  );
    reg->register_forward_dependency( this->get_id(), t_out );
    reg->register_forward_dependency( this->get_id(), gamma );
    reg->register_forward_dependency( this->get_id(), beta  );
    reg->register_forward_dependency( this->get_id(), resultRunningMean  );
    reg->register_forward_dependency( this->get_id(), resultRunningVariance  );
    printf("======>setup the forward batch normalization layer done:%d\n", this->get_id());
}
    
template <class value_type>
void batch_normalization_layer_t<value_type>::backward_setup(registry_t<value_type>* reg, cudnnHandle_t* cudnn_h) {
    printf("======>setup the backward batch normalization layer:%d\n", this->get_id());

    //setup the backward data
    int curt_l_id   = this->get_id();
    int input_l_id  = this->get_input_layer_id();
    int output_l_id = this->get_output_layer_id();

    tensor_t<value_type>* input = reg->get_reg_output(input_l_id, curt_l_id);
    input->increase_use_counter(BACKWARD);
    input->printTensorState("");
    
    tensor_t<value_type>* b_data = new tensor_t<value_type>(input->get_N(), input->get_C(), input->get_H(), input->get_W(), reg->get_vector(), B_DATA, this->get_id());
    this->set_b_data(b_data, reg);
    tensor_t<value_type>* g  = reg->get_reg_weight(curt_l_id);
    tensor_t<value_type>* b  = reg->get_reg_bias(curt_l_id);
    assert( g != NULL );
    assert( b != NULL );
    //TO DO: BN layer has two parameters to update, like weight(gamma) and bias(beta)
    tensor_t<value_type>* dEdGa = new tensor_t<value_type>( g->get_N(), g->get_C(), g->get_H(), g->get_W(), reg->get_vector(), GRAD, this->get_id());
    tensor_t<value_type>* dEdBe = new tensor_t<value_type>( b->get_N(), b->get_C(), b->get_H(), b->get_W(), reg->get_vector(), GRAD, this->get_id());
    //weight for dEdGa
    //bias   for dEdBe
    this->set_bias_grad( dEdBe, reg );
    this->set_weight_grad( dEdGa, reg );
    
    assert( this->get_weight_grad() != NULL );
    assert( this->get_bias_grad()   != NULL );
    assert( this->get_b_data()      != NULL );
    
    //register the backward dependency
    tensor_t<value_type>* t_in     = reg->get_reg_output(input_l_id,  curt_l_id);
    tensor_t<value_type>* dEdD_n   = reg->get_reg_b_data(output_l_id, curt_l_id);
    this->set_dy(dEdD_n);
    dEdD_n->increase_b_use_count();
    tensor_t<value_type>* dEdD_c   = this->get_b_data();
    tensor_t<value_type>* gamma    = this->get_weight();
    dEdGa                          = this->get_weight_grad();
    dEdBe                          = this->get_bias_grad();
    
    assert( t_in    != NULL );
    assert( dEdD_n  != NULL );
    assert( dEdD_c  != NULL );
    assert( dEdGa != NULL );
    assert( dEdBe != NULL );
    assert( gamma != NULL );
    
    reg->register_backward_dependency(this->get_id(), t_in    );
    reg->register_backward_dependency(this->get_id(), dEdD_n  );
    reg->register_backward_dependency(this->get_id(), dEdD_c  );
    reg->register_backward_dependency(this->get_id(), dEdGa   );
    reg->register_backward_dependency(this->get_id(), dEdBe   );
    reg->register_backward_dependency(this->get_id(), gamma   );

}

template <class value_type>
std::vector<value_type> batch_normalization_layer_t<value_type>::forward(network_stage stage, cublasHandle_t* cublas_h, cudnnHandle_t* cudnn_h, registry_t<value_type>* reg) {
    //gamma is the weight
    //beta  is the bias

    int input_l = this->get_input_layer_id();
    int curt_l  = this->get_id();
    tensor_t<value_type>* t_in   = reg->get_reg_output(input_l, curt_l);
    t_in->increase_cur_use_counter(FORWARD);
	// t_in->decrease_use_count();
    tensor_t<value_type>* t_out  = this->get_f_out();
    tensor_t<value_type>* gamma  = this->get_weight();
    tensor_t<value_type>* beta   = this->get_bias();

    value_type* running_mean;
    value_type* running_var;

    // TODO : this is not correct !!!!!!

//    if (stage == NET_TRAIN) {
        this->iter += 1;
//        running_mean  = resultRunningMean->get_gpu_ptr();
//        running_var   = resultRunningVariance->get_gpu_ptr();
//    } else {
//        size_t size = resultRunningMean->get_C()*resultRunningMean->get_H()*resultRunningMean->get_W()* sizeof(value_type);
//        cudaMalloc(&running_mean, size);
//        cudaMalloc(&running_var, size);
//        cudaMemcpy(running_mean, resultRunningMean->get_gpu_ptr(), size, cudaMemcpyDeviceToDevice);
//        cudaMemcpy(running_var, resultRunningVariance->get_gpu_ptr(), size, cudaMemcpyDeviceToDevice);
//    }
        // t_out->printTensor("");
        double avg_factor = 1.0f / (this->iter);
        checkCUDNN( cudnnBatchNormalizationForwardTraining(
                                                           *(cudnn_h),
                                                           this->mode,
                                                           &(this->one),
                                                           &(this->zero),
                                                           t_in->get_tensor_desc(),
                                                           t_in->get_gpu_ptr(),
                                                           t_out->get_tensor_desc(),
                                                           t_out->get_gpu_ptr(),
                                                           gamma->get_tensor_desc(),
                                                           gamma->get_gpu_ptr(),
                                                           beta->get_gpu_ptr(),
                                                           avg_factor,
                                                           NULL, NULL,
//                                                           running_mean,
//                                                           running_var,
                                                           epsilon,
                                                           NULL,
                                                           NULL) );
//    if (stage == NET_INFER) {
//        cudaFree(running_mean);
//        cudaFree(running_var);
//    }
//    } else if(stage == NET_INFER) {
//        checkCUDNN( cudnnBatchNormalizationForwardInference(
//                                                            *(cudnn_h),
//                                                            this->mode,
//                                                            &(this->one),
//                                                            &(this->zero),
//                                                            t_in->get_tensor_desc(),
//                                                            t_in->get_gpu_ptr(),
//                                                            t_out->get_tensor_desc(),
//                                                            t_out->get_gpu_ptr(),
//                                                            gamma->get_tensor_desc(),
//                                                            gamma->get_gpu_ptr(),
//                                                            beta->get_gpu_ptr(),
//                                                            resultRunningMean->get_gpu_ptr(),     //TO DO, this needs to be serialized as param but NOT update
//                                                            resultRunningVariance->get_gpu_ptr(), //TO DO, this needs to be serialized as param but NOT update
//                                                            epsilon ) );
//    }
    return std::vector<value_type>();
}

template <class value_type>
void batch_normalization_layer_t<value_type>::backward(network_stage stage, cublasHandle_t* cublas_h, cudnnHandle_t* cudnn_h, registry_t<value_type>* reg) {
    //gamma is the weight
    //beta  is the bias
    
    int input_l  = this->get_input_layer_id();
    int output_l = this->get_output_layer_id();
    int curt_l   = this->get_id();

    tensor_t<value_type>* t_in     = reg->get_reg_output(input_l, curt_l);
    t_in->increase_cur_use_counter(BACKWARD);
	// t_in->increase_use_count();
    tensor_t<value_type>* dEdD_n   = reg->get_reg_b_data(output_l, curt_l);
    dEdD_n->decrease_cur_b_use_count();
    tensor_t<value_type>* dEdD_c   = this->get_b_data();
    tensor_t<value_type>* dEdGa    = this->get_weight_grad();
    tensor_t<value_type>* dEdBe    = this->get_bias_grad();
    tensor_t<value_type>* gamma    = this->get_weight();
    
    assert( t_in    != NULL );
    assert( dEdD_n  != NULL );
    assert( dEdD_c  != NULL );
    assert( dEdGa != NULL );
    assert( dEdBe != NULL );
    assert( gamma != NULL );

#ifdef GRAD_ACC   

    checkCUDNN( cudnnBatchNormalizationBackward(
                                                 *(cudnn_h),
                                                 this->mode,
                                                 &(this->one),
                                                 &(this->zero),
                                                 &(this->one),
                                                 &(this->zero),
                                                 t_in->get_tensor_desc(),
                                                 t_in->get_gpu_ptr(),
                                                 dEdD_n->get_tensor_desc(),
                                                 dEdD_n->get_gpu_ptr(),
                                                 dEdD_c->get_tensor_desc(),
                                                 dEdD_c->get_gpu_ptr(),
                                                 gamma->get_tensor_desc(),
                                                 gamma->get_gpu_ptr(),
                                                 dEdGa->get_temp_gpu_ptr(),
                                                 dEdBe->get_temp_gpu_ptr(),
                                                 epsilon,
                                                 NULL,
                                                 NULL ) );

    size_t total_params;
    value_type one = 1.0;
    total_params = dEdGa->get_N() * dEdGa->get_C() * dEdGa->get_H() * dEdGa->get_W();
    checkCublasErrors(cublasSaxpy(*(cublas_h), total_params, (const float*)&one, (const float*)dEdGa->get_temp_gpu_ptr(), 1, (float*)dEdGa->get_gpu_ptr(), 1));
    total_params = dEdBe->get_N() * dEdBe->get_C() * dEdBe->get_H() * dEdBe->get_W();
    checkCublasErrors(cublasSaxpy(*(cublas_h), total_params, (const float*)&one, (const float*)dEdBe->get_temp_gpu_ptr(), 1, (float*)dEdBe->get_gpu_ptr(), 1));
#else

    checkCUDNN( cudnnBatchNormalizationBackward(
                                                 *(cudnn_h),
                                                 this->mode,
                                                 &(this->one),
                                                 &(this->zero),
                                                 &(this->one),
                                                 &(this->zero),
                                                 t_in->get_tensor_desc(),
                                                 t_in->get_gpu_ptr(),
                                                 dEdD_n->get_tensor_desc(),
                                                 dEdD_n->get_gpu_ptr(),
                                                 dEdD_c->get_tensor_desc(),
                                                 dEdD_c->get_gpu_ptr(),
                                                 gamma->get_tensor_desc(),
                                                 gamma->get_gpu_ptr(),
                                                 dEdGa->get_gpu_ptr(),
                                                 dEdBe->get_gpu_ptr(),
                                                 epsilon,
                                                 NULL,
                                                 NULL ) );
                                                
#endif    
}
    
template <class value_type>
void batch_normalization_layer_t<value_type>::fake_run(net_comp dir, registry_t<value_type>* reg) {
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
    }
}

INSTANTIATE_CLASS(batch_normalization_layer_t);

} //ATP namespace
