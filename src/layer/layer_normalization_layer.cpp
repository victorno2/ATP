#include <layer/layer_normalization_layer.h>
#include <util/mem_util.h>
#include <limits>
#include <cudnn.h>
namespace ATP {
    
template <class value_type>
void layer_normalization_layer_t<value_type>::forward_setup(registry_t<value_type>* reg, cudnnHandle_t* cudnn_h) {
    printf("======>setup the forward layer_normalization_layer_t:%d start\n", this->get_id());
    //hook the output of previous layer
    int input_l = this->get_input_layer_id();
    int curt_l  = this->get_id();
    tensor_t<value_type>* input = reg->get_reg_output(input_l, curt_l);
    input->increase_use_counter(FORWARD);
    assert( input != NULL );
	this->set_f_in(input, reg);
	input->increase_use_count_initial();
    input->increase_use_count();
    tensor_t<value_type>* f_out  = new tensor_t<value_type>( input->get_N(), input->get_C(), input->get_H(), input->get_W(), reg->get_vector(), DATA, this->get_id());
    //setup the output tensor
    
    this->set_f_out( f_out, reg );

    if (input->get_data_layout() == SEQ2SEQ_TNBV) {
        f_out->set_data_layout(SEQ2SEQ_TNBV);
    }
    
    // Create data point for RNN layer normaliazation
    this->seqLength = input->get_N();
    this->miniBatch = input->get_C();
    this->inputSize = input->get_H();
    x_seq = (value_type**)malloc(seqLength * sizeof(value_type*));
    y_seq = (value_type**)malloc(seqLength * sizeof(value_type*));
    xDesc = (cudnnTensorDescriptor_t*)malloc(this->seqLength * sizeof(cudnnTensorDescriptor_t));
    yDesc = (cudnnTensorDescriptor_t*)malloc(this->seqLength * sizeof(cudnnTensorDescriptor_t));
    gamma = new tensor_t<value_type>(this->seqLength, this->miniBatch, 1, 1, reg->get_vector(), PARAM, this->get_id());  // gamma is the weight
    beta  = new tensor_t<value_type>(this->seqLength, this->miniBatch, 1, 1, reg->get_vector(), PARAM, this->get_id());  // beta is the bias
    beta->init(new constant_initializer_t<value_type>(0));
    gamma->init(new constant_initializer_t<value_type>(1));
    this->set_weight( gamma, reg );
    this->set_bias( beta, reg );

            gamma_seq = (value_type**)malloc(this->seqLength * sizeof(value_type*));
        beta_seq = (value_type**)malloc(this->seqLength * sizeof(value_type*));
    // this->set_bias_prev( gamma_grad_prev, reg );
    // this->set_weight_prev( gamma_prev, reg );

    // this->resultRunningMean     = new tensor_t<value_type>(1, miniBatch, 1, 1, reg->get_vector(), BN_MEAN_VAR, this->get_id()); //TO DO
    // this->resultRunningVariance = new tensor_t<value_type>(1, miniBatch, 1, 1, reg->get_vector(), BN_MEAN_VAR, this->get_id()); //TO DO
    // this->resultSaveMean        = new tensor_t<value_type>(1, miniBatch, 1, 1, reg->get_vector(), BN_MEAN_VAR, this->get_id()); //TO DO
    // this->resultSaveInvVariance = new tensor_t<value_type>(1, miniBatch, 1, 1, reg->get_vector(), BN_MEAN_VAR, this->get_id()); //TO DO
    // resultRunningMean->init(new constant_initializer_t<value_type>(0));
    // resultRunningVariance->init(new constant_initializer_t<value_type>(0));
    // resultSaveMean->init(new constant_initializer_t<value_type>(0));
    // resultSaveInvVariance->init(new constant_initializer_t<value_type>(0));
    // printf("======>setup the resultSaveInvVariance->init(new constant_initializer_t<value_type>(0));:%d start\n", this->get_id());

    // resultRunningMean_seq = (value_type**)malloc(seqLength * sizeof(value_type*));
    // resultRunningVariance_seq = (value_type**)malloc(seqLength * sizeof(value_type*));
    // resultSaveMean_seq = (value_type**)malloc(seqLength * sizeof(value_type*));
    // resultSaveInvVariance_seq = (value_type**)malloc(seqLength * sizeof(value_type*));
    checkCUDNN(cudnnCreateTensorDescriptor(&bnScaleBiasMeanVarDesc));
    checkCUDNN(cudnnSetTensor4dDescriptor(bnScaleBiasMeanVarDesc, input->get_tensor_format(), mathPrec, 1, this->miniBatch, 1, 1) );
    // printf("======>setup the checkCUDNN(cudnnSetTensor4dDescriptor(bnScaleBiasMeanVarDesc, input->get_tensor_format(), mathPrec, 1, miniBatch, 1, 1) );:%d start\n", this->get_id());
    for (int i = 0; i < this->seqLength; i++) {
        checkCUDNN(cudnnCreateTensorDescriptor(&xDesc[i]));
        checkCUDNN(cudnnCreateTensorDescriptor(&yDesc[i]));
        checkCUDNN(cudnnSetTensor4dDescriptor(xDesc[i], input->get_tensor_format(), mathPrec, 1, this->miniBatch, this->inputSize, input->get_seq_vect()) );
        checkCUDNN(cudnnSetTensor4dDescriptor(yDesc[i], input->get_tensor_format(), mathPrec, 1, this->miniBatch, this->inputSize, input->get_seq_vect()) );
        // printf("resultSaveInvVariance_seq[%d] = resultSaveInvVariance->get_gpu_ptr_v2(i*miniBatch);\n", i);
    }

    assert(gamma != NULL);
    assert(beta  != NULL);

    //forward hookup check
    assert( this->get_f_in() != NULL );
    assert( this->get_bias() != NULL );
    assert( this->get_f_out() != NULL );
    assert( this->get_weight() != NULL );

    
    //register the forward dependency
    tensor_t<value_type>* t_in   = this->get_f_in();
    tensor_t<value_type>* t_out  = this->get_f_out();
    gamma  = this->get_weight();
    beta   = this->get_bias();
    
    reg->register_forward_dependency( this->get_id(), t_in  );
    reg->register_forward_dependency( this->get_id(), t_out );
    reg->register_forward_dependency( this->get_id(), gamma );
    reg->register_forward_dependency( this->get_id(), beta  );
    // reg->register_forward_dependency( this->get_id(), resultRunningMean  );
    // reg->register_forward_dependency( this->get_id(), resultRunningVariance  );
    // reg->register_forward_dependency( this->get_id(), resultSaveMean  );
    // reg->register_forward_dependency( this->get_id(), resultSaveInvVariance  );
    // printf("======>setup the forward layer_normalization_layer_t:%d done\n", this->get_id());
}
    
template <class value_type>
void layer_normalization_layer_t<value_type>::backward_setup(registry_t<value_type>* reg, cudnnHandle_t* cudnn_h) {
    printf("======>setup the backward batch normalization layer:%d\n", this->get_id());

    //setup the backward data
    int curt_l_id   = this->get_id();
    int input_l_id  = this->get_input_layer_id();
    int output_l_id = this->get_output_layer_id();

    tensor_t<value_type>* input = reg->get_reg_output(input_l_id, curt_l_id);
    input->increase_use_counter(BACKWARD);
    tensor_t<value_type>* b_data = new tensor_t<value_type>(input->get_N(), input->get_C(), input->get_H(), input->get_W(), reg->get_vector(), B_DATA, this->get_id());
    this->set_b_data(b_data, reg);

    if (input->get_data_layout() == SEQ2SEQ_TNBV) {
        b_data->set_data_layout(SEQ2SEQ_TNBV);
    }

    // tensor_t<value_type>* g  = reg->get_reg_weight(curt_l_id);
    // tensor_t<value_type>* b  = reg->get_reg_bias(curt_l_id);
    tensor_t<value_type>* g  = this->get_weight();
    tensor_t<value_type>* b  = this->get_bias();
    assert( g != NULL );
    assert( b != NULL );
    //TO DO: BN layer has two parameters to update, like weight(gamma) and bias(beta)
    gamma_grad = new tensor_t<value_type>( g->get_N(), g->get_C(), g->get_H(), g->get_W(), reg->get_vector(), GRAD, this->get_id());
    beta_grad = new tensor_t<value_type>( b->get_N(), b->get_C(), b->get_H(), b->get_W(), reg->get_vector(), GRAD, this->get_id());
    this->set_bias_grad( beta_grad, reg );
    this->set_weight_grad( gamma_grad, reg );
    
    dx_seq = (value_type**)malloc(this->seqLength * sizeof(value_type*));
        dy_seq = (value_type**)malloc(this->seqLength * sizeof(value_type*));
        gamma_grad_seq = (value_type**)malloc(this->seqLength * sizeof(value_type*));
        beta_grad_seq = (value_type**)malloc(this->seqLength * sizeof(value_type*));

    beta_grad_prev  = new tensor_t<value_type>(gamma->get_N(), gamma->get_C(), gamma->get_H(), gamma->get_W(), reg->get_vector(), GRAD, this->get_id());
    gamma_grad_prev = new tensor_t<value_type>(gamma->get_N(), gamma->get_C(), gamma->get_H(), gamma->get_W(), reg->get_vector(), GRAD, this->get_id());
    beta_grad_prev->init(new constant_initializer_t<value_type>(0));
    gamma_grad_prev->init(new constant_initializer_t<value_type>(0));
    this->set_bias_prev( beta_grad_prev, reg );
    this->set_weight_prev( gamma_grad_prev, reg );
    assert( this->get_bias_prev() != NULL );
    assert( this->get_weight_prev() != NULL );

    // Create data point for RNN layer normaliazation
    
    tensor_t<value_type>* t_in     = reg->get_reg_output(input_l_id,  curt_l_id);
    tensor_t<value_type>* dEdD_n   = reg->get_reg_b_data(output_l_id, curt_l_id);
    this->set_dy(dEdD_n);
    dEdD_n->increase_b_use_count();
    
    assert( this->get_weight_grad() != NULL );
    assert( this->get_bias_grad()   != NULL );
    assert( this->get_b_data()      != NULL );
    
    //register the backward dependency
    tensor_t<value_type>* dEdD_c   = this->get_b_data();
    gamma    = this->get_weight();
    gamma_grad                          = this->get_weight_grad();
    beta_grad                          = this->get_bias_grad();
    
    assert( t_in    != NULL );
    assert( dEdD_n  != NULL );
    assert( dEdD_c  != NULL );
    assert( dEdGa != NULL );
    assert( dEdBe != NULL );
    assert( gamma != NULL );
    
    reg->register_backward_dependency(this->get_id(), t_in    );
    reg->register_backward_dependency(this->get_id(), dEdD_n  );
    reg->register_backward_dependency(this->get_id(), dEdD_c  );
    reg->register_backward_dependency(this->get_id(), gamma_grad   );
    reg->register_backward_dependency(this->get_id(), beta_grad   );
    reg->register_backward_dependency(this->get_id(), gamma   );

}

template <class value_type>
std::vector<value_type> layer_normalization_layer_t<value_type>::forward(network_stage stage, cublasHandle_t* cublas_h, cudnnHandle_t* cudnn_h, registry_t<value_type>* reg) {
    //gamma is the weight
    //beta  is the bias
    // printf("forward layer%d layer_normalization_layer_t\n", this->get_id());
    int input_l = this->get_input_layer_id();
    int curt_l  = this->get_id();
    tensor_t<value_type>* t_in   = this->get_f_in();
	// t_in->decrease_use_count();
    t_in->increase_cur_use_counter(FORWARD);

    tensor_t<value_type>* t_out  = this->get_f_out();
    // t_out->printRNNTensor("t_out");
    tensor_t<value_type>* gamma  = this->get_weight();
    tensor_t<value_type>* beta   = this->get_bias();

    value_type* running_mean;
    value_type* running_var;

    this->iter += 1;
    double avg_factor = 1.0f / (this->iter);
    if (gamma_seq == NULL) {

    }
        for (int i = 0; i < this->seqLength; i++) {
            gamma_seq[i] = gamma->get_gpu_ptr_v2(i*(this->miniBatch));
            beta_seq[i] = beta->get_gpu_ptr_v2(i*(this->miniBatch));
            // resultRunningMean_seq[i]        = resultRunningMean->get_gpu_ptr_v2(i*miniBatch);
            // resultRunningVariance_seq[i]    = resultRunningVariance->get_gpu_ptr_v2(i*miniBatch);
            // resultSaveMean_seq[i]           = resultSaveMean->get_gpu_ptr_v2(i*miniBatch);
            // resultSaveInvVariance_seq[i]    = resultSaveInvVariance->get_gpu_ptr_v2(i*miniBatch);
            x_seq[i] = t_in->get_gpu_ptr_v2(i*(this->miniBatch)*(this->inputSize)*(t_in->get_seq_vect()));
            y_seq[i] = t_out->get_gpu_ptr_v2(i*(this->miniBatch)*(this->inputSize)*(t_in->get_seq_vect()));
        }
    

    for (size_t i = 0; i < this->seqLength; i++) {
        checkCUDNN( cudnnBatchNormalizationForwardTraining(
                                                           *(cudnn_h),
                                                           this->mode,
                                                           &(this->one),
                                                           &(this->zero),
                                                           xDesc[i],
                                                           x_seq[i],
                                                           yDesc[i],
                                                           y_seq[i],
                                                           bnScaleBiasMeanVarDesc,
                                                           gamma_seq[i],
                                                           beta_seq[i],
                                                           avg_factor,
                                                        //    resultRunningMean_seq[i],
                                                        //    resultRunningVariance_seq[i],
                                                           NULL, 
                                                           NULL,
                                                           epsilon,
                                                        //    resultSaveMean_seq[i],
                                                        //    resultSaveInvVariance_seq[i]
                                                           NULL,
                                                           NULL
                                                           ) );
    }
    // t_out->printTensorData("after cudnnBatchNormalizationForwardTraining tout", 2);
    return std::vector<value_type>();
}

template <class value_type>
void layer_normalization_layer_t<value_type>::backward(network_stage stage, cublasHandle_t* cublas_h, cudnnHandle_t* cudnn_h, registry_t<value_type>* reg) {
    //gamma is the weight
    //beta  is the bias
    
    int input_l  = this->get_input_layer_id();
    int output_l = this->get_output_layer_id();
    int curt_l   = this->get_id();

    tensor_t<value_type>* t_in     = reg->get_reg_output(input_l, curt_l);
    tensor_t<value_type>* x = this->get_f_in();
    t_in->increase_cur_use_counter(BACKWARD);
	// t_in->increase_use_count();
    tensor_t<value_type>* dEdD_n   = reg->get_reg_b_data(output_l, curt_l);
    dEdD_n->decrease_cur_b_use_count();
    tensor_t<value_type>* dEdD_c   = this->get_b_data();
    tensor_t<value_type>* dEdGa    = this->get_weight_grad();
    tensor_t<value_type>* dEdBe    = this->get_bias_grad();
    tensor_t<value_type>* gamma    = this->get_weight();
    
    // assert( t_in    != NULL );
    // assert( dEdD_n  != NULL );
    // assert( dEdD_c  != NULL );
    // assert( dEdGa != NULL );
    // assert( dEdBe != NULL );
    // assert( gamma != NULL );  
    // printf("start the backward layer normalization layer:%d\n", this->get_id());
    if (dx_seq == NULL) {
        
    }
        for (size_t i = 0; i < this->seqLength; i++) {
            dx_seq[i] = dEdD_c->get_gpu_ptr_v2(i*(this->miniBatch)*(this->inputSize)*(t_in->get_seq_vect()));
            dy_seq[i] = dEdD_n->get_gpu_ptr_v2(i*(this->miniBatch)*(this->inputSize)*(t_in->get_seq_vect()));
    #ifdef GRAD_ACC
            gamma_grad_seq[i] = gamma_grad->get_temp_gpu_ptr_v2(i*(this->miniBatch));
            beta_grad_seq[i] = beta_grad->get_temp_gpu_ptr_v2(i*(this->miniBatch));
    #else
            gamma_grad_seq[i] = gamma_grad->get_gpu_ptr_v2(i*(this->miniBatch));
            beta_grad_seq[i] = beta_grad->get_gpu_ptr_v2(i*(this->miniBatch));
    #endif
        }
    

    for (size_t i = 0; i < seqLength; i++) {
        checkCUDNN( cudnnBatchNormalizationBackward(
                                                *(cudnn_h),
                                                this->mode,
                                                &(this->one),
                                                &(this->zero),
                                                &(this->one),
                                                &(this->zero),
                                                xDesc[i],
                                                x_seq[i],
                                                yDesc[i],
                                                dy_seq[i],
                                                xDesc[i],
                                                dx_seq[i],
                                                bnScaleBiasMeanVarDesc,
                                                gamma_seq[i],
                                                gamma_grad_seq[i],
                                                beta_grad_seq[i],
                                                epsilon,
                                                NULL,
                                                NULL
                                                // resultSaveMean_seq[i],
                                                // resultSaveInvVariance_seq[i] 
                                                ) );
    }

#ifdef GRAD_ACC
    size_t total_params;
    value_type one = 1.0;
    total_params = gamma_grad->get_N() * gamma_grad->get_C() * gamma_grad->get_H() * gamma_grad->get_W();
    checkCublasErrors(cublasSaxpy(*(cublas_h), total_params, (const float*)&one, (const float*)gamma_grad->get_temp_gpu_ptr(), 1, (float*)gamma_grad->get_gpu_ptr(), 1));
    total_params = beta_grad->get_N() * beta_grad->get_C() * beta_grad->get_H() * beta_grad->get_W();
    checkCublasErrors(cublasSaxpy(*(cublas_h), total_params, (const float*)&one, (const float*)beta_grad->get_temp_gpu_ptr(), 1, (float*)beta_grad->get_gpu_ptr(), 1));
#endif   
    // printf("end the backward layer normalization layer:%d\n", this->get_id());
}
    
template <class value_type>
void layer_normalization_layer_t<value_type>::fake_run(net_comp dir, registry_t<value_type>* reg) {
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
    
INSTANTIATE_CLASS(layer_normalization_layer_t);

} //ATP namespace
