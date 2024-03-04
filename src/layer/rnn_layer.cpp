#include <layer/rnn_layer.h>
#include <util/mem_util.h>
#include <limits>
#include <cudnn.h>
// #define CONV_DEBUG


namespace ATP {
    
template <class value_type>
void rnn_layer_t<value_type>::forward_setup( registry_t<value_type>* reg, cudnnHandle_t* cudnn_h ) {
    printf("======>setup the forward rnn layer:%d\n", this->get_id());
    int curt_l  = this->get_id();
    int input_l = this->get_input_layer_id();
    tensor_t<value_type>* x = reg->get_reg_output(input_l, curt_l);
	
    assert( x != NULL );
    // x->increase_use_count_initial();
	// x->increase_use_count();
    x->increase_use_counter(FORWARD);
	this->set_f_in(x, reg);

    xDesc = (cudnnTensorDescriptor_t*)malloc(seqLength * sizeof(cudnnTensorDescriptor_t));
    yDesc = (cudnnTensorDescriptor_t*)malloc(seqLength * sizeof(cudnnTensorDescriptor_t));
    int dimA[3];
    int strideA[3];
    miniBatch = x->get_C();
    // inputSize = x->get_C();
    for (int i = 0; i < seqLength; i++) {
        dimA[0] = miniBatch;
        dimA[1] = inputSize;
        dimA[2] = 1;
        strideA[0] = dimA[2] * dimA[1];
        strideA[1] = dimA[2];
        strideA[2] = 1;
        checkCUDNN(cudnnCreateTensorDescriptor(&xDesc[i]));
        checkCUDNN(cudnnSetTensorNdDescriptor(xDesc[i], mathPrec, 3, dimA, strideA));
        
        dimA[0] = miniBatch;
        dimA[1] = hiddenSize;
        dimA[2] = 1;
        strideA[0] = dimA[2] * dimA[1];
        strideA[1] = dimA[2];
        strideA[2] = 1; 
        checkCUDNN(cudnnCreateTensorDescriptor(&yDesc[i]));
        checkCUDNN(cudnnSetTensorNdDescriptor(yDesc[i], mathPrec, 3, dimA, strideA));
    }
    tensor_t<value_type>* y = new tensor_t<value_type>(seqLength, miniBatch, hiddenSize, 1, reg->get_vector(), DATA, this->get_id());
    printf("y_size_bytes = %d\n", y->get_mem_size());
    this->set_f_out( y, reg );
    assert( this->get_f_out()  != NULL );
    //register the forward dependency
    x = reg->get_reg_output(input_l, curt_l);
    y = this->get_f_out();
    

    dimA[0] = numLayers;
    dimA[1] = miniBatch;
    dimA[2] = hiddenSize;
    strideA[0] = dimA[2] * dimA[1];
    strideA[1] = dimA[2];
    strideA[2] = 1;

    checkCUDNN(cudnnCreateTensorDescriptor(&hxDesc));
    checkCUDNN(cudnnCreateTensorDescriptor(&cxDesc));
    checkCUDNN(cudnnCreateTensorDescriptor(&hyDesc));
    checkCUDNN(cudnnCreateTensorDescriptor(&cyDesc));

    checkCUDNN(cudnnSetTensorNdDescriptor(hxDesc, mathPrec, 3, dimA, strideA));
    this->hx = new tensor_t<value_type>(numLayers, miniBatch, hiddenSize, 1, reg->get_vector(), DATA, this->get_id());
    printf("hx_size_bytes = %d\n", hx->get_mem_size());

    checkCUDNN(cudnnSetTensorNdDescriptor(cxDesc, mathPrec, 3, dimA, strideA));
    this->cx = new tensor_t<value_type>(numLayers, miniBatch, hiddenSize, 1, reg->get_vector(), DATA, this->get_id()); // for lstm
    printf("cx_size_bytes = %d\n", cx->get_mem_size());

    checkCUDNN(cudnnSetTensorNdDescriptor(hyDesc, mathPrec, 3, dimA, strideA));
    // this->hy = new tensor_t<value_type>(numLayers,miniBatch, hiddenSize, 1, reg->get_vector(), DATA, this->get_id());

    checkCUDNN(cudnnSetTensorNdDescriptor(cyDesc, mathPrec, 3, dimA, strideA));
    // this->cy = new tensor_t<value_type>(numLayers, miniBatch, hiddenSize, 1, reg->get_vector(), DATA, this->get_id()); // for lstm

    checkCUDNN(cudnnDropoutGetStatesSize(*cudnn_h, &(this->dropout_state_size_bytes)));
    printf("dropout_state_size_bytes = %d\n", (this->dropout_state_size_bytes));
    size_t stateSize = dropout_state_size_bytes / sizeof(value_type);
    this->dropout_state = new tensor_t<value_type>( stateSize, 1, 1, 1, reg->get_vector(), RNN_BUFF, this->get_id());
    unsigned long seed = (unsigned long) rand();
    checkCUDNN( cudnnSetDropoutDescriptor(dropoutDesc, *cudnn_h, dropout_rate, (this->dropout_state)->get_gpu_ptr(), this->dropout_state_size_bytes, seed));

    checkCUDNN(cudnnSetRNNDescriptor(*cudnn_h, rnnDesc, hiddenSize, numLayers, dropoutDesc, inputMode, direction, mode, algo, mathPrec));
    checkCUDNN(cudnnSetRNNBiasMode(this->rnnDesc, biasMode));

    checkCUDNN(cudnnCreateFilterDescriptor(&wDesc));
    checkCUDNN(cudnnGetRNNParamsSize(*cudnn_h, rnnDesc, xDesc[0], &weightsSize, mathPrec));
    int dimW[3];   
    dimW[0] = weightsSize / sizeof(value_type);
    dimW[1] = 1;
    dimW[2] = 1;
    checkCUDNN(cudnnSetFilterNdDescriptor(wDesc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, 3, dimW));
    w = new tensor_t<value_type>(dimW[0], dimW[1], dimW[2], 1, reg->get_vector(), PARAM, this->get_id());
    printf("w_size = %d\n", w->get_mem_size());
    this->set_weight(w, reg);
    w->init(this->weight_initializer);
    
    checkCUDNN(cudnnGetRNNWorkspaceSize(*cudnn_h, this->rnnDesc, this->seqLength, xDesc, &(this->f_rnn_buff_size)));
    size_t buff_W = (this->f_rnn_buff_size) / sizeof(value_type);
    this->f_rnn_buff = new tensor_t<value_type>(1, 1, 1, buff_W, reg->get_vector(), RNN_BUFF, this->get_id());
    reg->register_forward_dependency( this->get_id(), this->f_rnn_buff );
    printf("f_rnn_buff_size = %d\n", (this->f_rnn_buff_size));

    checkCUDNN(cudnnGetRNNTrainingReserveSize(*cudnn_h, rnnDesc, seqLength, xDesc, &reserveSpaceSizeInBytes));
    size_t reserveSpace = reserveSpaceSizeInBytes / sizeof(value_type);
    this->reserve_buff = new tensor_t<value_type>(1, 1, 1, this->reserveSpaceSizeInBytes, reg->get_vector(), RNN_RESERVE, this->get_id());
    this->set_reserve_buff(reserve_buff);

    // Weights
    int numLinearLayers = 0;
    if (this->mode == CUDNN_RNN_RELU || this->mode == CUDNN_RNN_TANH) {
        numLinearLayers = 2;
    }
    else if (this->mode == CUDNN_LSTM) {
        numLinearLayers = 8;
    }
    else if (this->mode == CUDNN_GRU) {
        numLinearLayers = 6;
    }

    reg->register_forward_dependency( this->get_id(), x );
    reg->register_forward_dependency( this->get_id(), w );
    reg->register_forward_dependency( this->get_id(), y );
    reg->register_forward_dependency( this->get_id(), cx );
    reg->register_forward_dependency( this->get_id(), hx );
    reg->register_backward_dependency( this->get_id(), f_rnn_buff );
    reg->register_forward_dependency( this->get_id(), this->reserve_buff );
    reg->register_forward_dependency( this->get_id(), this->dropout_state );
    // reg->register_forward_dependency( this->get_id(), cy );
    // reg->register_forward_dependency( this->get_id(), hy );
}
    
template <class value_type>
void rnn_layer_t<value_type>::backward_setup( registry_t<value_type>* reg, cudnnHandle_t* cudnn_h ) {
    //backward
    int curt_l_id   = this->get_id();
    int input_l_id  = this->get_input_layer_id();
    int output_l_id = this->get_output_layer_id();
    tensor_t<value_type>* dy = reg->get_reg_b_data(output_l_id, curt_l_id);
    this->set_dy(dy);
    dy->increase_b_use_count();
    printf("======>setup the backward rnn layer:%d\n", this->get_id());
    //register the backward dependency
    int dimA[3];
    int strideA[3];
    tensor_t<value_type> *x = this->get_f_in();
    x->increase_use_counter(BACKWARD);

    tensor_t<value_type> *y = this->get_f_out();
    y->increase_use_counter(BACKWARD);
    // miniBatch = x->get_N();
    dxDesc = (cudnnTensorDescriptor_t*)malloc(seqLength * sizeof(cudnnTensorDescriptor_t));
    dyDesc = (cudnnTensorDescriptor_t*)malloc(seqLength * sizeof(cudnnTensorDescriptor_t));
    for (int i = 0; i < seqLength; i++) {
        checkCUDNN(cudnnCreateTensorDescriptor(&dxDesc[i]));
        checkCUDNN(cudnnCreateTensorDescriptor(&dyDesc[i]));
    
        dimA[0] = miniBatch;
        dimA[1] = inputSize;
        dimA[2] = 1;
        strideA[0] = dimA[2] * dimA[1];
        strideA[1] = dimA[2];
        strideA[2] = 1;
        checkCUDNN(cudnnSetTensorNdDescriptor(dxDesc[i], mathPrec, 3, dimA, strideA));
        
        dimA[0] = miniBatch;
        dimA[1] = hiddenSize;
        dimA[2] = 1;
        strideA[0] = dimA[2] * dimA[1];
        strideA[1] = dimA[2];
        strideA[2] = 1;
        checkCUDNN(cudnnSetTensorNdDescriptor(dyDesc[i], mathPrec, 3, dimA, strideA));
    }
    tensor_t<value_type>* dx = new tensor_t<value_type>(miniBatch, seqLength, inputSize, 1, reg->get_vector(), B_DATA, this->get_id());
    this->set_b_data(dx, reg);
    assert( this->get_b_data() != NULL );
    printf("dx_size = %zd, dy_size = %zd\n", seqLength*miniBatch*hiddenSize, seqLength*miniBatch*hiddenSize);

    dimA[0] = numLayers;
    dimA[1] = miniBatch;
    dimA[2] = hiddenSize;
    strideA[0] = dimA[2] * dimA[1];
    strideA[1] = dimA[2];
    strideA[2] = 1;

    checkCUDNN(cudnnCreateTensorDescriptor(&dhxDesc));
    checkCUDNN(cudnnCreateTensorDescriptor(&dcxDesc));
    checkCUDNN(cudnnCreateTensorDescriptor(&dhyDesc));
    checkCUDNN(cudnnCreateTensorDescriptor(&dcyDesc));

    checkCUDNN(cudnnSetTensorNdDescriptor(dhxDesc, mathPrec, 3, dimA, strideA));
    // this->dhx = new tensor_t<value_type>(numLayers, miniBatch, hiddenSize, 1, reg->get_vector(), DATA, this->get_id());
    // dhx->init_tensor_data(1.f, true);

    checkCUDNN(cudnnSetTensorNdDescriptor(dcxDesc, mathPrec, 3, dimA, strideA));
    // this->dcx = new tensor_t<value_type>(numLayers, miniBatch, hiddenSize, 1, reg->get_vector(), DATA, this->get_id()); // for lstm
    // dcx->init_tensor_data(1.f, true);

    checkCUDNN(cudnnSetTensorNdDescriptor(dhyDesc, mathPrec, 3, dimA, strideA));
    // this->dhy = new tensor_t<value_type>(numLayers,miniBatch, hiddenSize, 1, reg->get_vector(), DATA, this->get_id());
    // dhy->init_tensor_data(1.f, true);

    checkCUDNN(cudnnSetTensorNdDescriptor(dcyDesc, mathPrec, 3, dimA, strideA));
    // this->dcy = new tensor_t<value_type>(numLayers, miniBatch, hiddenSize, 1, reg->get_vector(), DATA, this->get_id()); // for lstm
    // dcy->init_tensor_data(1.f, true);
   
    int dimW[3];   
    dimW[0] = weightsSize / sizeof(value_type);
    dimW[1] = 1;
    dimW[2] = 1;
    checkCUDNN(cudnnCreateFilterDescriptor(&dwDesc));  
    checkCUDNN(cudnnSetFilterNdDescriptor(dwDesc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, 3, dimW));     
    // dw = new tensor_t<value_type>(dimW[0], dimW[1], dimW[2], 1, reg->get_vector(), PARAM, this->get_id());
    dw = new tensor_t<value_type>(dimW[0], dimW[1], dimW[2], 1, reg->get_vector(), GRAD, this->get_id());
    // dw_prev = new tensor_t<value_type>(dimW[0], dimW[1], dimW[2], 1, reg->get_vector(), PARAM, this->get_id());
    this->set_weight_grad(dw, reg);
    // this->set_weight_prev(dw_prev, reg);
    dw->init(this->weight_initializer);
    // dw_prev->init(this->weight_initializer);
    printf("cudnnSetFilterNdDescriptor dw number = %d\n", weightsSize / sizeof(value_type));


    // reg->register_backward_dependency( this->get_id(), dw );
    // reg->register_backward_dependency( this->get_id(), dcx );
    // reg->register_backward_dependency( this->get_id(), dhx );
    // reg->register_backward_dependency( this->get_id(), dcy );
    // reg->register_backward_dependency( this->get_id(), dhy );

    reg->register_backward_dependency( this->get_id(), dx );
    reg->register_backward_dependency( this->get_id(), dy );
    reg->register_backward_dependency( this->get_id(), dw );   
    reg->register_backward_dependency( this->get_id(), x );   
    reg->register_backward_dependency( this->get_id(), w );
    reg->register_backward_dependency( this->get_id(), y );
    reg->register_backward_dependency( this->get_id(), hx );
    reg->register_backward_dependency( this->get_id(), cx );
    reg->register_backward_dependency( this->get_id(), f_rnn_buff );
    reg->register_forward_dependency( this->get_id(), this->reserve_buff );
    reg->register_forward_dependency( this->get_id(), this->dropout_state );
}


template <class value_type>
std::vector<value_type> rnn_layer_t<value_type>::forward(network_stage stage, cublasHandle_t* cublas_h, cudnnHandle_t* cudnn_h, registry_t<value_type>* reg) {
    assert( cudnn_h != NULL );
    assert( reg     != NULL );
    
    int input_l = this->get_input_layer_id();
    int curt_l  = this->get_id();
    tensor_t<value_type>* x  = reg->get_reg_output(input_l, curt_l);
	// x->decrease_use_count();
    x->increase_cur_use_counter(FORWARD);

    tensor_t<value_type>* y = this->get_f_out();
    tensor_t<value_type>* w = this->get_weight();

#ifdef DEBUG
    printf("input tensor from %d to %d\n", input_l, curt_l);
#endif

    checkCUDNN( cudnnRNNForwardTraining(
    /* cudnnHandle_t */                   *cudnn_h,
    /* const cudnnRNNDescriptor_t */      this->rnnDesc,
    /* const int */                       this->seqLength,
    /* const cudnnTensorDescriptor_t* */  xDesc,
    /* const void */                      x->get_gpu_ptr(),
    /* const cudnnTensorDescriptor_t */   hxDesc,
    /* const void */                      hx->get_gpu_ptr(),
    /* const cudnnTensorDescriptor_t */   cxDesc,
    /* const void */                      cx->get_gpu_ptr(),
    /* const cudnnFilterDescriptor_t */   this->wDesc,
    /* const void */                      w->get_gpu_ptr(),
    /* const cudnnTensorDescriptor_t* */  yDesc,
    /* void */                            y->get_gpu_ptr(),
    /* const cudnnTensorDescriptor_t */   hyDesc,
    /* void */                            NULL, // hy->get_gpu_ptr(),// 
    /* const cudnnTensorDescriptor_t */   cyDesc,
    /* void */                            NULL, // cy->get_gpu_ptr(),// 
    /* void */                            f_rnn_buff->get_gpu_ptr(),
    /* size_t */                          this->f_rnn_buff_size,
    /* void */                            reserve_buff->get_gpu_ptr(),
    /* size_t */                          this->reserveSpaceSizeInBytes)
    );

#ifdef DEBUG
    this->get_f_out()->printTensor("activation, forward output");
//    this->get_f_out()->GPUtoCPU();
#endif
    return std::vector<value_type>();
}
    
template <class value_type>
void rnn_layer_t<value_type>::backward(network_stage stage, cublasHandle_t* cublas_h, cudnnHandle_t* cudnn_h, registry_t<value_type>* reg ) {
    assert( cudnn_h != NULL );
    assert( reg     != NULL );
    
    int curt_l_id   = this->get_id();
    int input_l_id  = this->get_input_layer_id();
    int output_l_id = this->get_output_layer_id();
    
    tensor_t<value_type>* dx = this->get_b_data();
    tensor_t<value_type>* y = this->get_f_out();
    tensor_t<value_type>* x  = reg->get_reg_output(input_l_id, curt_l_id);
	// x->increase_use_count();
    x->increase_cur_use_counter(BACKWARD);
    y->increase_cur_use_counter(BACKWARD);
    tensor_t<value_type>* dy  = reg->get_reg_b_data(output_l_id, curt_l_id);
    dy->decrease_cur_b_use_count();
    // dy->set_rnn_dim(dy->get_N(), dy->get_C(), dy->get_H()*dy->get_W());

    checkCUDNN(cudnnRNNBackwardData(
                                *cudnn_h, 
                                this->rnnDesc, 
                                this->seqLength,                                
                                yDesc, 
                                y->get_gpu_ptr(),
                                dyDesc, 
                                dy->get_gpu_ptr(), 
                                dhyDesc, 
                                NULL, // dhy->get_gpu_ptr(), // 
                                dcyDesc, 
                                NULL, // dcy->get_gpu_ptr(), // 
                                wDesc, 
                                w->get_gpu_ptr(), 
                                hxDesc, 
                                hx->get_gpu_ptr(),
                                cxDesc, 
                                cx->get_gpu_ptr(),
                                dxDesc, 
                                dx->get_gpu_ptr(), 
                                dhxDesc,
                                NULL, // dhx->get_gpu_ptr(), // NULL, // 
                                dcxDesc,
                                NULL, // dcx->get_gpu_ptr(), // NULL, // 
                                f_rnn_buff->get_gpu_ptr(),
                                f_rnn_buff_size,
                                reserve_buff->get_gpu_ptr(), 
                                reserveSpaceSizeInBytes )
    );

#ifdef GRAD_ACC
    checkCUDNN(cudnnRNNBackwardWeights( 
                                *cudnn_h, 
                                this->rnnDesc, 
                                this->seqLength,  
                                xDesc, 
                                x->get_gpu_ptr(), 
                                hxDesc, 
                                hx->get_gpu_ptr(),                                                   
                                yDesc, 
                                y->get_gpu_ptr(),
                                f_rnn_buff->get_gpu_ptr(), 
                                f_rnn_buff_size, 
                                dwDesc, 
                                dw->get_temp_gpu_ptr(),
                                reserve_buff->get_gpu_ptr(), 
                                reserveSpaceSizeInBytes )
    );
    value_type one = 1.0;
    size_t total_params = dw->get_N() * dw->get_C() * dw->get_H() * dw->get_W();
    checkCublasErrors(cublasSaxpy(*(cublas_h), total_params, (const float*)&one, (const float*)dw->get_temp_gpu_ptr(), 1, (float*)dw->get_gpu_ptr(), 1));
#else
    checkCUDNN(cudnnRNNBackwardWeights( 
                                *cudnn_h, 
                                this->rnnDesc, 
                                this->seqLength,  
                                xDesc, 
                                x->get_gpu_ptr(), 
                                hxDesc, 
                                hx->get_gpu_ptr(),                                                   
                                yDesc, 
                                y->get_gpu_ptr(),
                                f_rnn_buff->get_gpu_ptr(), 
                                f_rnn_buff_size, 
                                dwDesc, 
                                dw->get_gpu_ptr(),
                                reserve_buff->get_gpu_ptr(), 
                                reserveSpaceSizeInBytes )
    );

#endif

    this->reserve_buff->set_data_state(FORWARD_DELETE_OK);
    this->reserve_buff->set_data_position(NO_DATA);

#ifdef DEBUG
    dEdD->printTensor("Result of Backward Activation");
//    dEdD->GPUtoCPU();
#endif
}

template <class value_type>
void rnn_layer_t<value_type>::fake_run(net_comp dir, registry_t<value_type>* reg) {
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
    
INSTANTIATE_CLASS(rnn_layer_t);
    
} //ATP namespace