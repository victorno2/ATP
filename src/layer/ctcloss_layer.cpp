
#include <layer/ctcloss_layer.h>
#include <util/mem_util.h>
#include <limits>
#include <cudnn.h>

namespace ATP {

template <class value_type>
void ctcloss_layer_t<value_type>::forward_setup( registry_t<value_type>* reg, cudnnHandle_t* cudnn_h ) {
    
    printf("======>setup the forward ctcloss layer:%d\n", this->get_id());
    //in-place operations, no need to create tensors
    //forward
    int input_l = this->get_input_layer_id();
    int curt_l  = this->get_id();
    tensor_t<value_type>* t_in = reg->get_reg_output(input_l, curt_l);
	this->set_f_in(t_in, reg);
    t_in->increase_use_counter(FORWARD);
	// t_in->increase_use_count_initial();
	// t_in->increase_use_count();
    assert(t_in != NULL);
    gradients = new tensor_t<value_type>(t_in->get_N(), t_in->get_C(), t_in->get_H(), t_in->get_W(), reg->get_vector(), DATA, this->get_id());
    
    //register the forward dependency
    t_in                               = reg->get_reg_output(input_l, curt_l);
    this->set_f_in(t_in, reg);
//    tensor_t<value_type>* t_out        = this->get_f_out();
    tensor_t<value_type>* label_train  = reg->get_train_label();
    // tensor_t<value_type>* label_test   = reg->get_test_label();
    
    assert( t_in        != NULL );
    assert( t_out       != NULL );
    assert( label_train != NULL );

    seqLength = t_in->get_N();
    miniBatch = t_in->get_C();
    inputSize = t_in->get_H();
    labelLength = label_train->get_H();

    checkCUDNN(cudnnCreateTensorDescriptor(&xDesc));
    // checkCUDNN(cudnnSetTensor4dDescriptor(xDesc, CUDNN_TENSOR_NCHW, mathPrec, seqLength, miniBatch, inputSize, 1));
    checkCUDNN(cudnnSetTensor4dDescriptor(xDesc, CUDNN_TENSOR_NCHW, mathPrec, seqLength, miniBatch*inputSize, 1, 1));
    checkCUDNN(cudnnCreateTensorDescriptor(&yDesc));
    // checkCUDNN(cudnnSetTensor4dDescriptor(yDesc, CUDNN_TENSOR_NCHW, mathPrec, seqLength, miniBatch, inputSize, 1));
    checkCUDNN(cudnnSetTensor4dDescriptor(yDesc, CUDNN_TENSOR_NCHW, mathPrec, seqLength, miniBatch*inputSize, 1, 1));

    int dim = 3;
    int dims[dim] {seqLength, miniBatch, labelLength};
    int strides[dim] {labelLength*miniBatch, labelLength, 1};
    checkCUDNN(cudnnCreateTensorDescriptor(&probsDesc));
    checkCUDNN(cudnnSetTensorNdDescriptor(probsDesc, mathPrec, dim, dims, strides));
    probs = new tensor_t<value_type>(seqLength, miniBatch, inputSize, 1, reg->get_vector(), DATA, this->get_id());
    

    checkCUDNN(cudnnCreateTensorDescriptor(&gradientsDesc));
    checkCUDNN(cudnnSetTensorNdDescriptor(gradientsDesc, mathPrec, dim, dims, strides));

    checkCUDNN(cudnnCreateTensorDescriptor(&sub_probsDesc));
    checkCUDNN(cudnnSetTensor4dDescriptor(sub_probsDesc, CUDNN_TENSOR_NCHW, mathPrec, miniBatch, inputSize, 1, 1));
    checkCUDNN(cudnnCreateTensorDescriptor(&sub_gradientsDesc));
    checkCUDNN(cudnnSetTensor4dDescriptor(sub_gradientsDesc, CUDNN_TENSOR_NCHW, mathPrec, miniBatch, inputSize, 1, 1));
    checkCUDNN(cudnnCreateTensorDescriptor(&sub_inputsDesc));
    checkCUDNN(cudnnSetTensor4dDescriptor(sub_inputsDesc, CUDNN_TENSOR_NCHW, mathPrec, miniBatch, inputSize, 1, 1));

    checkCUDNN(cudnnCreateCTCLossDescriptor(&ctcLossDesc));
    checkCUDNN(cudnnSetCTCLossDescriptor(ctcLossDesc, mathPrec));
    // checkCUDNN(cudnnSetCTCLossDescriptorEx(ctcLossDesc, mathPrec, CUDNN_LOSS_NORMALIZATION_SOFTMAX, CUDNN_PROPAGATE_NAN));

    costs = new tensor_t<value_type>(1, miniBatch, 1, 1, reg->get_vector(), DATA, this->get_id());
    // #ifdef TRAINING_CONFIGURATION_SEARCH
        costs->acquireSpaceCPU(costs->get_mem_size());
    // #endif
    this->set_f_out( costs, reg ); //use the inplace operation.
    assert( this->get_f_out()  != NULL );

    printf("init cudnnSetCTCLossDescriptor\n", 666);

    labelLengths = (int*)malloc(miniBatch*sizeof(int));
    // checkCudaErrors(cudaMallocManaged(&labelLengths, sizeof(int)*miniBatch));
    for (int i = 0; i < miniBatch; i++) {
        this->labelLengths[i] = labelLength;
    }

    labels = label_train->get_gpu_ptr_int();

    inputLengths = (int*)malloc(miniBatch*sizeof(int));
    // checkCudaErrors(cudaMallocManaged(&inputLengths, sizeof(int)*miniBatch));
    for (int i = 0; i < miniBatch; i++) {
        this->inputLengths[i] = seqLength;
        // printf("inputLengths[%d] = %d, ", i, inputLengths[i]);
    }
    printf("\n");
    printf("init labelLengths and inputLengths\n", 666);

    checkCUDNN(cudnnGetCTCLossWorkspaceSize(
                    *cudnn_h, probsDesc, gradientsDesc, 
                    labels, labelLengths, inputLengths,
                    algo,
                    ctcLossDesc, &workSpaceSizeInBytes));
    workspace = new tensor_t<value_type>(1, 1, 1, workSpaceSizeInBytes/sizeof(value_type)+8, reg->get_vector(), CONV_BUFF, this->get_id());
    checkCudaErrors(cudaMalloc(&workspace_void, workSpaceSizeInBytes+64));

    reg->register_forward_dependency( this->get_id(), t_in        );
    reg->register_forward_dependency( this->get_id(), gradients   );
    reg->register_forward_dependency( this->get_id(), label_train );
    reg->register_forward_dependency( this->get_id(), costs );
    reg->register_forward_dependency( this->get_id(), probs );
}
    
template <class value_type>
void ctcloss_layer_t<value_type>::backward_setup( registry_t<value_type>* reg, cudnnHandle_t* cudnn_h ) {
    printf("======>setup the backward ctcloss layer:%d\n", this->get_id());
    //in-place operations, no need to create tensors
    //backward hookup
    // this->set_b_data( this->get_f_out(), reg );
    //hookup checks
    // assert( this->get_b_data() != NULL );
    // this->get_b_data()->
    // int curt_l_id   = this->get_id();
    // int input_l_id  = this->get_input_layer_id();
    // int output_l_id = this->get_output_layer_id();
    // tensor_t<value_type>* dy = reg->get_reg_b_data(output_l_id, curt_l_id);
    // dy->increase_b_use_count();

    tensor_t<value_type>* t_out        = this->get_f_out();
    tensor_t<value_type>* label_train  = reg->get_train_label();
    tensor_t<value_type>* t_in         = reg->get_reg_output(this->get_input_layer_id(), this->get_id());
    t_in->increase_use_counter(BACKWARD);
    assert( t_out != NULL );
    assert( label_train != NULL );

    tensor_t<value_type>* dx = new tensor_t<value_type>(t_in->get_N(), t_in->get_C(), t_in->get_H(), t_in->get_W(), reg->get_vector(), B_DATA, this->get_id());
    this->set_b_data(dx, reg);
    assert( this->get_b_data() != NULL );
    //register the backward dependency
    reg->register_backward_dependency(this->get_id(), dx);
    reg->register_backward_dependency(this->get_id(), label_train  );
    reg->register_backward_dependency(this->get_id(), t_in);    // we should add the backward dependency to avoid freeing in recompute part
    reg->register_backward_dependency(this->get_id(), probs); 
    reg->register_backward_dependency(this->get_id(), costs); 
    reg->register_backward_dependency(this->get_id(), gradients); 

    checkCUDNN(cudnnCreateTensorDescriptor(&sub_dxDesc));
    checkCUDNN(cudnnSetTensor4dDescriptor(sub_dxDesc, CUDNN_TENSOR_NCHW, mathPrec, miniBatch, inputSize, 1, 1));

}

    
template <class value_type>
std::vector<value_type> ctcloss_layer_t<value_type>::forward(network_stage stage, cublasHandle_t* cublas_h, cudnnHandle_t* cudnn_h, registry_t<value_type>* reg ) {
    
    assert( cudnn_h != NULL );
    assert( reg     != NULL );
    
    int input_l = this->get_input_layer_id();
    int curt_l  = this->get_id();
    tensor_t<value_type>* input = reg->get_reg_output(input_l, curt_l);
    input->increase_cur_use_counter(FORWARD);
	// input->decrease_use_count();

    tensor_t<value_type> *label_train = reg->get_train_label();
    // tensor_t<value_type> *gradients = this->get_b_data();
    // label_train->printTensorInt("label_train");
    // probs->printRNNTensor("probs");
    // input->printRNNTensor("input");

    if (sub_inputs == NULL) {
        sub_inputs = (value_type**)malloc(seqLength * sizeof(value_type*));
        sub_probs = (value_type**)malloc(seqLength * sizeof(value_type*));
        sub_gradients = (value_type**)malloc(seqLength * sizeof(value_type*));
    }
        // size_t offset = 0;
    for (int i = 0; i < seqLength; i++) {
        sub_inputs[i] = input->get_gpu_ptr_v2(i*miniBatch*inputSize);
        sub_probs[i] = probs->get_gpu_ptr_v2(i*miniBatch*inputSize);
        sub_gradients[i] = gradients->get_gpu_ptr_v2(i*miniBatch*inputSize);
            // offset += miniBatch*inputSize;
    }
    
    for (int i = 0; i < seqLength; i++) {
        checkCUDNN( 
            cudnnSoftmaxForward(
               *cudnn_h,    
                CUDNN_SOFTMAX_ACCURATE,// CUDNN_SOFTMAX_FAST, // CUDNN_SOFTMAX_ACCURATE,
                CUDNN_SOFTMAX_MODE_INSTANCE,// CUDNN_SOFTMAX_MODE_CHANNEL,
               &alpha,
                sub_inputsDesc,
                sub_inputs[i],
               &beta,
                sub_probsDesc,
                sub_probs[i]
            ) 
        );
        // printf("cudnnSoftmaxForward %d\n", i);
    }

    checkCUDNN(cudnnCTCLoss(
                *cudnn_h,   
                probsDesc,
                probs->get_gpu_ptr(),
                label_train->get_cpu_ptr_int(),// labels,
                labelLengths,
                inputLengths,
                costs->get_gpu_ptr(),
                gradientsDesc,
                gradients->get_gpu_ptr(),
                CUDNN_CTC_LOSS_ALGO_DETERMINISTIC, // algo, // ,
                ctcLossDesc,
                workspace_void, //                (void*)workspace->get_gpu_ptr(),
                workSpaceSizeInBytes));

    this->get_loss(this->loss_mode, &(this->loss));
    return loss;
}
    
template <class value_type>
void ctcloss_layer_t<value_type>::backward(network_stage stage, cublasHandle_t* cublas_h, cudnnHandle_t* cudnn_h, registry_t<value_type>* reg ) {
	int input_l = this->get_input_layer_id();
    int curt_l  = this->get_id();
    // printf("ctcloss_layer_t backward layer %d\n", this->get_id());
    tensor_t<value_type>* input = reg->get_reg_output(input_l, curt_l);
    input->increase_cur_use_counter(BACKWARD);
	// input->increase_use_count();

    tensor_t<value_type>* output = this->get_f_out();
    tensor_t<value_type>* label  = reg->get_train_label();
    
    tensor_t<value_type>* dy  = gradients;
    tensor_t<value_type>* dx = this->get_b_data();
    tensor_t<value_type>* y = probs;

    if (sub_dx == NULL) {
        sub_dx = (value_type**)malloc(seqLength * sizeof(value_type*));
    }
    for (int i = 0; i < seqLength; i++) {
        sub_dx[i] = dx->get_gpu_ptr_v2(i*miniBatch*inputSize);
    }
    
    for (int i = 0; i < seqLength; i++) {
        checkCUDNN( cudnnSoftmaxBackward(
                *cudnn_h,
                CUDNN_SOFTMAX_ACCURATE,
                CUDNN_SOFTMAX_MODE_INSTANCE,
                &this->alpha,
                sub_probsDesc,
                sub_probs[i],
                sub_gradientsDesc,
                sub_gradients[i],
                &this->beta,
                sub_dxDesc,
                sub_dx[i]));
    }

    output->scale( 1.0f / output->get_N() );
    // printf("ctcloss_layer_t backward layer %d done\n", this->get_id());  

#ifdef DEBUG
    output->printTensor("Gradient from Softmax");
#endif
}

template <class value_type>
void ctcloss_layer_t<value_type>::get_loss(LOSS_MODE loss_mode, std::vector<value_type> *loss) {
    loss->clear();
    value_type sum = 0.0;
    // std::vector<value_type> *loss = new std::vector<value_type>;
    // #ifndef TRAINING_CONFIGURATION_SEARCH

        costs->GPUtoCPU();
        value_type *loss_data = costs->get_cpu_ptr();
        if (loss_mode == SUM || loss_mode == MEAN) {
            for (int i = 0; i < this->miniBatch; i++) {
                sum += loss_data[i];
            }
            if (loss_mode == SUM) {
                loss->push_back(sum);
            }
            else { // loss_mode == MEAN
                loss->push_back(sum/(value_type)(this->miniBatch));
            }
        }
        else if (loss_mode == NONE) {
            for (int i = 0; i < this->miniBatch; i++) {
                loss->push_back(loss_data[i]);
            }
        }
    // #endif
}

template <class value_type>
void ctcloss_layer_t<value_type>::fake_run(net_comp dir, registry_t<value_type>* reg) {
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

INSTANTIATE_CLASS(ctcloss_layer_t);
    
} //ATP namespace


