#include <layer/fully_connected_layer.h>
#include <stdlib.h>
#include <stdio.h>

namespace ATP {
    
#define DEBUG
template<class value_type>
void fully_connected_layer_t<value_type>::forward_setup( registry_t<value_type>* reg, cudnnHandle_t* cudnn_h ) {
    printf("======>setup the forward fully connected layer:%d start\n", this->get_id());
    int input_l = this->get_input_layer_id();
    int curt_l  = this->get_id();
    tensor_t<value_type>* input = reg->get_reg_output(input_l, curt_l);
    
#ifdef DEBUG
    printf("the layer's input tensor:%p, size : %zu %zu %zu %zu\n", input, input->get_N(), input->get_C(), input->get_H(), input->get_W());
#endif
    assert( input != NULL);
	this->set_f_in(input, reg);
	// input->increase_use_count_initial();
	// input->increase_use_count();
    input->increase_use_counter(FORWARD);

    this->input_dim = input->get_C() * input->get_H() * input->get_W();

    this->N = input->get_N();

    if (input->get_data_layout() == SEQ2SEQ_TNBV) {
        this->input_dim = input->get_seq_vect();
        this->N = input->get_seq_time() * input->get_seq_batch() * input->get_seq_beam();
    }

    //right align rule applied to tensor
    //setup weight
    tensor_t<value_type>* weight = new tensor_t<value_type>(output_dim, input_dim, 1, 1, reg->get_vector(), PARAM, this->get_id());
    this->set_weight(weight, reg);
    weight->init(this->weight_initializer);
    
    //setup bias
    tensor_t<value_type>* bias    = new tensor_t<value_type>(output_dim, 1, 1, 1, reg->get_vector(), PARAM, this->get_id());
    tensor_t<value_type>* bias_multiplier = new tensor_t<value_type>(1, 1, 1, this->N, reg->get_vector(), AUX, this->get_id());
    bias->init(this->bias_initializer);
    //to remove
    // for(size_t i = 0; i < N; i++)  bias_multiplier->set_scalar(0, 0, 0, i, 1);
    //bias_multiplier->init(new constant_initializer_t<value_type>(1.0f));
    
    this->set_bias(bias, reg);
    this->bias_multiplier = bias_multiplier;
    
    //setup output tensor
    tensor_t<value_type>* output  = new tensor_t<value_type>(N, output_dim, 1, 1, reg->get_vector(), DATA, this->get_id());
    this->set_f_out(output, reg);

    

    this->m_d = (size_t) this->get_f_out()->get_N();  //output dim
    this->k_d = (size_t) this->get_weight()->get_C(); //@line14, input dim
    this->n_d = (size_t) this->get_weight()->get_N();  //@line14, total images

    // tensor_t<value_type>* b_data  = new tensor_t<value_type>(input->get_N(), input->get_C(), input->get_H(), input->get_W(), reg->get_vector(), B_DATA, this->get_id());
    tensor_t<value_type>* b_data  = new tensor_t<value_type>(this->N, input_dim, 1, 1, reg->get_vector(), B_DATA, this->get_id());
    this->set_b_data(b_data, reg);

    

    if (input->get_data_layout() == SEQ2SEQ_TNBV) {
        output->reshape(input->get_seq_time(), input->get_seq_batch(), input->get_seq_beam(), this->output_dim);  
        output->set_data_layout(SEQ2SEQ_TNBV);
    }
    
    assert( this->get_weight() != NULL );
    assert( this->get_bias()   != NULL );
    assert( this->get_f_out()  != NULL );
    assert( this->bias_multiplier != NULL );
    
    //register the forward dependency
    tensor_t<value_type>* t_in   = reg->get_reg_output(input_l, curt_l);
    tensor_t<value_type>* t_out  = this->get_f_out();
    bias                         = this->get_bias();
    weight                       = this->get_weight();
    
    assert( t_in   != NULL );
    assert( weight != NULL );
    assert( t_out  != NULL );
    assert( bias   != NULL );
    
    reg->register_forward_dependency( this->get_id(), t_in );
    reg->register_forward_dependency( this->get_id(), weight );
    reg->register_forward_dependency( this->get_id(), t_out );
    reg->register_forward_dependency( this->get_id(), bias );
    reg->register_forward_dependency( this->get_id(), bias_multiplier );
    printf("======>setup the forward fully connected layer:%d done\n", this->get_id());
}
#undef DEBUG

template<class value_type>
void fully_connected_layer_t<value_type>::backward_setup( registry_t<value_type>* reg, cudnnHandle_t* cudnn_h ) {
    printf("======>setup the backward fully connected layer:%d\n", this->get_id());
    assert(reg != NULL);
    assert(cudnn_h != NULL);
    
    int input_l = this->get_input_layer_id();
    int curt_l  = this->get_id();
    
    //setup backward data
    tensor_t<value_type>* input   = reg->get_reg_output(input_l, curt_l);
    input->increase_use_counter(BACKWARD);
    tensor_t<value_type>* weight  = this->get_weight();
    assert(input  != NULL);
    assert(weight != NULL);
    
    int curt_l_id   = this->get_id();
    int input_l_id  = this->get_input_layer_id();
    int output_l_id = this->get_output_layer_id();
    tensor_t<value_type>* dEdD_n          = reg->get_reg_b_data(output_l_id, curt_l_id);
    tensor_t<value_type>* dEdD_c          = this->get_b_data();
    this->ld_a =  weight->get_W();
    this->ld_b =  dEdD_n->get_W();
    this->ld_c =  (dEdD_c->get_C() * dEdD_c->get_H() * dEdD_c->get_W());

    // size_t N  = input->get_N();
    // size_t C  = input->get_C();
    // size_t H  = input->get_H();
    // size_t W  = input->get_W();
    // size_t input_dim = C*H*W;
    // size_t ouput_dim = this->output_dim;

    // tensor_t<value_type>* b_data  = new tensor_t<value_type>(input->get_N(), input->get_C(), input->get_H(), input->get_W(), reg->get_vector(), B_DATA, this->get_id());
    
    //setup backward weight grad
    tensor_t<value_type>* weight_grad = new tensor_t<value_type>(weight->get_N(), weight->get_C(), weight->get_H(), weight->get_W(), reg->get_vector(), GRAD, this->get_id());
    tensor_t<value_type>* weight_prev = new tensor_t<value_type>(weight->get_N(), weight->get_C(), weight->get_H(), weight->get_W(), reg->get_vector(), GRAD, this->get_id());
    weight_prev->init(new constant_initializer_t<value_type>(0));
//    weight_prev->const_fill(0); //zero init
    this->set_weight_grad(weight_grad, reg);
    this->set_weight_prev(weight_prev, reg);
    
    //setup backward bias grad
    tensor_t<value_type>* bias      = this->get_bias();
    tensor_t<value_type>* bias_grad = new tensor_t<value_type>(bias->get_N(), bias->get_C(), bias->get_H(), bias->get_W(), reg->get_vector(), GRAD, this->get_id());
    tensor_t<value_type>* bias_prev = new tensor_t<value_type>(bias->get_N(), bias->get_C(), bias->get_H(), bias->get_W(), reg->get_vector(), GRAD, this->get_id());
    bias_prev->init(new constant_initializer_t<value_type>(0));
//    bias_prev->const_fill(0); //zero init
    this->set_bias_grad(bias_grad, reg);
    this->set_bias_prev(bias_prev, reg);
    
    assert( this->get_weight_grad() != NULL );
    assert( this->get_b_data()      != NULL );
    assert( this->get_bias_grad()   != NULL );
    
    //register the backward dependency
    
    this->set_dy(dEdD_n);
    dEdD_n->increase_b_use_count();
    weight          = this->get_weight();
    bias_grad       = this->get_bias_grad();
    weight_grad     = this->get_weight_grad();
    
    if (input->get_data_layout() == SEQ2SEQ_TNBV) {
        dEdD_c->set_data_layout(SEQ2SEQ_TNBV);
        dEdD_c->reshape(input->get_N(), input->get_C(), input->get_H(), input->get_W());
    }


    assert( input        != NULL );
    assert( dEdD_n      != NULL );
    assert( dEdD_c      != NULL );
    assert( weight      != NULL );
    assert( bias_grad   != NULL );
    assert( weight_grad != NULL );
    
    reg->register_backward_dependency(this->get_id(), input        );
    reg->register_backward_dependency(this->get_id(), dEdD_n      );
    reg->register_backward_dependency(this->get_id(), dEdD_c      );
    reg->register_backward_dependency(this->get_id(), weight      );
    reg->register_backward_dependency(this->get_id(), bias_grad   );
    reg->register_backward_dependency(this->get_id(), weight_grad );
    reg->register_backward_dependency( this->get_id(), bias_multiplier );
}

template<>
void fully_connected_layer_t<float>::mat_multiply(cublasHandle_t* cublas_h,
                                                 int m, int n, int k,
                                                 cublasOperation_t TransA, cublasOperation_t TransB,
                                                 float alpha, float beta,
                                                 float* A, int lda,
                                                 float* B, int ldb,
                                                 float* C, int ldc) {
    checkCublasErrors(
                      cublasSgemm(*(cublas_h),
                                  TransA, TransB,
                                  m, n, k,
                                  &alpha,
                                  A, lda,
                                  B, ldb,
                                  &beta,
                                  C, ldc)
                      );
}
    
template<>
void fully_connected_layer_t<double>::mat_multiply(cublasHandle_t* cublas_h,
                                                 int m, int n, int k,
                                                 cublasOperation_t TransA, cublasOperation_t TransB,
                                                 double alpha, double beta,
                                                 double* A, int lda,
                                                 double* B, int ldb,
                                                 double* C, int ldc)
{
    checkCublasErrors(
                      cublasDgemm(*(cublas_h),
                                  TransA, TransB,
                                  m, n, k,
                                  &alpha,
                                  A, lda,
                                  B, ldb,
                                  &beta,
                                  C, ldc)
                      );
}
    
// #define DEBUG
template<class value_type>
std::vector<value_type> fully_connected_layer_t<value_type>::forward(network_stage stage, cublasHandle_t* cublas_h, cudnnHandle_t* cudnn_h, registry_t<value_type> *reg) {
    //--------forward data operation--------//
    // const int m_d = (int) this->get_f_out()->get_N();  //output dim
    // const int k_d = (int) this->get_weight()->get_C(); //@line14, input dim
    // const int n_d = (int) this->get_weight()->get_N();  //@line14, total images
    // printf("forward layer%d fully connected\n", this->get_id());
    int input_l = this->get_input_layer_id();
    int curt_l  = this->get_id();
    tensor_t<value_type>* t_in   = reg->get_reg_output(input_l, curt_l);
    t_in->increase_cur_use_counter(FORWARD);
	// t_in->decrease_use_count();
    tensor_t<value_type>* weight = this->get_weight();
    tensor_t<value_type>* t_out  = this->get_f_out();
    tensor_t<value_type>* bias   = this->get_bias();
    
#ifdef DEBUG
    t_in->printTensorData("before fully connected, input", 2);
    printf("before input tensor from %d to %d\n", input_l, curt_l);
    this->get_weight()->printTensorData("before fully connected, weight", 2);
    this->bias_multiplier->printTensorData("before fully connected, bias multiplier", 2);
    
    this->get_bias()->printTensorData("before fully connected, bias", 2);
    this->get_f_out()->printTensorData("before fully connected, bias output", 2);
#endif
    //forward data
    // printf("\nstart forward data\n");
    fflush(stdout);
    // printf("\nstart forward data\n");
    // system("pause");
    mat_multiply(cublas_h,
                 n_d, m_d, k_d,
                 CUBLAS_OP_T, CUBLAS_OP_N,
                 this->one, this->zero,
                 weight->get_gpu_ptr(), k_d,
                 t_in->get_gpu_ptr(),   k_d,
                 t_out->get_gpu_ptr(),  n_d );
    
    //--------forward bias operation--------//
    if(this->is_bias_enable()) {
        // printf("\nstart forward bias\n");
        mat_multiply(cublas_h,
                     n_d, m_d, this->one,
                     CUBLAS_OP_N, CUBLAS_OP_N,
                     this->one, this->one,
                     bias->get_gpu_ptr(),
                     (int) n_d,
                     this->bias_multiplier->get_gpu_ptr(),
                     (int) this->one,
                     t_out->get_gpu_ptr(),
                     (int) n_d );
        // printf("\nend forward bias\n");
    }

#ifdef DEBUG
    this->get_weight()->printTensorData("after fully connected, weight", 2);
    this->bias_multiplier->printTensorData("after fully connected, bias multiplier", 2);
    t_in->printTensorData("after fully connected, input", 2);
    this->get_bias()->printTensorData("after fully connected, bias", 2);
    this->get_f_out()->printTensorData("after fully connected, output", 2);
#endif

    // printf("\nend fully_connected_layer_t<value_type>::forward\n");

    return std::vector<value_type>();
}
// #undef DEBUG  

template<class value_type>
void fully_connected_layer_t<value_type>::backward_data(cublasHandle_t* cublas_h, cudnnHandle_t* cudnn_h, registry_t<value_type> *reg) {
    int curt_l_id   = this->get_id();
    int output_l_id = this->get_output_layer_id();

    tensor_t<value_type>* dEdD_n   = reg->get_reg_b_data(output_l_id, curt_l_id);
    tensor_t<value_type>* dEdD_c   = this->get_b_data();
    tensor_t<value_type>* weight      = this->get_weight();
    // w[inputxoutput]*sigma(l+1)'[outputxN] = sigma[inputxN] column format
    // takes another transpose to the row format
    
    // const int m_d = (int) this->get_f_out()->get_N();  //output dim
    // const int k_d = (int) this->get_weight()->get_C(); //@line14, input dim
    // const int n_d = (int) this->get_weight()->get_N();  //@line14, total images
    int ld_a = (int) weight->get_W();
    int ld_b = (int) dEdD_n->get_W();
    int ld_c = (int) (dEdD_c->get_C() * dEdD_c->get_H() * dEdD_c->get_W());
    // value_type one   = 1;
    // value_type zero  = 0;
    
    mat_multiply(cublas_h, k_d, m_d, n_d,
                 CUBLAS_OP_N, CUBLAS_OP_N,
                 this->one, this->zero,
                 weight->get_gpu_ptr(), k_d,
                 dEdD_n->get_gpu_ptr(), n_d,
                 dEdD_c->get_gpu_ptr(), k_d);
    dEdD_n->decrease_cur_b_use_count();

#ifdef DEBUG
    this->get_b_data()->printTensor("fully connected, backward data");
#endif
}

template<class value_type>
void fully_connected_layer_t<value_type>::backward_weight(cublasHandle_t* cublas_h, cudnnHandle_t* cudnn_h, registry_t<value_type> *reg) {
    int input_l_id  = this->get_input_layer_id();
    int curt_l_id   = this->get_id();
    int output_l_id = this->get_output_layer_id();


    tensor_t<value_type>* t_in        = reg->get_reg_output(input_l_id, curt_l_id);
    tensor_t<value_type>* weight_grad = this->get_weight_grad();
    tensor_t<value_type>* dEdD_n      = reg->get_reg_b_data(output_l_id, curt_l_id);
    //input[flatten_dim, N] * dEdD[N, y] = weight_grad[flatten_dim,y]
    // dEdD_n->decrease_b_use_count();    // dEdD_n b_use_count has been decreased in backward_data()
    // const int m_d = (int) this->get_f_out()->get_N();  //output dim
    // const int k_d = (int) this->get_weight()->get_C(); //@line14, input dim
    // const int n_d = (int) this->get_weight()->get_N();  //@line14, total images
// #undef GRAD_ACC
#ifdef GRAD_ACC
    // weight_grad = one * dEdD * t_in + one * weight_grad, weight_grad is constantly being accumulated until grad_zero;
    mat_multiply(cublas_h, k_d, n_d, m_d,
                 CUBLAS_OP_N, CUBLAS_OP_T,
                 this->one, this->one,
                 t_in->get_gpu_ptr(),   //bottom data
                 (int) k_d,
                 dEdD_n->get_gpu_ptr(), //top diff
                 (int) n_d,
                 weight_grad->get_temp_gpu_ptr(),
                 (int) k_d );
    size_t total_params;
    // value_type one = 1.0;
    // total_params = weight_grad->get_N() * weight_grad->get_C() * weight_grad->get_H() * weight_grad->get_W();
    // checkCublasErrors(cublasSaxpy(*(cublas_h), total_params, (const float*)&(this->one), (const float*)weight_grad->get_temp_gpu_ptr(), 1, (float*)weight_grad->get_gpu_ptr(), 1));
#else
    // mat_multiply(cublas_h, k_d, n_d, m_d,
    //              CUBLAS_OP_N, CUBLAS_OP_T,
    //              this->one, this->one,
    //              t_in->get_gpu_ptr(),   //bottom data
    //              (int) k_d,
    //              dEdD_n->get_gpu_ptr(), //top diff
    //              (int) n_d,
    //              weight_grad->get_gpu_ptr(),
    //              (int) k_d );
     mat_multiply(cublas_h, k_d, n_d, m_d,
                 CUBLAS_OP_N, CUBLAS_OP_T,
                 this->one, this->zero,
                 t_in->get_gpu_ptr(),   //bottom data
                 (int) k_d,
                 dEdD_n->get_gpu_ptr(), //top diff
                 (int) n_d,
                 weight_grad->get_gpu_ptr(),
                 (int) k_d );
#endif
// #define GRAD_ACC
// #define DEBUG
#ifdef DEBUG
    printf("m:%d n:%d, k:%d: ld_dEdD:%zu ld_weight_grad:%zu\n",m_d, n_d, k_d, dEdD_n->get_W(), weight_grad->get_W());
    this->get_weight_grad()->printTensor("fully connected, backward weight grad");
#endif
// #undef DEBUG
}
    
template<class value_type>
void fully_connected_layer_t<value_type>::backward_bias(cublasHandle_t* cublas_h, cudnnHandle_t* cudnn_h, registry_t<value_type> *reg)
{
    int curt_l_id   = this->get_id();
    int output_l_id = this->get_output_layer_id();
    
    tensor_t<value_type>* dEdD            = reg->get_reg_b_data(output_l_id, curt_l_id);
    tensor_t<value_type>* bias_multiplier = this->bias_multiplier;
    //input[1, N] * dEdD[N, output] = bias_diff[1, output]
    tensor_t<value_type>* bias_grad       = this->get_bias_grad();
    
    const int m_d = (int) this->get_f_out()->get_N();  //output dim
    const int n_d = (int) this->get_weight()->get_N();  //@line14, total images
// #undef GRAD_ACC
#ifdef GRAD_ACC
    // bias_grad = one * dEdD * bias_multiplier + one * bias_grad, bias_grad is constantly being accumulated until grad_zero;
    cublas_gemv(cublas_h, CUBLAS_OP_N,
                n_d, m_d,
                &(this->one),
                dEdD->get_gpu_ptr(), n_d,
                bias_multiplier->get_gpu_ptr(), 1,
                &(this->one),
                bias_grad->get_gpu_ptr(), 1 );
    // size_t total_params;
    // value_type one = 1.0;
    // total_params = bias_grad->get_N() * bias_grad->get_C() * bias_grad->get_H() * bias_grad->get_W();
    // checkCublasErrors(cublasSaxpy(*(cublas_h), total_params, (const float*)&(this->one), (const float*)bias_grad->get_temp_gpu_ptr(), 1, (float*)bias_grad->get_gpu_ptr(), 1));
#else
    cublas_gemv(cublas_h, CUBLAS_OP_N,
                n_d, m_d,
                &(this->one),
                dEdD->get_gpu_ptr(), n_d,
                bias_multiplier->get_gpu_ptr(), 1,
                &(this->zero),
                bias_grad->get_gpu_ptr(), 1 );
#endif
// #define GRAD_ACC
#ifdef DEBUG
    this->get_bias_grad()->printTensor("fully connected, backward bias grad");
#endif
}
    
template<class value_type>
void fully_connected_layer_t<value_type>::backward(network_stage stage, cublasHandle_t* cublas_h, cudnnHandle_t* cudnn_h, registry_t<value_type> *reg) {
    int input_l = this->get_input_layer_id();
    int curt_l  = this->get_id();
    tensor_t<value_type>* t_in   = reg->get_reg_output(input_l, curt_l);
    t_in->increase_cur_use_counter(BACKWARD);
	// t_in->increase_use_count();
    // printf("\nstart backward_weight\n");
    backward_weight(cublas_h, cudnn_h, reg);
    // printf("\nstart backward_data\n");
    backward_data(cublas_h, cudnn_h, reg);
    // printf("\nend backward\n");
    if(this->is_bias_enable() ) {
        backward_bias(cublas_h, cudnn_h, reg);
    }
}

template <class value_type>
void fully_connected_layer_t<value_type>::fake_run(net_comp dir, registry_t<value_type>* reg) {
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

INSTANTIATE_CLASS(fully_connected_layer_t);

}// ATP namespace
