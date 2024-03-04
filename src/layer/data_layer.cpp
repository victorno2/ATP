//
// Created by ay27 on 17/3/31.
//

#include <layer/data_layer.h>
#include <util/error_util.h>

namespace ATP {

template<class value_type>
void data_layer_t<value_type>::data_fake_init() {
    size_t total;
	value_type *data_fake_cpu;
    if (is_rnn_model) {

        total = seqLength * miniBatch * inputSize;
        data_fake_cpu = new value_type[total];
        double x;
        double xDegrees = 90.0;
        for (size_t i = 0; i < seqLength; i++) {
            for (size_t j = 0; j < miniBatch; j++) {
                for (size_t k = 0; k < inputSize; k++) {
                    x = (xDegrees + 10.0*(double)i) * 3.14159 / 180.0;
                    data_fake_cpu[i*miniBatch*inputSize + j*inputSize + k] = (value_type)sin(x + 0.5*(double)k + 0.5*(double)i);// - 0.5*(double)inputSize);
                    // printf("%lf ", data_fake_cpu[i*miniBatch*inputSize + j*inputSize + k]);
                }
            }
        }
        // printf("\n");
        cudaMalloc((void**)&fake_data, sizeof(value_type)*total);
        cudaMemcpy(fake_data, data_fake_cpu, sizeof(value_type)*total, cudaMemcpyHostToDevice);

        total = miniBatch * labelSize;
        fake_label_int_cpu = new int[total];
        int xDegrees_int = 0;
        for (size_t i = 0; i < miniBatch; i++) {
            for (size_t j = 0; j < labelSize; j++) {
                fake_label_int_cpu[i*labelSize + j] = j%4 + 1;
                // printf("%d ", fake_label_int_cpu[i*labelSize + j]);
            }
        }
        // printf("\n");
    }
    else if (this->is_seq) {  // 
        total = seqLength * miniBatch * beamDim * embedSize;
        data_fake_cpu = new value_type[total];
        value_type *label_fake_cpu;
        label_fake_cpu = new value_type[seqLength];
        for (size_t i = 0; i < total; i++) {
            *(data_fake_cpu+i) = init_data;
        }
        for (size_t i = 0; i < seqLength; i++) {
            *(label_fake_cpu+i) = init_data;
        }
        cudaMalloc((void**)&fake_data, sizeof(value_type)*total);
        cudaMalloc((void**)&fake_label, sizeof(value_type)*seqLength);
        cudaMemcpy(fake_data, data_fake_cpu, sizeof(value_type)*total, cudaMemcpyHostToDevice);
        cudaMemcpy(fake_label, label_fake_cpu, sizeof(value_type)*seqLength, cudaMemcpyHostToDevice);
    }
    else {
        total = this->N * this->C * this->H * this->W;
        data_fake_cpu = new value_type[total];
        for (size_t i = 0; i < total; i++) { 
            *(data_fake_cpu+i) = init_data;
        }
        value_type *label_fake_cpu = new value_type[N];
        for (size_t i = 0; i < N; i++) {
            *(label_fake_cpu+i) = init_data;
        }
    // #ifdef POOL_MALLOC_MODE

    //  mem_controller->alloc_mem_by_gpu_mem_pool(&fake_data, sizeof(value_type)*total);
    //  mem_controller->alloc_mem_by_gpu_mem_pool(&fake_label, sizeof(value_type)*N);
    // #else
        cudaMalloc((void**)&fake_data, sizeof(value_type)*total);
        cudaMalloc((void**)&fake_label, sizeof(value_type)*N);
    // #endif
        cudaMemcpy(fake_data, data_fake_cpu, sizeof(value_type)*total, cudaMemcpyHostToDevice);
        cudaMemcpy(fake_label, label_fake_cpu, sizeof(value_type)*N, cudaMemcpyHostToDevice);
    }
    
}

template<class value_type>
void data_layer_t<value_type>::data_fake_init_pool_malloc_mode(void* gpu_pool_ptr, size_t* offset) {
    size_t _offset = *offset;
    size_t total;
	value_type *data_fake_cpu;
	
    if (this->is_rnn_model) {
        total = seqLength * miniBatch * inputSize;
        data_fake_cpu = new value_type[total];
        double x;
        double xDegrees = 90.0;
        for (size_t i = 0; i < seqLength; i++) {
            for (size_t j = 0; j < miniBatch; j++) {
                for (size_t k = 0; k < inputSize; k++) {
                    x = (xDegrees + 10.0*(double)i) * 3.14159 / 180.0;
                    data_fake_cpu[i*miniBatch*inputSize + j*inputSize + k] = (value_type)sin(x + 0.5*(double)k + 0.5*(double)i);// - 0.5*(double)inputSize);
                    // printf("%lf ", data_fake_cpu[i*miniBatch*inputSize + j*inputSize + k]);
                }
            }
        }
        // printf("\n");
        // while(1);
        this->fake_data = (value_type*)(gpu_pool_ptr + _offset);
        _offset += sizeof(value_type) * total;
        cudaMemcpy(fake_data, data_fake_cpu, sizeof(value_type)*seqLength*miniBatch*inputSize, cudaMemcpyHostToDevice);

        total = miniBatch * labelSize;
        fake_label_int_cpu = new int[total];
        int xDegrees_int = 0;
        for (size_t i = 0; i < miniBatch; i++) {
            for (size_t j = 0; j < labelSize; j++) {
                fake_label_int_cpu[i*labelSize + j] = j%4 + 1;
                // printf("%d ", fake_label_int_cpu[i*labelSize + j]);
            }
        }
        // printf("\n");
    }
    else if (this->is_seq) {  // 
        total = seqLength * miniBatch * beamDim * embedSize;
        data_fake_cpu = new value_type[total];
        value_type *label_fake_cpu;
        label_fake_cpu = new value_type[seqLength];
        for (size_t i = 0; i < total; i++) {
            *(data_fake_cpu+i) = init_data;
        }
        for (size_t i = 0; i < seqLength; i++) {
            *(label_fake_cpu+i) = init_data;
        }
        this->fake_data = (value_type*)(gpu_pool_ptr + _offset);
        _offset += sizeof(value_type) * total;
        fake_label = (value_type*)(gpu_pool_ptr + _offset);
        _offset += sizeof(value_type) * seqLength;
        *offset = _offset;
        cudaMemcpy(fake_data, data_fake_cpu, sizeof(value_type)*total, cudaMemcpyHostToDevice);
        cudaMemcpy(fake_label, label_fake_cpu, sizeof(value_type)*seqLength, cudaMemcpyHostToDevice);
    }
    else {
        total = this->N * this->C * this->H * this->W;
        data_fake_cpu = new value_type[total];
        value_type *label_fake_cpu;
        label_fake_cpu = new value_type[N];
        printf("total=%d, N=%d\n", total, N);
        for (size_t i = 0; i < total; i++) {
            *(data_fake_cpu+i) = init_data;
        }
        for (size_t i = 0; i < N; i++) {
            *(label_fake_cpu+i) = init_data;
        }
        this->fake_data = (value_type*)(gpu_pool_ptr + _offset);
        _offset += sizeof(value_type) * total;
        fake_label = (value_type*)(gpu_pool_ptr + _offset);
        _offset += sizeof(value_type) * N;
        *offset = _offset;
        // cudaMalloc((void**)&fake_data, sizeof(value_type)*total);
        // cudaMalloc((void**)&fake_label, sizeof(value_type)*N);
        cudaMemcpy(fake_data, data_fake_cpu, sizeof(value_type)*total, cudaMemcpyHostToDevice);
        cudaMemcpy(fake_label, label_fake_cpu, sizeof(value_type)*N, cudaMemcpyHostToDevice);
    }
}

template<class value_type>
void data_layer_t<value_type>::get_batch_fake(tensor_t<value_type> *_data, tensor_t<value_type> *_label) {
    _data->set_gpu_ptr(fake_data);
	_label->set_gpu_ptr(fake_label);
}

template<class value_type>
void data_layer_t<value_type>::get_batch_fake_int(tensor_t<value_type> *_data, tensor_t<value_type> *_label) {
    _data->set_gpu_ptr(fake_data);
	_label->set_cpu_ptr_int(fake_label_int);
}

template<class value_type>
size_t data_layer_t<value_type>::get_fake_data_label_size() {
    size_t temp_size = 0;
    size_t cuda_mem_block = 2097152;  // 2MB
    size_t block_num = 0;
    size_t total_size = 0;
    size_t align_size = 512;
#ifdef POOL_MALLOC_MODE
    temp_size = sizeof(value_type) * N*C*H*W;
    temp_size = (temp_size / align_size) * align_size + ((temp_size % align_size > 0 ? align_size : 0));  // align to 512
    total_size += temp_size;
    temp_size = sizeof(value_type) * N;
    temp_size = (temp_size / align_size) * align_size + ((temp_size % align_size > 0 ? align_size : 0));  // align to 512
    total_size += temp_size;
    printf("get_fake_data_label_size in data_layer = %zd\n", total_size);
    return total_size;
#else
    // temp_size = sizeof(value_type) * (N*C*H*W + N);
    while(true) {
        block_num++;
        if (temp_size > cuda_mem_block) {
            temp_size -= cuda_mem_block;
        }
        else {
            break;
        }
    }
    printf("get_fake_data_label_size in data_layer = %zd\n", block_num * cuda_mem_block);
    return block_num * cuda_mem_block;
#endif
}

template<class value_type>
void data_layer_t<value_type>::forward_setup(registry_t<value_type> *reg, cudnnHandle_t *cudnn_h) {
    // create tensor to store data and label
    printf("======>setup the forward data layer:%d\n", this->get_id());
    // this->fake_data = new tensor_t<value_type>(this->N, this->C, this->H, this->W, reg->get_vector(), DATA_SOURCE, this->get_id());
    // this->fake_label = new tensor_t<value_type>(this->N, 1, 1, 1, reg->get_vector(), DATA_SOURCE, this->get_id());
    
    tensor_t<value_type> *f_out;
    tensor_t<value_type> *label;
    
    if (is_rnn_model) {
        f_out = new tensor_t<value_type>(seqLength, miniBatch, inputSize, 1, reg->get_vector(), DATA_SOURCE, this->get_id());
        label = new tensor_t<value_type>(1, miniBatch, labelSize, 1, reg->get_vector(), DATA_SOURCE_INT, this->get_id());     
    } 
    else if (this->is_seq) {
        f_out = new tensor_t<value_type>(this->N, this->C, this->H, this->W, reg->get_vector(), DATA_SOURCE, this->get_id());
        printf("f_out->get_tensor_desc() = %d %d %d %d\n", f_out->get_N(), f_out->get_C(), f_out->get_H(), f_out->get_W());
        label = new tensor_t<value_type>(this->N, 1, 1, 1, reg->get_vector(), DATA_SOURCE, this->get_id());  
        printf("label->get_tensor_desc() = %d %d %d %d\n", label->get_N(), label->get_C(), label->get_H(), label->get_W());
        f_out->set_data_layout(SEQ2SEQ_TNBV);
    }
    else {
        f_out = new tensor_t<value_type>(this->N, this->C, this->H, this->W, reg->get_vector(), DATA_SOURCE, this->get_id());
        printf("f_out->get_tensor_desc() = %d %d %d %d\n", f_out->get_N(), f_out->get_C(), f_out->get_H(), f_out->get_W());
        label = new tensor_t<value_type>(this->N, 1, 1, 1, reg->get_vector(), DATA_SOURCE, this->get_id());  
        printf("label->get_tensor_desc() = %d %d %d %d\n", label->get_N(), label->get_C(), label->get_H(), label->get_W());
    }
    if(this->mode == DATA_TRAIN) {
        reg->set_train_label(label);
    } 
    else if(this->mode == DATA_TEST) {
        reg->set_test_label(label);
    }
    this->set_f_out(f_out, reg);

    // int cur_l_id = this->get_id();
    // int output_l_id = this->get_output_layer_id();
    // tensor_t<value_type>* dEdD_n = reg->get_reg_b_data(output_l_id, cur_l_id);
    // dEdD_n->increase_b_use_count();

    //register the forward dependency
    tensor_t<value_type>* output = this->get_f_out();
    reg->register_forward_dependency( this->get_id(), output );


#ifdef FAKE_TRAIN
    // output->set_gpu_ptr(this->fake_data);
    // // output->printRNNTensor("data_layer out");
    // printf("is_rnn_model = %d\n", is_rnn_model);
    // if (is_rnn_model) {
    //     label->set_cpu_ptr_int(this->fake_label_int_cpu); 
    //     label->printTensorInt("data_layer label"); 
    // }
    // else {
    //     label->set_gpu_ptr(this->fake_label); 
    // }
#endif  
}

template<class value_type>
void data_layer_t<value_type>::backward_setup(registry_t<value_type> *reg, cudnnHandle_t *cudnn_h) {
    int cur_l_id = this->get_id();
    int output_l_id = this->get_output_layer_id();
    tensor_t<value_type>* dEdD_n = reg->get_reg_b_data(output_l_id, cur_l_id);
    dEdD_n->increase_b_use_count();
    this->set_dy(dEdD_n);
}

template<class value_type>
std::vector<value_type> data_layer_t<value_type>::forward(network_stage stage, cublasHandle_t *cublas_h, cudnnHandle_t *cudnn_h, registry_t<value_type> *reg) {
    double end, start;
    // start = get_cur_time();
    tensor_t<value_type>* label = NULL;
    if(this->mode == DATA_TRAIN) {
        label = reg->get_train_label();
    } else {
        label = reg->get_test_label();
    }
    tensor_t<value_type>* output = this->get_f_out();
    // end = get_cur_time();
    // printf("before FAKE_TRAIN = %lf\n", end - start);
    // output->printTensor("output 1");
#ifdef FAKE_TRAIN
    output->set_gpu_ptr(this->fake_data);
    
    // printf("is_rnn_model = %d\n", is_rnn_model);
    if (is_rnn_model) {
        label->set_cpu_ptr_int(this->fake_label_int_cpu); 
        // output->printRNNTensor("data_layer out");
        // label->printTensorInt("data_layer label");
    }
    else {
        label->set_gpu_ptr(this->fake_label); 
    }
#else
    // checkCudaErrors( cudaFree(ptr) );
    reader->get_batch(output, label);
#endif

    // while(1);

#ifdef DEBUG
    output->printTensor("output from data layer");
#endif
#ifdef PRINT_DATA
    if(this->mode == DATA_TEST) {
        printf("------------------testing@layer%p output%p label%p---------------------\n", this, output, label);
        output->printTensorNoDebug("test input image");
        label->printTensorNoDebug("test labels");
    } else {
        printf("------------------training@layer%p output%p label%p---------------------\n", this, output, label);
        output->writeToFile("training_tensor");
        label->printTensorNoDebug("train labels");
    }
#endif
	// output->printTensor("output");
    return std::vector<value_type>();
}

template<class value_type>
void data_layer_t<value_type>::backward(network_stage stage, cublasHandle_t *cublas_h, cudnnHandle_t *cudnn_h, registry_t<value_type> *reg) {
    int curt_l_id   = this->get_id();
    int output_l_id = this->get_output_layer_id();
    tensor_t<value_type>* dEdD_n = reg->get_reg_b_data(output_l_id, curt_l_id);
    dEdD_n->decrease_cur_b_use_count();
}

template<class value_type>
void data_layer_t<value_type>::fake_run(net_comp dir, registry_t<value_type> *reg) {
    if (dir == FORWARD) {

    }
    else {
        
    }
}

INSTANTIATE_CLASS(data_layer_t);
}
