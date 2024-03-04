#if !defined(_CUDNN_RNN_H_)
#define _CUDNN_RNN_H_
#include <switch.h>
#include <util/common.h>
#include <layer/base_network_layer.h>

namespace ATP{
    
template <class value_type>
class rnn_layer_t:base_network_layer_t<value_type>
{
private:
    cudnnRNNDescriptor_t rnnDesc;
	tensor_t<value_type>* hx;
	tensor_t<value_type>* cx;
	tensor_t<value_type>* w;
	tensor_t<value_type>* hy;
	tensor_t<value_type>* cy;
	tensor_t<value_type>* dhx;
	tensor_t<value_type>* dcx;
	tensor_t<value_type>* dw;
	tensor_t<value_type>* dw_prev;
	tensor_t<value_type>* dhy;
	tensor_t<value_type>* dcy;
    cudnnFilterDescriptor_t wDesc, dwDesc;
	cudnnTensorDescriptor_t *xDesc, *yDesc, *dxDesc, *dyDesc;
	cudnnTensorDescriptor_t hxDesc, cxDesc;
	cudnnTensorDescriptor_t hyDesc, cyDesc;
	cudnnTensorDescriptor_t dhxDesc, dcxDesc;
	cudnnTensorDescriptor_t dhyDesc, dcyDesc;
    // void                           *workspace;
    // size_t                          workSpaceSizeInBytes;
    // void                           *reserveSpace;
    // size_t                          reserveSpaceSizeInBytes;
	cudnnDropoutDescriptor_t   	dropoutDesc;
	cudnnRNNInputMode_t 	   	inputMode;
	cudnnDirectionMode_t	   	direction;
	cudnnRNNMode_t 			   	mode;
	cudnnRNNAlgo_t			   	algo;
	cudnnDataType_t 		   	mathPrec;
	
	int 						minibatch; 
	cudnnPersistentRNNPlan_t	plan;

    size_t miniBatch;
    size_t inputSize;
    size_t seqLength;
    size_t hiddenSize; 
    size_t numLayers;

	size_t weightsSize;
    size_t f_rnn_buff_size;
    tensor_t<value_type>* f_rnn_buff;
    size_t b_rnn_buff_size;
    tensor_t<value_type>* b_rnn_buff;
    size_t reserveSpaceSizeInBytes;
	tensor_t<value_type>* reserve_buff;
	
	initializer_t<value_type> *weight_initializer;
    initializer_t<value_type> *bias_initializer;
	
	cudnnRNNBiasMode_t biasMode;
	
	size_t dropout_state_size_bytes;
	float dropout_rate;
	tensor_t<value_type>* dropout_state;
	
	void createDesc() {
        checkCUDNN( cudnnCreateRNNDescriptor(&(this->rnnDesc)));
		checkCUDNN( cudnnCreateDropoutDescriptor(&(this->dropoutDesc)));
		checkCUDNN( cudnnCreateFilterDescriptor(&(this->wDesc)));
    }
	
	void init_gpu_data(void* gpu_ptr, size_t num, value_type data) {
		float *cpu_ptr;
		cudaMallocHost(&cpu_ptr, num*sizeof(float));
		for (size_t i = 0; i < num; i++) {
			*(cpu_ptr+i) = data;
		}
		checkCudaErrors(cudaMemcpy((void*)gpu_ptr, (void*)cpu_ptr, num*sizeof(value_type), cudaMemcpyHostToDevice));
		checkCudaErrors(cudaFreeHost(cpu_ptr));
}
    
public:
	rnn_layer_t(size_t inputSize, size_t hiddenSize, size_t seqLength, size_t numLayers, 
		RNN_TYPE rnn_type = RNN_TANH, 
		RNN_BIAS rnn_bias = DOUBLE, 
		RNN_DIRECTION rnn_direction = UNIDIRECTIONAL, 
		float dropout_rate = 0.5,
		initializer_t<value_type> *weight_initializer = new gaussian_initializer_t<float>(0, 0.01),
		initializer_t<value_type> *bias_initializer = new constant_initializer_t<float>(0.0),
		cudnnRNNInputMode_t inputMode = CUDNN_LINEAR_INPUT, 
		cudnnRNNAlgo_t algo = CUDNN_RNN_ALGO_STANDARD)
		: seqLength(seqLength), inputSize(inputSize), hiddenSize(hiddenSize), numLayers(numLayers),
        weight_initializer(weight_initializer), 
		dropout_rate(dropout_rate),		
		inputMode(inputMode), algo(algo),
		base_network_layer_t<value_type>(RNN)
    {
		switch (rnn_type)
        {
            case RNN_RELU	: mode = CUDNN_RNN_RELU;   	break;
            case RNN_TANH	: mode = CUDNN_RNN_TANH;  	break;
            case LSTM		: mode = CUDNN_LSTM; 		break;
			case GRU		: mode = CUDNN_GRU; 		break;
            default 		: FatalError("Unsupported RNN type");
        }
		
		switch (rnn_bias)
        {
            case NO			: biasMode = CUDNN_RNN_NO_BIAS;   		break;
            case SINGLE		: biasMode = CUDNN_RNN_SINGLE_INP_BIAS; break;
            case DOUBLE		: biasMode = CUDNN_RNN_DOUBLE_BIAS; 	break;
            default 		: FatalError("Unsupported bias mode");
        }
		
		switch (rnn_direction)
        {
            case UNIDIRECTIONAL	: direction = CUDNN_UNIDIRECTIONAL; break;
            case BIDIRECTIONAL	: direction = CUDNN_BIDIRECTIONAL; 	break;
            default 			: FatalError("Unsupported bias mode");
        }
		
		switch (sizeof(value_type))
        {
            case 2: mathPrec = CUDNN_DATA_HALF;   break;
            case 4: mathPrec = CUDNN_DATA_FLOAT;  break;
            case 8: mathPrec = CUDNN_DATA_DOUBLE; break;
            default : FatalError("Unsupported data type");
        }
        createDesc();
		this->enable_bias(false);
    }
    
    ~rnn_layer_t() {
        checkCUDNN( cudnnDestroyRNNDescriptor(this->rnnDesc) );
		checkCUDNN( cudnnDestroyDropoutDescriptor(this->dropoutDesc) );
		checkCUDNN( cudnnDestroyFilterDescriptor((this->wDesc)) );
    }
	
	// tensor_t<value_type>* get_rnn_reserve_buff() {
	// 	return this->reserve_buff;
	// }
	
    void forward_setup( registry_t<value_type>* reg, cudnnHandle_t* cudnn_h );
    
    void backward_setup( registry_t<value_type>* reg, cudnnHandle_t* cudnn_h );
        
    std::vector<value_type> forward (network_stage stage, cublasHandle_t* cublas_h, cudnnHandle_t* cudnn_h, registry_t<value_type>* reg);
    
    void backward(network_stage stage, cublasHandle_t* cublas_h, cudnnHandle_t* cudnn_h, registry_t<value_type>* reg);

    void gen_description(char* buff, size_t* len_in_byte) {
        size_t meta_len_in_byte;
        this->gen_meta_description(buff, &meta_len_in_byte);

//        typedef enum
//        {
//            CUDNN_ACTIVATION_SIGMOID      = 0,
//            CUDNN_ACTIVATION_RELU         = 1,
//            CUDNN_ACTIVATION_TANH         = 2,
//            CUDNN_ACTIVATION_CLIPPED_RELU = 3
//        } cudnnActivationMode_t;

        value_type _m = mode;
        memcpy(buff+meta_len_in_byte, &_m, sizeof(value_type));

        *len_in_byte = meta_len_in_byte + sizeof(value_type);
    }

	void fake_run(net_comp dir, registry_t<value_type>* reg);
};

} // ATP namespace
#endif // _CUDNN_RNN_H_



