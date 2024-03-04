#if !defined(_CUDNN_CTCLOSS_H_)
#define _CUDNN_CTCLOSS_H_
#include <math.h>       /* log */
#include <switch.h>
#include <tensor.h>
#include <util/common.h>
#include <layer/base_network_layer.h>

namespace ATP{
    
template <class value_type>
class ctcloss_layer_t:base_network_layer_t<value_type>
{
private:
	
	LOSS_MODE loss_mode;
	std::vector<value_type> loss;
	
	const cudnnCTCLossAlgo_t algo;
	cudnnCTCLossDescriptor_t ctcLossDesc;
	
	const value_type alpha = 1.0;
	const value_type beta = 0.0;
	
	cudnnTensorDescriptor_t yDesc;
	cudnnTensorDescriptor_t xDesc;
	
	int *labelLengths;
	int *inputLengths;
	int *labels;
	
	int seqLength;
    int labelLength;
	int miniBatch;
	int inputSize;
	cudnnDataType_t mathPrec;
	cudnnTensorDescriptor_t gradientsDesc;
	cudnnTensorDescriptor_t probsDesc;
	
	tensor_t<value_type> *probs;
	tensor_t<value_type> *costs;
	tensor_t<value_type> *gradients;
	value_type **sub_probs = NULL;
	value_type **sub_gradients = NULL;
	value_type **sub_inputs = NULL;
	value_type **sub_dx = NULL;
	cudnnTensorDescriptor_t sub_probsDesc;
	cudnnTensorDescriptor_t sub_gradientsDesc;
	cudnnTensorDescriptor_t sub_inputsDesc;
	cudnnTensorDescriptor_t sub_dxDesc;
	
	void *workspace_void;
    tensor_t<value_type> *workspace = NULL;
    size_t workSpaceSizeInBytes;

public:
    ctcloss_layer_t(LOSS_MODE loss_mode = MEAN, cudnnCTCLossAlgo_t algo = CUDNN_CTC_LOSS_ALGO_DETERMINISTIC)
    :algo(algo), loss_mode(loss_mode), base_network_layer_t<value_type>(SOFTMAX)
    {
		switch (sizeof(value_type))
        {
            case 2: mathPrec = CUDNN_DATA_HALF;   break;
            case 4: mathPrec = CUDNN_DATA_FLOAT;  break;
            case 8: mathPrec = CUDNN_DATA_DOUBLE; break;
            default : FatalError("Unsupported data type");
        }
    }
    
    ~ctcloss_layer_t()
    {
        
    }
    
    void forward_setup( registry_t<value_type>* reg, cudnnHandle_t* cudnn_h );
    
    void backward_setup( registry_t<value_type>* reg, cudnnHandle_t* cudnn_h );
    
    std::vector<value_type> forward(network_stage stage, cublasHandle_t* cublas_h, cudnnHandle_t* cudnn_h, registry_t<value_type>* reg );
    
    void backward(network_stage stage, cublasHandle_t* cublas_h, cudnnHandle_t* cudnn_h, registry_t<value_type>* reg );

	void get_loss(LOSS_MODE loss_mode, std::vector<value_type>* loss);

    void gen_description(char* buff, size_t* len_in_byte) {
        this->gen_meta_description(buff, len_in_byte);
    }

	void fake_run(net_comp dir, registry_t<value_type>* reg);
};
    
} //ATP namespace

#endif



