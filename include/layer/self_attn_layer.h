
#if !defined(_CUDNN_SATTN_H_)
#define _CUDNN_SATTN_H_
#include <switch.h>
#include <util/common.h>
#include <layer/base_network_layer.h>

namespace ATP{

template <class value_type>
class self_attn_layer_t:base_network_layer_t<value_type>
{

private:
    size_t embed_dim;
    size_t num_head;
    double sm_scaler;
    double dropout_rate;
    bool enable_bias;
	bool add_bias_kv; 
    bool add_zero_attn; 
    size_t kdim; 
    size_t vdim;
    bool batch_first;
    bool residual;

    int qSize, kSize, vSize;
    int qProjSize, kProjSize, vProjSize, oProjSize;
    int qoMaxSeqLength, kvMaxSeqLength, maxBatchSize, maxBeamSize;

    size_t seqLength, miniBatch, beamDim, embedSize;

    cudnnDataType_t dataType;
    cudnnDataType_t computePrec;
    cudnnMathType_t mathType;

    unsigned attnMode;

	cudnnAttnDescriptor_t attnDesc;
	int *loWinIdx;
	int *hiWinIdx;
    size_t SeqLengthsQOSize;
    size_t SeqLengthsKVSize; 
    int* SeqLengthsQO;
    int* SeqLengthsKV;
	int *devSeqLengthsQO;
	int *devSeqLengthsKV;
	cudnnSeqDataDescriptor_t qDesc;
    tensor_t<value_type>* Q;
    tensor_t<value_type>* dQ;
    tensor_t<value_type>* residuals;
	cudnnSeqDataDescriptor_t kDesc;
	tensor_t<value_type>* K;
    tensor_t<value_type>* dK;
	cudnnSeqDataDescriptor_t vDesc;
	tensor_t<value_type>* V;
    tensor_t<value_type>* dV;
	cudnnSeqDataDescriptor_t oDesc;
    tensor_t<value_type>* O;
    tensor_t<value_type>* dO;
    tensor_t<value_type>* weights;
    tensor_t<value_type>* dweights;
    tensor_t<value_type>* work_space;
    tensor_t<value_type>* reserve_buff;
    size_t weightSizeInBytes;
	size_t workSpaceSizeInBytes;
	size_t reserveSpaceSizeInBytes;
    void *reserveBuff = NULL;

    initializer_t<value_type> *weight_initializer;
    initializer_t<value_type> *bias_initializer;

	float attn_dropout_rate;
	tensor_t<value_type>* attn_dropout_state;
    cudnnDropoutDescriptor_t attnDropoutDesc;
    size_t attn_dropout_state_size_bytes;

	float post_dropout_rate;
	tensor_t<value_type>* post_dropout_state;
    cudnnDropoutDescriptor_t postDropoutDesc;
    size_t post_dropout_state_size_bytes;

    void createDesc() {
        checkCUDNN( cudnnCreateAttnDescriptor(&(this->attnDesc)));
        checkCUDNN( cudnnCreateSeqDataDescriptor(&(this->qDesc)));
        checkCUDNN( cudnnCreateSeqDataDescriptor(&(this->kDesc)));
        checkCUDNN( cudnnCreateSeqDataDescriptor(&(this->vDesc)));
        checkCUDNN( cudnnCreateSeqDataDescriptor(&(this->oDesc)));

        checkCUDNN( cudnnCreateDropoutDescriptor(&(this->attnDropoutDesc)));
        checkCUDNN( cudnnCreateDropoutDescriptor(&(this->postDropoutDesc)));
    }

public:
	self_attn_layer_t(size_t embed_dim, size_t num_head, size_t seqLength, size_t miniBatch, size_t beamDim, size_t embedSize, 
        // unsigned attn_mode=CUDNN_ATTN_QUERYMAP_ONE_TO_ONE,
        unsigned attn_mode=CUDNN_ATTN_DISABLE_PROJ_BIASES,
        double sm_scaler=1.0, double attn_dropout_rate=0.0, double post_dropout_rate=0.0, 
        initializer_t<value_type> *weight_initializer = new gaussian_initializer_t<float>(0, 0.01),
		initializer_t<value_type> *bias_initializer = new constant_initializer_t<float>(0.0),
        int qproj_size = -1, int kproj_size = -1, int vproj_size = -1, int oproj_size = -1, 
        bool bias=false, bool add_bias_kv=false, bool add_zero_attn=false, bool residual=true,
        size_t kdim=0, size_t vdim=0,
        bool batch_first=false) 
		: embed_dim(embed_dim), num_head(num_head), 
        seqLength(seqLength), miniBatch(miniBatch), beamDim(beamDim), embedSize(embedSize),
        attnMode(attn_mode), 
        sm_scaler(sm_scaler), attn_dropout_rate(attn_dropout_rate), post_dropout_rate(post_dropout_rate), 
        enable_bias(bias), add_bias_kv(add_bias_kv), add_zero_attn(add_zero_attn), kdim(kdim), vdim(vdim),
		batch_first(batch_first), residual(residual),
        weight_initializer(weight_initializer), bias_initializer(bias_initializer),
		base_network_layer_t<value_type>(SATTN)
    {
        createDesc();

		switch (sizeof(value_type))
        {
            case 2: dataType = CUDNN_DATA_HALF;   break;
            case 4: dataType = CUDNN_DATA_FLOAT;  break;
            case 8: dataType = CUDNN_DATA_DOUBLE; break;
            default : FatalError("Unsupported data type");
        }
        computePrec = dataType;
        mathType = CUDNN_DEFAULT_MATH;

        this->qProjSize = qproj_size == -1 ? embed_dim : qproj_size; 
        assert(this->qProjSize > 0);
        this->kProjSize = kproj_size == -1 ? embed_dim : kproj_size; 
        assert(this->kProjSize > 0);
        this->vProjSize = vproj_size == -1 ? embed_dim : vproj_size; 
        assert(this->vProjSize > 0);
        this->oProjSize = oproj_size == -1 ? embed_dim : oproj_size; 
        assert(this->oProjSize > 0);

        this->qSize = embed_dim;
        this->kSize = embed_dim;
        this->vSize = embed_dim;
        
    }
    
    ~self_attn_layer_t() {
        checkCUDNN( cudnnDestroyAttnDescriptor(this->attnDesc) );
		checkCUDNN( cudnnDestroySeqDataDescriptor(this->qDesc) );
		checkCUDNN( cudnnDestroySeqDataDescriptor((this->kDesc)) );
        checkCUDNN( cudnnDestroySeqDataDescriptor(this->vDesc) );
		checkCUDNN( cudnnDestroySeqDataDescriptor((this->oDesc)) );
    }   

    void set_input_size(size_t seqLength, size_t miniBatch, size_t beamDim, size_t embedSize) {
        this->seqLength = seqLength;
        this->miniBatch = miniBatch;
        this->beamDim = beamDim;
        this->embedSize = embedSize;
    }

    tensor_t<value_type>* get_attn_reserve_buff() {
		return this->reserve_buff;
	}
	
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

        value_type _m = attnMode;
        memcpy(buff+meta_len_in_byte, &_m, sizeof(value_type));

        *len_in_byte = meta_len_in_byte + sizeof(value_type);
    }

    void fake_run(net_comp dir, registry_t<value_type>* reg);

    tensor_t<value_type>* get_QKV(net_comp dir, int id) {
        if (dir == FORWARD) {
            switch (id) {
                case 0: return this->Q;
                case 1: return this->K;
                case 2: return this->V;
                default: printf("illegal id %d\n", id); exit(0);
            }
        }
        else {
            switch (id) {
                case 0: return this->dQ;
                case 1: return this->dK;
                case 2: return this->dV;
                default: printf("illegal id %d\n", id); exit(0);
            }
        }
    }

    tensor_t<value_type>* set_QKV(net_comp dir, int id, value_type* ptr) {
        if (dir == FORWARD) {
            switch (id) {
                case 0: this->Q->set_gpu_ptr(ptr); break;
                case 1: this->K->set_gpu_ptr(ptr); break;
                case 2: this->V->set_gpu_ptr(ptr); break;
                default: printf("illegal id %d\n", id); exit(0);
            }
        }
        else {
            switch (id) {
                case 0: this->dQ->set_gpu_ptr(ptr); break;
                case 1: this->dK->set_gpu_ptr(ptr); break;
                case 2: this->dV->set_gpu_ptr(ptr); break;
                default: printf("illegal id %d\n", id); exit(0);
            }
        }
    }

};

} // ATP namespace
#endif // _CUDNN_SATTN_H_