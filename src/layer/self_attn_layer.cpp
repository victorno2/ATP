
#include <layer/self_attn_layer.h>
#include <util/mem_util.h>
#include <limits>
#include <cudnn.h>

namespace ATP {

template <class value_type>
void self_attn_layer_t<value_type>::forward_setup( registry_t<value_type>* reg, cudnnHandle_t* cudnn_h ) {
    printf("======>setup the forward self_attn_layer:%d\n", this->get_id());
    int curt_l  = this->get_id();
    int input_l = this->get_input_layer_id();
    tensor_t<value_type>* x = reg->get_reg_output(input_l, curt_l);
    x->increase_use_counter(FORWARD);
	this->set_f_in(x, reg);

    // in self-attention, Q = K = V, Q* = x * W_Q, K* = x * W_K, V* = x * W_V
    // this->Q = x;
    // this->K = x;
    // this->V = x;
    x->set_data_layout(SEQ2SEQ_TNBV);
    x->reshape(this->seqLength, this->miniBatch, this->beamDim, this->embedSize);
    this->Q = new tensor_t<value_type>( x->get_seq_time(), x->get_seq_batch(), x->get_seq_beam(), x->get_seq_vect(), reg->get_vector(), QKV_DATA, this->get_id());
    this->K = new tensor_t<value_type>( x->get_seq_time(), x->get_seq_batch(), x->get_seq_beam(), x->get_seq_vect(), reg->get_vector(), QKV_DATA, this->get_id());
    this->V = new tensor_t<value_type>( x->get_seq_time(), x->get_seq_batch(), x->get_seq_beam(), x->get_seq_vect(), reg->get_vector(), QKV_DATA, this->get_id());

    this->O = new tensor_t<value_type>( Q->get_seq_time(), Q->get_seq_batch(), Q->get_seq_beam(), this->vProjSize, reg->get_vector(), DATA, this->get_id());
    this->O->set_data_layout(SEQ2SEQ_TNBV);
    printf("O_size_bytes = %d\n", this->O->get_mem_size());
    this->set_f_out( this->O, reg );
    assert( this->get_f_out()  != NULL );

    
    // checkCUDNN(cudnnDropoutGetStatesSize(*cudnn_h, &attn_dropout_state_size_bytes));
    // printf("attn_dropout_state_size_bytes = %d\n", attn_dropout_state_size_bytes);
    // this->attn_dropout_state = new tensor_t<value_type>( attn_dropout_state_size_bytes / sizeof(value_type), 1, 1, 1, reg->get_vector(), RNN_BUFF, this->get_id());
    // checkCUDNN( cudnnSetDropoutDescriptor(attnDropoutDesc, *cudnn_h, this->attn_dropout_rate, this->attn_dropout_state->get_gpu_ptr(), attn_dropout_state_size_bytes, 0));
    // printf("dropoutRate=%f, dropoutBufSize=%d\n", this->attn_dropout_rate, attn_dropout_state_size_bytes);

    // checkCUDNN(cudnnDropoutGetStatesSize(*cudnn_h, &post_dropout_state_size_bytes));
    // printf("post_dropout_state_size_bytes = %d\n", post_dropout_state_size_bytes);
    // this->post_dropout_state = new tensor_t<value_type>( post_dropout_state_size_bytes / sizeof(value_type), 1, 1, 1, reg->get_vector(), RNN_BUFF, this->get_id());
    // checkCUDNN( cudnnSetDropoutDescriptor(postDropoutDesc, *cudnn_h, this->post_dropout_rate, (this->post_dropout_state)->get_gpu_ptr(), post_dropout_state_size_bytes, 0));
    // printf("dropoutRate=%f, dropoutBufSize=%d\n", this->post_dropout_rate, post_dropout_state_size_bytes);

    // Set the suported tensor layout call seq2seq, with four dim.
    int nbDims = 4;
    int dimA[CUDNN_SEQDATA_DIM_COUNT];
    dimA[CUDNN_SEQDATA_TIME_DIM] = x->get_seq_time();  // the max sequence legth, each sequence legth must less than or equal to dimA[CUDNN_SEQDATA_TIME_DIM]
    dimA[CUDNN_SEQDATA_BATCH_DIM] = x->get_seq_batch();  // batch size
    dimA[CUDNN_SEQDATA_BEAM_DIM] = x->get_seq_beam();  // beam size
    dimA[CUDNN_SEQDATA_VECT_DIM] = x->get_seq_vect();  // embedding size
    cudnnSeqDataAxis_t axes[CUDNN_SEQDATA_DIM_COUNT];  // define sequence data layout
    axes[0] = CUDNN_SEQDATA_TIME_DIM;
    axes[1] = CUDNN_SEQDATA_BATCH_DIM;
    axes[2] = CUDNN_SEQDATA_BEAM_DIM;
    axes[3] = CUDNN_SEQDATA_VECT_DIM;  // 3 = nbDims-1
    
    size_t seqLengthArraySize = dimA[CUDNN_SEQDATA_BATCH_DIM] * dimA[CUDNN_SEQDATA_BEAM_DIM];
    int *seqLengthArray = new int[seqLengthArraySize];
    for (size_t i = 0; i < seqLengthArraySize; i++) {
        seqLengthArray[i] = dimA[CUDNN_SEQDATA_TIME_DIM];
    }

    int *qSeqArray = (int *)calloc(x->get_seq_batch() * x->get_seq_beam(), sizeof(int));
    int *kSeqArray = (int *)calloc(x->get_seq_batch(), sizeof(int));
    for (size_t i = 0; i < (x->get_seq_batch() * x->get_seq_beam()); ++i) {
        qSeqArray[i] = this->Q->get_seq_time();
    }
    for (size_t i = 0; i < x->get_seq_batch(); ++i) {
        kSeqArray[i] = this->K->get_seq_time();
    }
    int qSeqArraySize = x->get_seq_beam() * x->get_seq_batch();
    int kSeqArraySize = x->get_seq_batch();
    size_t size = sizeof(qSeqArray[0]) * qSeqArraySize;
    checkCudaErrors(cudaMalloc((void **)&devSeqLengthsQO, size));
    checkCudaErrors(cudaMemcpy(devSeqLengthsQO, qSeqArray, size, cudaMemcpyHostToDevice));
    size = sizeof(kSeqArray[0]) * kSeqArraySize;
    checkCudaErrors(cudaMalloc((void **)&devSeqLengthsKV, size));
    checkCudaErrors(cudaMemcpy(devSeqLengthsKV, kSeqArray, size, cudaMemcpyHostToDevice));


    axes[0] = CUDNN_SEQDATA_TIME_DIM;
    axes[1] = CUDNN_SEQDATA_BATCH_DIM;
    axes[2] = CUDNN_SEQDATA_BEAM_DIM;
    axes[3] = CUDNN_SEQDATA_VECT_DIM;  // 3 = nbDims-1

    dimA[CUDNN_SEQDATA_TIME_DIM] = Q->get_seq_time();  // the max sequence legth, each sequence legth must less than or equal to dimA[CUDNN_SEQDATA_TIME_DIM]
    dimA[CUDNN_SEQDATA_BATCH_DIM] = Q->get_seq_batch();  // batch size
    dimA[CUDNN_SEQDATA_BEAM_DIM] = Q->get_seq_beam();  // beam size
    dimA[CUDNN_SEQDATA_VECT_DIM] = Q->get_seq_vect();  // embedding size
    checkCUDNN( cudnnSetSeqDataDescriptor(qDesc, this->dataType, CUDNN_SEQDATA_DIM_COUNT, dimA, axes, qSeqArraySize, qSeqArray, NULL) );
    // printf("Q dataType=%d, dimA[0]=%d, dimA[1]=%d, dimA[2]=%d, dimA[3]=%d, seqLengthArraySize=%d, seqLengthArray[0]=%d, seqLengthArray[1]=%d\n", 
    //         dataType, dimA[0], dimA[1], dimA[2], dimA[3], seqLengthArraySize, seqLengthArray[0], seqLengthArray[1]);

    dimA[CUDNN_SEQDATA_TIME_DIM] = O->get_seq_time();  // the max sequence legth, each sequence legth must less than or equal to dimA[CUDNN_SEQDATA_TIME_DIM]
    dimA[CUDNN_SEQDATA_BATCH_DIM] = O->get_seq_batch();  // batch size
    dimA[CUDNN_SEQDATA_BEAM_DIM] = O->get_seq_beam();  // beam size
    dimA[CUDNN_SEQDATA_VECT_DIM] = O->get_seq_vect();  // embedding size
    checkCUDNN( cudnnSetSeqDataDescriptor(oDesc, this->dataType, CUDNN_SEQDATA_DIM_COUNT, dimA, axes, qSeqArraySize, qSeqArray, NULL) );
    // printf("O dataType=%d, dimA[0]=%d, dimA[1]=%d, dimA[2]=%d, dimA[3]=%d, seqLengthArraySize=%d, seqLengthArray[0]=%d, seqLengthArray[1]=%d\n", 
    //         dataType, dimA[0], dimA[1], dimA[2], dimA[3], seqLengthArraySize, seqLengthArray[0], seqLengthArray[1]);

    dimA[CUDNN_SEQDATA_TIME_DIM] = K->get_seq_time();  // the max sequence legth, each sequence legth must less than or equal to dimA[CUDNN_SEQDATA_TIME_DIM]
    dimA[CUDNN_SEQDATA_BATCH_DIM] = K->get_seq_batch();  // batch size
    dimA[CUDNN_SEQDATA_BEAM_DIM] = K->get_seq_beam();  // beam size
    dimA[CUDNN_SEQDATA_VECT_DIM] = K->get_seq_vect();  // embedding size
    checkCUDNN( cudnnSetSeqDataDescriptor(kDesc, this->dataType, CUDNN_SEQDATA_DIM_COUNT, dimA, axes, kSeqArraySize, kSeqArray, NULL) );
    // printf("K dataType=%d, dimA[0]=%d, dimA[1]=%d, dimA[2]=%d, dimA[3]=%d, seqLengthArraySize=%d, seqLengthArray[0]=%d, seqLengthArray[1]=%d\n", 
    //         dataType, dimA[0], dimA[1], dimA[2], dimA[3], seqLengthArraySize, seqLengthArray[0], seqLengthArray[1]);

    dimA[CUDNN_SEQDATA_TIME_DIM] = V->get_seq_time();  // the max sequence legth, each sequence legth must less than or equal to dimA[CUDNN_SEQDATA_TIME_DIM]
    dimA[CUDNN_SEQDATA_BATCH_DIM] = V->get_seq_batch();  // batch size
    dimA[CUDNN_SEQDATA_BEAM_DIM] = V->get_seq_beam();  // beam size
    dimA[CUDNN_SEQDATA_VECT_DIM] = V->get_seq_vect();  // embedding size
    checkCUDNN( cudnnSetSeqDataDescriptor(vDesc, this->dataType, CUDNN_SEQDATA_DIM_COUNT, dimA, axes, kSeqArraySize, kSeqArray, NULL) );
    // printf("V dataType=%d, dimA[0]=%d, dimA[1]=%d, dimA[2]=%d, dimA[3]=%d, seqLengthArraySize=%d, seqLengthArray[0]=%d, seqLengthArray[1]=%d\n", 
    //         dataType, dimA[0], dimA[1], dimA[2], dimA[3], seqLengthArraySize, seqLengthArray[0], seqLengthArray[1]);

    this->qoMaxSeqLength = Q->get_seq_time();
    this->kvMaxSeqLength = K->get_seq_time();
    this->maxBatchSize = Q->get_seq_batch();
    this->maxBeamSize = Q->get_seq_beam();

    // printf("qSize=%zd, kSize=%zd, vSize=%zd\n", qSize, kSize, vSize);
    // // qProjSize = 32; kProjSize = 32; vProjSize = 32; oProjSize = 32; 
    // printf("qProjSize=%zd, kProjSize=%zd, vProjSize=%zd, oProjSize=%zd\n", qProjSize, kProjSize, vProjSize, oProjSize);
    // // qoMaxSeqLength = 32; kvMaxSeqLength = 32;
    // printf("qoMaxSeqLength=%zd, kvMaxSeqLength=%zd\n", qoMaxSeqLength, kvMaxSeqLength);
    // // maxBatchSize = 1; maxBeamSize = 1;
    // printf("maxBatchSize=%zd, maxBeamSize=%zd\n", maxBatchSize, maxBeamSize);

    // this->loWinIdx = (int *)calloc(Q->get_seq_time(), sizeof(int));
    // this->hiWinIdx = (int *)calloc(Q->get_seq_time(), sizeof(int));
    this->loWinIdx = (int *)calloc(qoMaxSeqLength, sizeof(int));
    this->hiWinIdx = (int *)calloc(qoMaxSeqLength, sizeof(int));
    for (int i = 0; i < qoMaxSeqLength; ++i) {
        loWinIdx[i] = 0;
        // hiWinIdx[i] = kvMaxSeqLength;
        hiWinIdx[i] = INT_MAX;
    }
    printf("INT_MAX = %zd\n", INT_MAX);
    // SeqLengthsQOSize = dimA[CUDNN_SEQDATA_BATCH_DIM] * dimA[CUDNN_SEQDATA_BEAM_DIM];
    // SeqLengthsKVSize = dimA[CUDNN_SEQDATA_BATCH_DIM];
    // SeqLengthsQO = new int[SeqLengthsQOSize];
    // SeqLengthsKV = new int[SeqLengthsKVSize];
    // checkCudaErrors( cudaMalloc((void**)&devSeqLengthsQO, SeqLengthsQOSize*sizeof(int) ) );
    // checkCudaErrors( cudaMemcpy((void*)devSeqLengthsQO, (void*)SeqLengthsQO, SeqLengthsQOSize*sizeof(int), cudaMemcpyHostToDevice) );
    // checkCudaErrors( cudaMalloc((void**)&devSeqLengthsKV, SeqLengthsKVSize*sizeof(int) ) );
    // checkCudaErrors( cudaMemcpy((void*)devSeqLengthsKV, (void*)SeqLengthsKV, SeqLengthsKVSize*sizeof(int), cudaMemcpyHostToDevice) );
    // printf("devSeqLengthsQO size = %d, devSeqLengthsQO = ", SeqLengthsQOSize);
    // for (size_t i = 0; i < SeqLengthsQOSize; i++) {
    //     SeqLengthsQO[i] = qoMaxSeqLength;
    //     printf("%d ", SeqLengthsQO[i]);
    // }
    // printf("\n");
    // printf("devSeqLengthsKV size = %d, devSeqLengthsKV = ", SeqLengthsKVSize);
    // for (size_t i = 0; i < SeqLengthsKVSize; i++) {
    //     SeqLengthsKV[i] = kvMaxSeqLength;
    //     printf("%d ", SeqLengthsKV[i]);
    // }
    // printf("\n");


    // printf("queryMap=%d\n numHeads=%d\n smScaler=%f\n dataType=%d\n compPrec=%d\n qSize=%d\n kSize=%d\n vSize=%d\n qProjSize=%d\n kProjSize=%d\n vProjSize=%d\n oProjSize=%d\n seqLenQ=%d\n seqLenK=%d\n batchSize=%d\n beamSize=%d\n",
    //     attnMode, 12, sm_scaler, dataType, computePrec, qSize, kSize, vSize,
    //     qProjSize, kProjSize, vProjSize, oProjSize, qoMaxSeqLength, kvMaxSeqLength, maxBatchSize, maxBeamSize);

    checkCUDNN( cudnnSetAttnDescriptor( this->attnDesc,
                                        this->attnMode,
                                        num_head,  // this->num_head,
                                        this->sm_scaler,
                                        this->dataType,
                                        this->computePrec,
                                        CUDNN_DEFAULT_MATH,  // this->mathType,
                                        NULL,  // this->attnDropoutDesc,
                                        NULL,  // this->postDropoutDesc,
                                        this->qSize,
                                        this->kSize,
                                        this->vSize,
                                        this->qProjSize,
                                        this->kProjSize,
                                        this->vProjSize,
                                        this->oProjSize,
                                        this->qoMaxSeqLength,
                                        this->kvMaxSeqLength,
                                        this->maxBatchSize,
                                        this->maxBeamSize) );

    // devSeqLengthsQO[] and devSeqLengthsKV[] recoded SeqLengths of Q, O, K and V, the two arrays are in GPU
    
    checkCUDNN( cudnnGetMultiHeadAttnBuffers(*cudnn_h, this->attnDesc, &weightSizeInBytes, &workSpaceSizeInBytes, &reserveSpaceSizeInBytes) );
    printf("weightSizeInBytes=%zd, workSpaceSizeInBytes=%zd, reserveSpaceSizeInBytes=%zd\n", weightSizeInBytes, workSpaceSizeInBytes, reserveSpaceSizeInBytes);

    this->weights = new tensor_t<value_type>( weightSizeInBytes/sizeof(value_type), 1, 1, 1, reg->get_vector(), PARAM, this->get_id());
    this->set_weight(this->weights, reg);
    this->weights->init(this->weight_initializer);
    printf("weights_size = %zd\n", this->weights->get_mem_size());

    this->work_space = new tensor_t<value_type>( workSpaceSizeInBytes/sizeof(value_type), 1, 1, 1, reg->get_vector(), RNN_BUFF, this->get_id());
    printf("self-attn work_space_size = %zd\n", work_space->get_mem_size());

    this->reserve_buff = new tensor_t<value_type>( reserveSpaceSizeInBytes/sizeof(value_type), 1, 1, 1, reg->get_vector(), RNN_RESERVE, this->get_id());
    printf("self-attn reserve_buff_size = %zd\n", reserve_buff->get_mem_size());
    this->set_reserve_buff(reserve_buff);

    reg->register_forward_dependency( this->get_id(), x );
    reg->register_forward_dependency( this->get_id(), this->Q );
    reg->register_forward_dependency( this->get_id(), this->K );
    reg->register_forward_dependency( this->get_id(), this->V );
    // reg->register_forward_dependency( this->get_id(), this->residuals );
    reg->register_forward_dependency( this->get_id(), this->O );
    reg->register_forward_dependency( this->get_id(), this->weights );
    reg->register_backward_dependency( this->get_id(), this->work_space );
    reg->register_forward_dependency( this->get_id(), this->reserve_buff );
    // reg->register_forward_dependency( this->get_id(), this->post_dropout_state );
    // reg->register_forward_dependency( this->get_id(), this->attn_dropout_state );
}
    
template <class value_type>
void self_attn_layer_t<value_type>::backward_setup( registry_t<value_type>* reg, cudnnHandle_t* cudnn_h ) {
    //backward
    int curt_l_id   = this->get_id();
    int input_l_id  = this->get_input_layer_id();
    int output_l_id = this->get_output_layer_id();
    this->dO = reg->get_reg_b_data(output_l_id, curt_l_id);
    this->set_dy(dO);
    this->dO->increase_b_use_count();
    tensor_t<value_type>* x = reg->get_reg_output(input_l_id, curt_l_id);
    x->increase_use_counter(BACKWARD);

    printf("======>setup the backward self_attn_layer%d\n", this->get_id());
    this->dO->reshape(seqLength, miniBatch, beamDim, embedSize);
    this->dO->set_data_layout(SEQ2SEQ_TNBV);
    this->dQ = new tensor_t<value_type>( Q->get_seq_time(), Q->get_seq_batch(), Q->get_seq_beam(), Q->get_seq_vect(), reg->get_vector(), DQKV_DATA, this->get_id());
    this->dK = new tensor_t<value_type>( K->get_seq_time(), K->get_seq_batch(), K->get_seq_beam(), K->get_seq_vect(), reg->get_vector(), DQKV_DATA, this->get_id());
    this->dV = new tensor_t<value_type>( V->get_seq_time(), V->get_seq_batch(), V->get_seq_beam(), V->get_seq_vect(), reg->get_vector(), DQKV_DATA, this->get_id());
    
    tensor_t<value_type>* dx = new tensor_t<value_type>( Q->get_seq_time(), Q->get_seq_batch(), Q->get_seq_beam(), Q->get_seq_vect(), reg->get_vector(), B_DATA, this->get_id());
    dx->set_data_layout(SEQ2SEQ_TNBV);
    this->set_b_data(dx, reg);

    this->dweights = new tensor_t<value_type>( this->weights->get_mem_size()/sizeof(value_type), 1, 1, 1, reg->get_vector(), GRAD, this->get_id());
    this->set_weight_grad(dweights, reg);
    dweights->init(this->weight_initializer); 
    printf("dweights size = %d\n", this->dweights->get_mem_size()/sizeof(value_type));

    reg->register_backward_dependency( this->get_id(), this->Q ); 
    reg->register_backward_dependency( this->get_id(), this->dQ );
    reg->register_backward_dependency( this->get_id(), this->K );
    reg->register_backward_dependency( this->get_id(), this->dK );
    reg->register_backward_dependency( this->get_id(), this->V );
    reg->register_backward_dependency( this->get_id(), this->dV );
    reg->register_backward_dependency( this->get_id(), this->O );
    reg->register_backward_dependency( this->get_id(), this->dO );
    reg->register_backward_dependency( this->get_id(), x );
    reg->register_backward_dependency( this->get_id(), dx ); 
    reg->register_backward_dependency( this->get_id(), this->dweights );   
    reg->register_backward_dependency( this->get_id(), this->weights );
    reg->register_backward_dependency( this->get_id(), this->weights );
    // reg->register_forward_dependency( this->get_id(), this->attn_dropout_state );
    // reg->register_forward_dependency( this->get_id(), this->post_dropout_state );
    reg->register_forward_dependency( this->get_id(), this->work_space );
    reg->register_forward_dependency( this->get_id(), this->reserve_buff );
    // reg->register_forward_dependency( this->get_id(), this->residuals );
}

template <class value_type>
std::vector<value_type> self_attn_layer_t<value_type>::forward(network_stage stage, cublasHandle_t* cublas_h, cudnnHandle_t* cudnn_h, registry_t<value_type>* reg) {
    assert( cudnn_h != NULL );
    assert( reg     != NULL );

    int currIdx = -1;  // only support training mode, cudnnMultiHeadAttnForward() will automatically sweep through all Q time-steps.

    int curt_l  = this->get_id();
    int input_l = this->get_input_layer_id();
    tensor_t<value_type>* x = reg->get_reg_output(input_l, curt_l);
    x->increase_cur_use_counter(FORWARD);

    // for (int i = 0; i < SeqLengthsQOSize; ++i) {
    //     printf("attention_window[time=%d]=%d:%d\n", i, loWinIdx[i], hiWinIdx[i]);
    // }
    // printf("\n");

    // for (size_t i = 0; i < SeqLengthsQOSize; ++i) {
    //     printf("sequence_length_q[idx=%lu]=%d\n", i, SeqLengthsQO[i]);
    // }
    // printf("\n");

    // for (size_t i = 0; i < SeqLengthsQOSize; ++i) {
    //     printf("sequence_length_k[idx=%lu]=%d\n", i, SeqLengthsKV[i]);
    // }
    // printf("\n\nforward layer%d cudnnMultiHeadAttnForward\n", this->get_id());
    this->Q->copy(x);
    this->K->copy(x);
    this->V->copy(x);
    // this->Q->printTensorData("before cudnnMultiHeadAttnForward Q", 2);
    // this->K->printTensorData("before cudnnMultiHeadAttnForward K", 2);
    // this->V->printTensorData("before cudnnMultiHeadAttnForward V", 2);
    // this->weights->printTensorData("before cudnnMultiHeadAttnForward W", 2);
    // this->get_f_out()->printTensorData("before cudnnMultiHeadAttnForward output", 2);
    // printf("sizeWeights=%d, sizeWkspace=%d, sizeReserve=%d\n", weightSizeInBytes, workSpaceSizeInBytes, reserveSpaceSizeInBytes);
    // this->work_space->set_gpu_ptr(NULL);
    // printf("weights=%x, work_space=%x, reserve_buff=%x\n", this->weights->get_gpu_ptr(), this->work_space->get_gpu_ptr(), this->reserve_buff->get_gpu_ptr());
    // this->reserve_buff->printTensorData("before cudnnMultiHeadAttnForward  reserve_buff", 2);
    // checkCUDNN( cudnnSetDropoutDescriptor(attnDropoutDesc, *cudnn_h, this->attn_dropout_rate, this->attn_dropout_state->get_gpu_ptr(), attn_dropout_state_size_bytes, 0));
    // checkCUDNN( cudnnSetDropoutDescriptor(postDropoutDesc, *cudnn_h, this->post_dropout_rate, this->post_dropout_state->get_gpu_ptr(), post_dropout_state_size_bytes, 0));
    // printf("queryMap=%d\n numHeads=%d\n smScaler=%f\n dataType=%d\n computePrec=%d\n qSize=%d\n kSize=%d\n vSize=%d\n qProjSize=%d\n kProjSize=%d\n vProjSize=%d\n oProjSize=%d\n seqLenQ=%d\n seqLenK=%d\n batchSize=%d\n beamSize=%d\n",
    //     attnMode, num_head, sm_scaler, dataType, computePrec, qSize, kSize, vSize,
    //     qProjSize, kProjSize, vProjSize, oProjSize, qoMaxSeqLength, kvMaxSeqLength, maxBatchSize, maxBeamSize);
    
    checkCUDNN( cudnnSetAttnDescriptor( this->attnDesc,
                                        attnMode, // CUDNN_ATTN_QUERYMAP_ONE_TO_ONE,  // this->attnMode,
                                        num_head,  // this->num_head,
                                        this->sm_scaler,
                                        this->dataType,
                                        this->computePrec,
                                        CUDNN_DEFAULT_MATH,  // this->mathType,
                                        NULL,  // this->attnDropoutDesc,
                                        NULL,  // this->postDropoutDesc,
                                        this->qSize,
                                        this->kSize,
                                        this->vSize,
                                        this->qProjSize,
                                        this->kProjSize,
                                        this->vProjSize,
                                        this->oProjSize,
                                        this->qoMaxSeqLength,
                                        this->kvMaxSeqLength,
                                        this->maxBatchSize,
                                        this->maxBeamSize) );

    checkCUDNN( cudnnMultiHeadAttnForward(
                                        *cudnn_h,
                                        this->attnDesc,
                                        -1, //currIdx,
                                        this->loWinIdx,
                                        this->hiWinIdx,
                                        this->devSeqLengthsQO,
                                        this->devSeqLengthsKV,
                                        this->qDesc,
                                        this->Q->get_gpu_ptr(),  // const void *queries,
                                        this->Q->get_gpu_ptr(),  //this->residuals->get_gpu_ptr(),  // this->Q->get_gpu_ptr(),  // this->residuals->get_gpu_ptr(),  // const void *residuals,
                                        this->kDesc,
                                        this->K->get_gpu_ptr(),  // const void *keys,
                                        this->vDesc,
                                        this->V->get_gpu_ptr(),  // const void *values,
                                        this->oDesc,
                                        this->O->get_gpu_ptr(),
                                        weightSizeInBytes,
                                        this->weights->get_gpu_ptr(),
                                        workSpaceSizeInBytes,
                                        NULL,  // this->work_space->get_gpu_ptr(),
                                        reserveSpaceSizeInBytes,
                                        this->reserve_buff->get_gpu_ptr()) );

    // this->Q->printTensorData("after cudnnMultiHeadAttnForward Q", 2);
    // this->K->printTensorData("after cudnnMultiHeadAttnForward K", 2);
    // this->V->printTensorData("after cudnnMultiHeadAttnForward V", 2);
    // this->get_f_out()->printTensorData("after cudnnMultiHeadAttnForward output", 2);
    // this->reserve_buff->printTensorData("after cudnnMultiHeadAttnForward reserve_buff", 2);

    #ifdef DEBUG
    this->get_f_out()->printTensor("cudnnMultiHeadAttnForward output");
    //    this->get_f_out()->GPUtoCPU();
    #endif
    return std::vector<value_type>();
}
    
template <class value_type>
void self_attn_layer_t<value_type>::backward(network_stage stage, cublasHandle_t* cublas_h, cudnnHandle_t* cudnn_h, registry_t<value_type>* reg ) {
    assert( cudnn_h != NULL );
    assert( reg     != NULL );
    
    int curt_l_id   = this->get_id();
    int input_l_id  = this->get_input_layer_id();
    int output_l_id = this->get_output_layer_id();
    this->dO = reg->get_reg_b_data(output_l_id, curt_l_id);
    tensor_t<value_type>* x = reg->get_reg_output(input_l_id, curt_l_id);
    // checkCudaErrors( cudaMalloc((void**)&devSeqLengthsQO, SeqLengthsQOSize*sizeof(int) ) );
    // checkCudaErrors( cudaMemcpy((void*)devSeqLengthsQO, (void*)SeqLengthsQO, SeqLengthsQOSize*sizeof(int), cudaMemcpyHostToDevice) );
    // checkCudaErrors( cudaMalloc((void**)&devSeqLengthsKV, SeqLengthsKVSize*sizeof(int) ) );
    // checkCudaErrors( cudaMemcpy((void*)devSeqLengthsKV, (void*)SeqLengthsKV, SeqLengthsKVSize*sizeof(int), cudaMemcpyHostToDevice) );

    // this->Q->printTensorData("before cudnnMultiHeadAttnBackwardData Q", 2);
    // this->K->printTensorData("before cudnnMultiHeadAttnBackwardData K", 2);
    // this->V->printTensorData("before cudnnMultiHeadAttnBackwardData V", 2);
    // this->dO->printTensorData("before cudnnMultiHeadAttnBackwardData dO", 2);
    // this->dQ->printTensorData("before cudnnMultiHeadAttnBackwardData dQ", 2);
    // this->dV->printTensorData("before cudnnMultiHeadAttnBackwardData dV", 2);
    // this->dK->printTensorData("before cudnnMultiHeadAttnBackwardData dK", 2);
    // this->weights->printTensorData("before cudnnMultiHeadAttnBackwardData weights", 2);
    // this->reserve_buff->printTensorData("before cudnnMultiHeadAttnBackwardData reserve_buff", 2);
    this->Q->copy(x);
    this->K->copy(x);
    this->V->copy(x);
    checkCUDNN( cudnnMultiHeadAttnBackwardData(
                                            *cudnn_h,
                                            this->attnDesc,
                                            this->loWinIdx,
                                            this->hiWinIdx,
                                            this->devSeqLengthsQO,
                                            this->devSeqLengthsKV,
                                            this->oDesc,
                                            this->dO->get_gpu_ptr(),  // devDO, //   // const void *dout,
                                            this->qDesc,  // const cudnnSeqDataDescriptor_t dqDesc,
                                            this->dQ->get_gpu_ptr(),  // devDQ, //  // void *dqueries,
                                            this->Q->get_gpu_ptr(),  // const void *queries,
                                            this->kDesc,  // const cudnnSeqDataDescriptor_t dkDesc,
                                            this->dK->get_gpu_ptr(),  // devDK, //  // void *dkeys,
                                            this->K->get_gpu_ptr(),  // const void *keys,
                                            this->vDesc,  // const cudnnSeqDataDescriptor_t dvDesc,
                                            this->dV->get_gpu_ptr(),  // devDV, //  // void *dvalues,
                                            this->V->get_gpu_ptr(),  // const void *values,
                                            weightSizeInBytes,
                                            this->weights->get_gpu_ptr(),  // const void *weights,
                                            workSpaceSizeInBytes,
                                            NULL,  // this->work_space->get_gpu_ptr(),  // void *workSpace,
                                            reserveSpaceSizeInBytes,
                                            this->reserve_buff->get_gpu_ptr() ) );

#ifdef GRAD_ACC
    checkCUDNN(cudnnMultiHeadAttnBackwardWeights(
                                    *cudnn_h,
                                    this->attnDesc,
                                    CUDNN_WGRAD_MODE_SET,
                                    this->qDesc,  // const cudnnSeqDataDescriptor_t qDesc,
                                    this->Q->get_gpu_ptr(),  // const void *queries,
                                    this->kDesc,  // const cudnnSeqDataDescriptor_t kDesc,
                                    this->K->get_gpu_ptr(),  // const void *keys,
                                    this->vDesc,  // const cudnnSeqDataDescriptor_t vDesc,
                                    this->V->get_gpu_ptr(),  // const void *values,
                                    this->oDesc,  // const cudnnSeqDataDescriptor_t doDesc,
                                    this->dO->get_gpu_ptr(),  // const void *dout,
                                    weightSizeInBytes,  // size_t weightSizeInBytes,
                                    this->weights->get_gpu_ptr(),  // const void *weights,
                                    this->dweights->get_temp_gpu_ptr(),  // void *dweights,
                                    workSpaceSizeInBytes, // size_t workSpaceSizeInBytes,
                                    NULL,  // this->work_space->get_gpu_ptr(),  // void *workSpace,
                                    reserveSpaceSizeInBytes,  // size_t reserveSpaceSizeInBytes,
                                    this->reserve_buff->get_gpu_ptr()) );
    value_type one = 1.0;
    size_t total_params = dweights->get_N() * dweights->get_C() * dweights->get_H() * dweights->get_W();
    checkCublasErrors(cublasSaxpy(*(cublas_h), total_params, (const float*)&one, (const float*)dweights->get_temp_gpu_ptr(), 1, (float*)dweights->get_gpu_ptr(), 1));

#else
    checkCUDNN(cudnnMultiHeadAttnBackwardWeights(
                                    *cudnn_h,
                                    this->attnDesc,
                                    CUDNN_WGRAD_MODE_SET,
                                    this->qDesc,  // const cudnnSeqDataDescriptor_t qDesc,
                                    this->Q->get_gpu_ptr(),  // const void *queries,
                                    this->kDesc,  // const cudnnSeqDataDescriptor_t kDesc,
                                    this->K->get_gpu_ptr(),  // const void *keys,
                                    this->vDesc,  // const cudnnSeqDataDescriptor_t vDesc,
                                    this->V->get_gpu_ptr(),  // const void *values,
                                    this->oDesc,  // const cudnnSeqDataDescriptor_t doDesc,
                                    this->dO->get_gpu_ptr(),  // const void *dout,
                                    weightSizeInBytes,
                                    this->weights->get_gpu_ptr(),  // const void *weights,
                                    this->dweights->get_gpu_ptr(),  // void *dweights,
                                    workSpaceSizeInBytes,
                                    NULL,  // this->work_space->get_gpu_ptr(),  // void *workSpace,
                                    reserveSpaceSizeInBytes,
                                    this->reserve_buff->get_gpu_ptr()) );

#endif

    tensor_t<value_type>* dx = this->get_b_data();
    dx->copy(this->Q);
    dx->sum(this->K);
    dx->sum(this->V);

    x->increase_cur_use_counter(BACKWARD);
    this->dO->decrease_cur_b_use_count();
    this->reserve_buff->set_data_state(FORWARD_DELETE_OK);
    this->reserve_buff->set_data_position(NO_DATA);

#ifdef DEBUG
    dEdD->printTensor("Result of Backward Activation");
//    dEdD->GPUtoCPU();
#endif
}

template <class value_type>
void self_attn_layer_t<value_type>::fake_run(net_comp dir, registry_t<value_type>* reg) {
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
        // tensor_t<value_type>* output = this->get_f_out();
        // output->increase_cur_use_counter(BACKWARD);
    }
}
    
INSTANTIATE_CLASS(self_attn_layer_t);
    
} //ATP namespace