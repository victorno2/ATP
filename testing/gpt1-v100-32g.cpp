#include <stdlib.h>
#include <ATPSearch.h>
#include <ATPConfigSearch.h>
#include <ATP.h>
#include <ga.h>

using namespace ATP;

base_layer_t<float>* DecorderModule(base_layer_t<float> *first_layer, size_t seqLength, size_t miniBatch, size_t beamDim, size_t embedSize, size_t numHead, size_t numModule, float dropout_rate = 0.0) 
{
    base_layer_t<float>* last_layer;
    for (int i = 0; i < numModule; i++) {
        base_layer_t<float>* self_attn = (base_layer_t<float>*) new self_attn_layer_t<float>(embedSize, numHead, seqLength, miniBatch, beamDim, embedSize);
        base_layer_t<float>* layer_norm1 = (base_layer_t<float>*) new layer_normalization_layer_t<float>(CUDNN_BATCHNORM_SPATIAL, 0.0001);
        size_t linear_out_size = embedSize;
        base_layer_t<float>* linear = (base_layer_t<float> *) new fully_connected_layer_t<float>(linear_out_size, new xavier_initializer_t<float>(), true);
        base_layer_t<float>* layer_norm2 = (base_layer_t<float>*) new layer_normalization_layer_t<float>(CUDNN_BATCHNORM_SPATIAL, 0.0001);
        
        if (i == 0) {
            first_layer->hook(self_attn);
        }
        else {
            last_layer->hook(self_attn);
        }
        self_attn->hook(layer_norm1);
        layer_norm1->hook(linear);
        linear->hook(layer_norm2);
        last_layer = layer_norm2;
    }
    return last_layer;
}

int main(int argc, char **argv) {

    char* train_label_bin;
    char* train_image_bin;
    char* train_mean_file;
    
    char* test_label_bin;
    char* test_image_bin;
    char* checkpoint_file;
    
    cudaSetDevice(static_cast<const size_t>(atoi(argv[1])));
	int deviceId;
	cudaGetDevice(&deviceId);
	printf("cuda device = %d\n", deviceId);
    
    train_mean_file = (char *) "/data/lwang53/dataset/imgnet/bin_256x256_imgnet/train_mean.bin";
    train_image_bin = (char *) "/data/lwang53/dataset/imgnet/bin_256x256_imgnet/train_data_0.bin";
    train_label_bin = (char *) "/data/lwang53/dataset/imgnet/bin_256x256_imgnet/train_label_0.bin";
    test_image_bin  = (char *) "/data/lwang53/dataset/imgnet/bin_256x256_imgnet/val_data_0.bin";
    test_label_bin  = (char *) "/data/lwang53/dataset/imgnet/bin_256x256_imgnet/val_label_0.bin";
    checkpoint_file = (char *) "/data/lwang53/checkpoints/alexnet/alexnet_checkpoint";
    
    size_t batch_size = static_cast<const size_t>(atoi(argv[2])); //train and test must be same
    size_t training_iter_time = static_cast<const size_t>(atoi(argv[3])); //train and test must be same

    size_t seqLength = 512;
    size_t beamDim = 1;
    size_t embedSize = 768;
    size_t N = seqLength; 
    size_t C = batch_size;
    size_t H = beamDim;
    size_t W = embedSize;
    size_t numHead = 12;
    size_t numModule = 12;
    

    preprocessor<float>* processor_train = new preprocessor<float>();
    base_layer_t<float> *data_seq2seq = (base_layer_t<float>*) new data_layer_t<float>(seqLength, batch_size, beamDim, embedSize, true, DATA_TRAIN);
    parallel_reader_t<float >* reader1 = (parallel_reader_t<float >*)new parallel_reader_t<float >(train_image_bin, train_label_bin, 4, N, C, H, W, processor_train, 4, 2);

    // base_layer_t<float>* self_attn = (base_layer_t<float>*) new self_attn_layer_t<float>(embedSize, numHead);

    base_layer_t<float>* net = DecorderModule(data_seq2seq, seqLength, batch_size, beamDim, embedSize, numHead, numModule);

    base_layer_t<float> *full_conn_1 = (base_layer_t<float> *) new fully_connected_layer_t<float>(10, new xavier_initializer_t<float>(), true);
    base_layer_t<float> *softmax = (base_layer_t<float> *) new softmax_layer_t<float>(CUDNN_SOFTMAX_ACCURATE, CUDNN_SOFTMAX_MODE_INSTANCE);

    // data_seq2seq->hook(self_attn);
    net->hook(full_conn_1);
    // self_attn->hook(full_conn_1);
    full_conn_1->hook(softmax);
    
    // /*--------------network configuration--------------*/
    base_solver_t<float> *solver = (base_solver_t<float> *) new sgd_solver_t<float>(0.0000001, 0.0004);
    // solver->set_lr_decay_policy(ITER, {100000, 200000, 300000, 400000}, {0.001, 0.0001, 0.00001, 0.000001});
    network_t<float> n(solver);

    printf("after hook network, memory cost = %zd =  %f\n", query_used_mem(), BYTE_TO_MB(query_used_mem()));
    size_t one_batch_inherent_size = 773;
    double inherent_size_step = 4.5;
    
#ifdef BEST_BATCHSIZE
    std::map<int, size_t> modnn_offload_size;
    ATPSearch<float> *swap_net = new ATPSearch<float>(&n);
    // size_t gpu_mem = 17179869184;  // 16GB 16384MB
    size_t gpu_mem = 34078720000;  // 32GB
    double pcie_bandwidth = 11084901888;  // PCIE 3.0
	swap_net->set_simulator(gpu_mem, pcie_bandwidth);

    const size_t baseline_batchsize = static_cast<const size_t>(atoi(argv[2]));
    const size_t batchsize_step = static_cast<const size_t>(atoi(argv[3]));
	const size_t max_batchsize = static_cast<const size_t>(atoi(argv[4]));
    const size_t iter_times = static_cast<const size_t>(atoi(argv[5]));
    const size_t ga_iter_times = static_cast<const size_t>(atoi(argv[6]));
    const size_t population_size = static_cast<const size_t>(atoi(argv[7]));
    const size_t batch_size_num = (max_batchsize - baseline_batchsize) / batchsize_step + 1;
    size_t no_update_win = 2;

    ThroughputPeakSearch(  data_seq2seq, softmax, &n, processor_train, reader1, gpu_mem, pcie_bandwidth, 
                baseline_batchsize, max_batchsize, batchsize_step, iter_times, no_update_win, true,
                population_size, 0.0005, ga_iter_times);    

#else
    
    n.init_network_trainning(
            batch_size, 0, 
			processor_train,
			reader1,
		    data_seq2seq, softmax,
		    train_image_bin, train_label_bin, train_mean_file);
    // while(1);


#ifdef ATP_SOLUTION
    // const size_t code_size = n.GetLayerNum();
    const size_t code_size = 63;

    /* ATP: INPUT SIZE = 512 * 26 * 1 * 768 */
    int rs_code[code_size] = {0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,1,0,0,1,1,0,0,1,0,1,1,1,1,1,1,1,0,1,1,0,1,1,0,1,1,1,1,0,1,0,0,1,1,0,0,1,0,1,0,0,1,1,1,1,1,1,0};
    /*********************************************/
    /* ATP: INPUT SIZE = 512 * 24 * 1 * 768 */
    // int rs_code[code_size] = {0,1,1,1,1,1,1,0,1,1,0,0,0,1,1,0,0,1,1,0,1,1,0,0,1,0,1,1,0,1,0,1,0,1,1,1,1,0,1,0,0,1,0,0,0,1,0,1,1,1,1,1,1,0,1,1,1,1,1,1,1,1,0};
    /*********************************************/
    /* ATP: batchsize = 512 * 13 * 128 * 1 */
    // int rs_code[code_size] = {0,1,1,1,0,1,0,0,0,1,0,0,1,0,0,0,1,0,0,1,1,0,0,1,0,0,0,1,1,1,0,0,0,1,0,0,0,1,1,0,1,0,0,0,1,1,0,1,0,1,0,1,0,0,1,1,0,0,0,1,1,1,0};
    /*********************************************/
    /* ATP: batchsize = 512 * 14 * 128 * 1 */
    // int rs_code[code_size] = {0,1,1,1,0,1,0,1,1,0,0,1,1,0,1,0,0,0,0,1,0,0,1,0,0,0,1,0,0,0,0,0,1,0,0,1,1,0,0,0,0,0,0,0,1,0,0,1,0,0,1,0,0,1,0,0,0,1,0,1,1,1,0};
    /*********************************************/
    /* ATP: batchsize = 512 * 16 * 128 * 1 */
    // int rs_code[code_size] = {0,1,1,1,0,1,1,0,0,1,0,0,0,0,0,0,1,0,0,1,0,0,1,1,0,0,0,1,1,1,1,0,0,1,1,1,1,0,0,1,0,1,0,0,1,0,1,0,1,0,0,0,0,1,1,0,0,1,1,1,1,1,0};
    /*********************************************/
    /* ATP: batchsize = 512 * 18 * 128 * 1 */
    // int rs_code[code_size] = {0,1,1,1,0,0,0,0,1,0,0,1,0,0,1,1,1,0,0,0,1,1,0,0,1,0,0,1,1,1,1,0,0,0,0,1,0,0,1,1,1,1,0,0,1,0,1,0,1,0,1,0,1,1,1,0,1,0,0,1,1,1,0};
    /*********************************************/
    /* ATP: batchsize = 512 * 20 * 128 * 1 */
    // int rs_code[code_size] = {0,1,1,1,0,1,1,0,0,1,1,0,0,0,0,0,1,1,1,0,1,1,0,1,0,1,0,0,0,0,0,1,0,0,1,1,1,0,1,0,0,1,0,1,0,1,0,1,0,1,1,0,0,0,0,0,1,1,0,1,1,1,0};
    /*********************************************/
    /* ATP: batchsize = 512 * 22 * 128 * 1 */
    // int rs_code[code_size] = {0,1,1,1,1,1,0,0,0,0,1,1,0,0,1,0,1,0,0,0,0,1,0,1,0,1,1,0,0,1,0,1,0,0,0,0,0,0,0,1,1,0,1,0,1,0,1,1,0,0,0,1,0,0,0,0,1,1,0,1,1,1,0};
    /*********************************************/    

    if (n.SetRecomputeSwapTensorsbyRoute(rs_code, code_size)) {
        printf("SetRecomputeSwapTensorsbyRoute down\n");
    }
    else {
        exit(0);
    }
#endif   

    const size_t train_imgs = 50000;
    const size_t tracking_window = train_imgs / batch_size;
    n.train(training_iter_time, tracking_window, 10);
    exit(0);

#endif
}
