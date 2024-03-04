#include <stdlib.h>
#include <ATPSearch.h>
#include <ATPConfigSearch.h>
#include <ATP.h>
#include <ga.h>

using namespace ATP;

base_layer_t<float>* rnn_network(base_layer_t<float> *first_layer, size_t hiddenSize, size_t seqLength, size_t numLayers, 
                            bool with_ln,
                            RNN_TYPE rnn_type = RNN_TANH, RNN_BIAS rnn_bias = DOUBLE, RNN_DIRECTION rnn_direction = UNIDIRECTIONAL,
                            float dropout_rate = 0.5) 
{
    base_layer_t<float>* last_layer;
    for (int i = 0; i < numLayers; i++) {
        base_layer_t<float>* rnn = (base_layer_t<float>*) new rnn_layer_t<float>(hiddenSize, hiddenSize, seqLength, 1, 
                                                                            rnn_type, rnn_bias, rnn_direction,
                                                                            dropout_rate);
        if (i == 0) {
            first_layer->hook(rnn);
        }
        else {
            last_layer->hook(rnn);
        }
        if (with_ln) {
            base_layer_t<float>* ln = (base_layer_t<float>*) new layer_normalization_layer_t<float>(CUDNN_BATCHNORM_SPATIAL, 0.0001);
            rnn->hook(ln);
            last_layer = ln;
        }
        else {
            last_layer = rnn;
        }
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
    size_t inputSize = 128;
    size_t hiddenSize = 1024;
    size_t class_size = 64;
    size_t N = seqLength; 
    size_t C = batch_size;
    size_t H = inputSize;
    size_t W = 1;
    size_t labelSize = 32;
    size_t numLayers = 64;
    RNN_TYPE rnn_type = LSTM;
    
    preprocessor<float>* processor_train = new preprocessor<float>();

    base_layer_t<float> *data_rnn = (base_layer_t<float>*) new data_layer_t<float>(seqLength, batch_size, inputSize, labelSize);
    base_layer_t<float>* rnn_1 = (base_layer_t<float>*) new rnn_layer_t<float>(inputSize, hiddenSize, seqLength, 1, rnn_type);
    base_layer_t<float>* ln_2 = (base_layer_t<float>*) new layer_normalization_layer_t<float>(CUDNN_BATCHNORM_SPATIAL, 0.001);

    parallel_reader_t<float >* reader1 = (parallel_reader_t<float >*)new parallel_reader_t<float >(train_image_bin, train_label_bin, 4, N, C, H, W, processor_train, 4, 2);
    
    // /*--------------network configuration--------------*/
    base_solver_t<float> *solver = (base_solver_t<float> *) new sgd_solver_t<float>(0.0000001, 0.0004);
    // solver->set_lr_decay_policy(ITER, {100000, 200000, 300000, 400000}, {0.001, 0.0001, 0.00001, 0.000001});
    network_t<float> n(solver);
    data_rnn->hook(rnn_1);
    rnn_1->hook(ln_2);
    base_layer_t<float> *net = rnn_network(ln_2, hiddenSize, seqLength, numLayers, true, rnn_type);
    base_layer_t<float>* rnn_last = (base_layer_t<float>*) new rnn_layer_t<float>(hiddenSize, hiddenSize, seqLength, 1, rnn_type);
    base_layer_t<float>* ctcloss = (base_layer_t<float>*) new ctcloss_layer_t<float>();
    net->hook(rnn_last);
    rnn_last->hook(ctcloss);

    printf("after hook network, memory cost = %zd =  %f\n", query_used_mem(), BYTE_TO_MB(query_used_mem()));
    size_t one_batch_inherent_size = 771;
    double inherent_size_step = 0.23;
    
#ifdef BEST_BATCHSIZE
    const size_t baseline_batchsize = static_cast<const size_t>(atoi(argv[2]));
    const size_t batchsize_step = static_cast<const size_t>(atoi(argv[3]));
	const size_t max_batchsize = static_cast<const size_t>(atoi(argv[4]));
    const size_t iter_times = static_cast<const size_t>(atoi(argv[5]));
    const size_t ga_iter_times = static_cast<const size_t>(atoi(argv[6]));
    const size_t population_size = static_cast<const size_t>(atoi(argv[7]));
    const size_t batch_size_num = (max_batchsize - baseline_batchsize) / batchsize_step + 1;

    ATPSearch<float> *swap_net = new ATPSearch<float>(&n);
    // size_t gpu_mem = 17179869184;  // 16GB
    size_t gpu_mem = 34078720000;  // 32GB
    double pcie_bandwidth = 11084901888;  // PCIE 3.0
	swap_net->set_simulator(gpu_mem, pcie_bandwidth);
    size_t no_update_win = 2;

    ThroughputPeakSearch(  data_rnn, ctcloss, &n, processor_train, reader1, gpu_mem, pcie_bandwidth, 
                baseline_batchsize, max_batchsize, batchsize_step, iter_times, no_update_win, true,
                population_size, 0.0005, ga_iter_times);

#else

    n.init_network_trainning(
            batch_size, 0, 
			processor_train,
			reader1,
		    data_rnn, ctcloss,
		    train_image_bin, train_label_bin, train_mean_file);
    // while(1);

#ifdef ATP_SOLUTION
    // const size_t code_size = n.GetLayerNum();
    const size_t code_size = 199;
    /* ATP: batchsize = 512 * 10 * 128 * 1 */
    // int rs_code[code_size] = {0,1,1,1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,1,0,0,1,1,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,1,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,1,1,0,1,0,0,0,0,1,0,0,0,0,0,1,1,0,1,0,0,1,0,0,0,1,0,0,0,0,0,0,1,0,0,0,0,0,0,1,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,1,1,1,0};
    /*********************************************/
    /* ATP: batchsize = 512 * 12 * 128 * 1 */
    // int rs_code[code_size] = {0,1,1,1,0,0,0,0,1,0,0,0,1,0,0,0,1,0,0,1,0,0,0,0,0,0,1,0,0,1,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,1,1,0,0,1,0,0,0,0,0,0,1,0,0,1,0,0,0,1,0,0,1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,1,1,0,0,1,1,0,0,0,0,0,1,0,0,0,1,0,0,1,1,0,0,1,0,0,1,1,0,1,0,0,0,0,1,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,1,1,0,1,1,0,0,1,0,0,1,0,0,0,0,0,0,1,1,0,0,1,0,0,1,0,0,0,0,1,0,0,0,0,0,0,0,0,1,1,0,0,1,0,0,0,1,0,0,1,1,1,0};
    /*********************************************/
    /* ATP: batchsize = 512 * 14 * 128 * 1 */
    // int rs_code[code_size] = {0,1,1,1,0,0,0,0,0,1,0,0,0,1,0,0,0,1,1,0,0,1,1,0,0,0,1,1,0,0,0,1,1,0,0,0,1,1,1,0,0,0,1,0,0,1,1,0,1,0,0,0,0,0,0,0,1,0,0,1,1,0,0,1,0,1,1,0,1,0,0,0,1,0,0,1,0,0,0,1,0,1,1,1,0,0,0,0,0,0,1,0,1,1,1,0,1,0,0,0,1,0,0,1,1,0,1,0,0,1,0,0,0,1,1,0,0,0,1,0,0,0,0,1,1,0,1,1,0,0,0,0,0,1,1,0,0,0,0,1,0,0,0,1,0,1,0,0,0,0,1,0,1,0,1,0,1,1,0,1,0,0,0,1,0,1,0,0,0,0,1,1,0,0,0,0,0,1,0,0,1,1,1,0,1,0,0,0,0,1,0,1,0,0,1,1,1,1,0};
    /*********************************************/
    /* ATP: batchsize = 512 * 16 * 128 * 1 */
    int rs_code[code_size] = {0,1,1,1,1,0,1,0,0,0,0,0,1,0,0,1,1,0,1,0,0,1,0,0,0,1,1,1,0,0,1,0,0,1,0,1,0,1,0,0,0,1,0,1,1,1,0,0,1,0,1,0,1,0,1,1,0,0,1,0,1,0,0,0,0,1,1,1,0,1,0,0,0,1,0,1,1,0,1,0,0,1,1,0,0,0,1,1,0,1,1,1,0,1,1,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,1,1,1,0,0,1,0,0,0,0,0,0,0,0,1,1,1,1,0,0,0,0,0,1,1,0,1,0,1,1,0,0,0,0,1,1,1,0,0,1,0,1,0,1,0,0,0,0,1,0,1,0,0,0,0,1,0,1,1,0,0,1,0,0,0,0,1,1,1,0,0,1,0,1,1,0,0,0,1,0,1,1,1,0,0,1,1,1,0};
    /*********************************************/
    /* ATP: batchsize = 512 * 18 * 128 * 1 */
    // int rs_code[code_size] = {0,1,1,1,0,0,2,1,1,2,1,0,2,0,0,2,1,1,2,0,0,2,0,0,2,0,0,2,1,0,2,1,0,2,0,1,2,1,0,2,0,0,2,0,0,2,1,1,2,1,0,2,1,0,2,0,1,2,1,1,0,1,0,2,0,1,2,1,1,2,0,0,2,0,0,2,1,1,2,0,0,2,0,0,0,1,0,2,1,0,2,0,0,2,1,1,2,1,0,2,0,0,2,0,0,2,1,0,1,0,1,2,1,1,2,0,0,2,1,0,2,0,0,2,1,0,2,1,1,2,0,0,2,0,0,2,0,0,2,1,0,2,0,0,2,1,0,2,1,1,2,0,0,2,0,0,2,0,0,2,0,0,2,1,0,2,1,0,2,0,0,2,0,0,0,1,0,2,1,1,0,0,1,2,0,0,2,1,1,2,1,1,2,1,1,1,1,1,0};
    /*********************************************/
    /* ATP: batchsize = 512 * 20 * 128 * 1 */
    // int rs_code[code_size] = {0,1,1,1,0,0,2,1,0,2,1,0,2,1,1,2,1,0,2,0,1,2,0,1,2,1,1,2,0,1,2,1,0,2,0,1,2,1,0,2,1,0,2,1,1,2,0,0,2,0,1,2,1,0,2,0,1,2,1,1,1,1,0,2,1,0,2,1,0,2,0,0,2,1,0,2,1,1,2,1,0,2,0,1,2,1,0,2,0,1,2,0,1,2,0,1,0,0,0,2,1,0,2,0,0,2,1,0,2,1,0,2,1,0,2,0,1,2,1,0,2,0,1,2,0,0,2,1,1,2,0,0,2,0,1,2,0,1,2,0,1,2,1,0,2,1,1,2,1,0,2,1,0,2,0,0,2,1,0,2,1,1,2,0,0,2,1,0,2,0,1,2,1,1,2,1,0,2,1,0,2,0,1,2,1,0,0,0,1,2,1,0,2,0,1,1,1,1,0};
    /*********************************************/
    /* ATP: batchsize = 512 * 22 * 128 * 1 */
    // int rs_code[code_size] = {0,1,1,1,0,1,0,1,1,2,1,0,1,1,0,1,1,1,2,1,1,2,0,0,2,0,0,2,1,0,2,0,1,2,0,1,2,0,0,2,1,0,2,0,1,2,1,1,2,1,1,2,0,0,2,1,0,2,1,1,2,0,1,2,0,0,2,1,0,2,0,1,2,0,0,2,1,1,2,1,0,2,0,0,2,1,1,2,1,1,2,1,0,2,0,1,2,1,0,2,1,1,2,1,0,2,1,1,2,1,1,2,1,0,2,1,0,2,1,0,2,0,1,2,0,0,2,1,0,2,1,1,2,0,1,2,0,0,2,1,0,2,0,1,2,1,0,2,1,1,2,1,0,2,1,0,2,0,0,2,1,0,2,1,0,2,1,1,2,1,0,2,0,1,2,1,1,2,0,0,2,1,0,2,0,0,2,1,0,2,0,0,2,0,1,1,1,1,0};
    /*********************************************/
    /* ATP: batchsize = 512 * 24 * 128 * 1 */
    // int rs_code[code_size] = {0,1,1,1,1,1,2,0,0,2,1,0,2,1,0,2,0,0,2,1,1,2,1,1,2,1,0,2,1,1,2,0,1,2,1,0,2,0,1,2,0,1,2,1,1,2,0,1,2,1,0,2,1,1,2,0,1,2,0,0,0,1,1,2,1,1,2,1,0,2,1,1,2,0,1,2,1,1,2,0,0,2,1,0,2,1,1,2,1,0,2,1,1,2,0,0,2,0,0,2,1,0,2,0,0,2,1,0,2,1,1,2,0,0,2,1,0,2,0,1,2,0,1,2,1,1,2,1,0,0,1,1,2,1,0,2,0,1,2,1,0,2,1,0,2,0,1,2,1,0,2,1,0,2,1,0,2,1,1,2,0,0,2,1,1,2,1,0,2,1,0,2,0,1,2,0,0,2,1,1,2,1,0,1,0,1,2,0,1,2,1,0,2,1,1,1,1,1,0};
    /*********************************************/
    /* ATP: batchsize = 512 * 26 * 128 * 1 */
    // int rs_code[code_size] = {0,1,1,1,0,0,0,1,0,0,1,1,1,0,1,2,1,1,2,1,1,2,1,0,2,1,1,2,1,0,2,1,0,2,0,1,2,0,1,2,0,1,2,0,1,2,1,0,2,1,1,2,1,0,2,1,1,2,1,0,2,1,1,2,0,0,2,1,0,2,1,1,2,0,1,2,1,0,2,1,1,2,0,1,2,0,0,2,1,0,2,0,1,2,1,0,2,0,0,2,1,1,2,1,0,2,0,1,2,1,0,2,1,1,2,1,0,2,1,0,2,1,0,2,1,1,2,0,0,2,1,0,2,1,1,2,0,0,2,1,1,2,1,0,2,0,1,2,1,1,2,1,1,2,0,1,2,1,1,2,1,1,2,1,1,2,1,0,2,0,0,2,1,1,2,0,0,2,1,0,2,1,1,2,1,0,2,0,1,2,1,0,2,1,1,1,1,1,0};
    /*********************************************/
    /* ATP: batchsize = 512 * 30 * 128 * 1 */
    // int rs_code[code_size] = {0,1,1,1,1,0,0,1,1,1,0,1,2,1,0,1,1,0,2,0,1,2,1,0,2,1,1,2,1,0,2,1,1,2,1,1,2,1,0,2,1,1,2,1,1,2,1,1,2,1,0,2,1,1,2,1,0,2,1,1,2,1,1,2,1,1,2,0,1,2,0,1,2,1,1,2,1,1,2,1,0,2,1,1,2,1,0,2,1,1,2,0,1,2,1,1,2,1,0,2,1,1,2,0,1,2,1,1,2,0,1,2,0,0,2,1,1,2,0,1,2,1,1,2,1,0,2,1,0,2,1,1,2,1,0,2,0,1,2,0,0,2,1,0,2,1,1,2,1,1,2,0,1,2,1,0,2,1,1,2,1,1,2,1,0,2,1,1,2,1,0,2,0,1,2,0,0,2,1,1,2,0,0,2,1,1,2,0,0,2,1,0,2,0,1,1,1,1,0};
    /*********************************************/
    /* ATP: batchsize = 512 * 34 * 128 * 1 */
    // int rs_code[code_size] = {0,1,1,1,1,1,2,1,1,2,0,1,2,1,0,2,1,1,2,1,0,2,1,1,2,1,0,2,1,1,2,0,1,2,1,0,2,0,1,2,1,0,2,1,1,2,1,0,2,1,1,2,1,0,2,1,1,2,0,1,2,1,1,2,1,1,2,0,0,2,1,1,2,1,1,2,1,1,2,1,0,2,1,0,2,0,1,2,1,0,2,1,1,2,1,0,2,1,1,2,0,0,1,1,1,1,1,1,2,1,1,2,0,1,2,1,1,2,1,0,2,1,1,2,0,1,2,1,1,2,0,1,2,1,0,2,1,1,2,1,0,2,1,0,2,1,0,2,1,1,2,1,1,2,1,0,2,1,1,0,0,1,2,0,1,2,1,1,2,0,1,2,1,1,2,0,0,2,1,1,2,1,1,2,1,0,2,1,1,2,0,0,2,1,1,1,1,1,0};
    /*********************************************/
    /* ATP: batchsize = 512 * 38 * 128 * 1 */
    // int rs_code[code_size] = {0,1,1,1,1,0,1,1,0,1,1,0,0,1,1,2,1,1,2,1,0,2,0,1,2,1,0,2,1,1,2,0,0,2,1,1,2,0,1,2,1,0,2,1,1,2,1,0,2,1,1,2,1,0,2,1,1,2,0,1,2,1,1,2,1,1,2,1,0,2,0,1,2,1,0,2,1,1,2,0,0,2,1,1,2,0,1,2,1,1,2,1,1,2,1,0,2,1,0,2,1,1,2,1,1,2,1,1,2,0,0,2,1,1,2,1,1,2,1,0,2,1,0,2,1,1,2,1,0,2,1,1,2,0,1,2,1,1,2,1,1,2,0,1,2,1,1,2,1,0,2,1,0,2,0,1,2,1,0,2,1,1,2,1,1,2,1,1,2,0,0,2,1,1,2,1,1,2,1,1,2,0,1,2,1,1,2,1,1,2,1,0,2,1,1,1,1,1,0};
    /*********************************************/
    /* ATP: batchsize = 512 * 42 * 128 * 1 */
    // int rs_code[code_size] = {0,1,1,1,1,1,1,1,1,1,1,1,2,0,1,1,0,1,2,1,1,2,1,1,2,1,0,2,1,1,2,1,1,2,1,0,2,1,1,2,1,1,2,1,1,2,1,1,2,1,0,2,1,1,2,1,0,2,1,1,2,1,1,2,1,0,2,0,1,2,1,0,2,1,1,2,0,1,2,1,1,2,1,1,2,1,0,2,0,1,2,1,1,2,1,1,2,1,0,2,1,1,2,1,1,2,1,0,2,1,1,2,1,0,2,1,1,2,0,0,2,1,1,2,1,1,2,1,1,2,0,1,2,1,0,2,1,1,2,1,1,2,1,1,2,0,0,2,1,1,2,1,1,2,1,0,2,0,1,2,1,0,2,1,1,2,0,0,2,1,1,2,1,1,2,1,1,2,0,1,2,1,0,2,1,1,2,1,0,2,1,1,2,1,1,1,1,1,0};
    /*********************************************/
    /* ATP: batchsize = 512 * 44 * 128 * 1 */
    // int rs_code[code_size] = {0,1,1,1,0,1,1,1,0,1,1,1,1,1,1,2,0,1,2,1,1,2,1,1,2,1,0,2,1,1,2,1,0,2,1,1,2,1,1,2,1,1,2,1,1,2,1,1,2,0,1,2,1,0,2,1,1,2,1,1,2,1,1,2,1,1,2,0,1,2,1,1,2,1,1,2,1,0,2,0,1,2,1,1,2,1,0,2,1,1,2,0,1,2,1,1,2,1,1,2,1,0,2,1,1,2,1,1,2,1,1,2,1,0,2,1,1,2,1,0,2,1,1,2,0,1,2,1,1,2,1,1,2,1,0,2,1,1,2,1,1,2,0,1,2,1,1,2,0,1,2,1,1,2,1,1,2,1,0,2,1,1,2,1,0,2,1,1,2,1,1,2,1,0,2,1,0,2,1,1,2,1,0,2,1,1,2,1,1,2,1,1,2,0,1,1,1,1,0};
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
