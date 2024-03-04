#if !defined(_NETWORK_H_)
#define _NETWORK_H_

#include <deque>
#include <registry.h>
#include <mem_control.h>
#include <util/common.h>
#include <util/error_util.h>
#include <layer/base_layer.h>
#include <layer/data_layer.h>
#include <util/ATP_math.h>
#include <solver.h>
#include <util/error_util.h>
#include <stream_singleton.h>
#include <gpu_malloc.h>
#include <util/mem_util.h>
#include <layer/cudnn_convolution_layer.h>
#include <util/saver.h>
#include <string>
#include <fstream>
// #include <layer/base_structure_layer.h>
// #include <layer/base_network_layer.h>
// using namespace std;

namespace ATP{

template <class value_type>
class network_t
{
private:
	
	MODEL_TYPE model_type = CNN_NETWORK;
	net_comp training_state = BACKWARD;
	bool start_flag = false;
	
	int trainning_set_flag; // if network has been set for trainning once, trainning_set_flag = 1;
	
    /*-solver configurations-*/
    base_solver_t<value_type>* solver;
    const value_type clip_gradient_limit;
    size_t test_iter;
    size_t test_interval;
    size_t batchsize;
	size_t big_batchsize = 0;

	int max_layer_id;

    /*-network configurations-*/
    size_t GPU_id;
    bool is_forward_setup;
    bool is_testing_ready;
    bool is_backward_setup;
    bool is_network_computable;
    math_util<value_type> math;
    cudnnHandle_t   cudnn_handle;
    cublasHandle_t  cublas_handle;
    cudnnDataType_t cudnn_data_type;
    cudaStream_t stream = stream_singleton::get_compute_stream();

	int abandon = 0;   // record time after 800 iters
	int max_swap_time_test_times = 0;
	int cur_swap_time_test_times = 0;

    /*-computation route-*/
    //we maintain the uni-direction of forward_backward route,
    //this guides the creation, offloadig and deletion of tensors
    //std::vector<std::pair<int, net_comp> >      net_comp_route;
    //std::map<int, std::vector< tensor_t<value_type>* > > tensor_by_layer;

    /*--test swapping--*/
    //we do test by swapping the head of data reader,
    //softmax_loss_layer is tracked for backward computation
    //train
    base_layer_t<value_type>*  train_data_layer       = NULL;
    base_layer_t<value_type>*  softmax_loss_layer     = NULL;
    //test
    base_layer_t<value_type>*  test_data_layer        = NULL;
    /*--registry records the info of every tensor--*/
    registry_t<value_type> *reg;
    mem_controller_t<value_type> mem_controller;

	std::vector<tensor_t<value_type>* > recompute_tensors;
	std::vector<tensor_t<value_type>* > swap_tensors;
	std::vector<tensor_t<value_type>* > prefetch_tensors;
	std::vector<double> fpst;
	std::vector<double> bpst;
	std::vector<double> sct;

	std::stack<int> recompute_layers_stack;

	std::map<LAYER, double> layer_type_ft;
	std::map<LAYER, double> layer_type_bt; 
    std::map<LAYER, int> layer_type_num;
    std::map<LAYER, double> layer_type_mem;
    std::map<LAYER, double> layer_type_b_mem;

	value_type* swap_test_block_gpu = NULL;
	value_type* swap_test_block_cpu = NULL;
	std::map<size_t, double> offload_time_by_size;
	std::map<size_t, double> fetch_time_by_size;
	std::map<size_t, size_t> offload_test_times_by_size;
	std::map<size_t, size_t> fetch_test_times_by_size;
	std::map<size_t, double> offload_time_by_tensor;
	std::map<size_t, double> fetch_time_by_tensor;
	std::map<size_t, size_t> offload_test_times_by_tensor;
	std::map<size_t, size_t> fetch_test_times_by_tensor;

    std::map<int, double> layers_ft;
    std::map<int, double> layers_bt;
	std::map<int, double> layers_ft2;
    std::map<int, double> layers_bt2;
	double backward_last_time = 0;
	double average_backward_last_time = 0;
    double f_time = 0;
    double b_time = 0;
	
	/* data load */
	base_preprocess_t<value_type> *mean_sub;
    base_preprocess_t<value_type> *pad;
    base_preprocess_t<value_type> *crop;
    base_preprocess_t<value_type> *flip;
    base_preprocess_t<value_type> *bright;
    base_preprocess_t<value_type> *contrast;
    base_preprocess_t<value_type> *standardization;
	preprocessor<value_type> *processor;
	parallel_reader_t<value_type> *reader;
	base_layer_t<value_type> *data_layer;
	base_layer_t<value_type> *loss_layer;
	
	double* forward_pur_swap_times;
	double* backward_pur_swap_times;
	double* swap_ctrl_times;

    void createHandles()
    {
        // cudaSetDevice(1);
		size_t temp1, temp2;
		temp1 = query_used_mem();
        checkCUDNN( cudnnCreate(&cudnn_handle) );
		temp2 = query_used_mem();
		printf("cudnnCreate(&cudnn_handle) use %lf MB\n", BYTE_TO_MB(temp2-temp1));
        cudnnSetStream(cudnn_handle, stream);
		temp1 = query_used_mem();
		printf("cudnnSetStream(cudnn_handle, stream) use %lf MB\n", BYTE_TO_MB(temp1-temp2));
        checkCublasErrors( cublasCreate(&cublas_handle) );
		temp2 = query_used_mem();
		printf("cublasCreate(&cublas_handle) use %lf MB\n", BYTE_TO_MB(temp2-temp1));
        cublasSetStream(cublas_handle, stream);
		temp1 = query_used_mem();
		printf("cublasSetStream(cublas_handle, stream) use %lf MB\n", BYTE_TO_MB(temp1-temp2));
    }

    void destroyHandles()
    {
        checkCUDNN( cudnnDestroy(cudnn_handle) );
        checkCublasErrors( cublasDestroy(cublas_handle) );
    }

    void gradient_check_kernel(int l_id, size_t n, size_t c, size_t h, size_t w, tensor_t<value_type>* data, tensor_t<value_type>* diff, const char* str);

    void update_kernel( base_layer_t<value_type>* l, size_t iter);

    void regularization(base_layer_t<value_type>* l);
 
    void calculate_update_value(base_layer_t<value_type>* l);

    void forward_test(network_stage stage, base_layer_t<value_type>* b, std::vector<value_type>* acc);
	
	void simulated_forward_kernel(
		network_stage stage, base_layer_t<value_type>* b, std::vector<value_type>* loss,
		std::map<int, double> *layers_ft);

    void forward_kernel(network_stage stage, base_layer_t<value_type>* b, std::vector<value_type>* loss);

    void backward_with_update_kernel(base_layer_t<value_type>* l, size_t iter);

	void simulated_backward_kernel(base_layer_t<value_type>* b, std::map<int, double> *layers_bt);

    void backward_kernel(base_layer_t<value_type>* b);
	
	void recompute_kernel(std::map<int, void* >* net_layers, base_layer_t<value_type>* b, net_comp dir);

    void fsetup_kernel(base_layer_t<value_type>* start_layer);

    void bsetup_kernel(base_layer_t<value_type>* start_layer);

    void test();

    // void write_tensor(int32_t layer_id, tensor_t<value_type> *t, std::ofstream *out);
    // void read_tensor(int32_t *layer_id, tensor_t<value_type> **t, std::ifstream *in);
    
    void meta_setup() {
        //CAUTION: the sequence of registry and mem_controller matters
        printf("************network layer configurations************\n");
        // reg->print_net_comp_route();
        mem_controller.init(reg);
        // mem_controller.print_regulated_tensors();
        // printf("****************************************************\n");
    }

    std::vector<double> network_perf_profile();

    std::shared_ptr<std::thread> query_thread;
    std::atomic_bool query_stop;

	void swap_ctrl(tensor_t<value_type>* tensor, MODEL_TYPE model_type, net_comp dir, 
					double* swap_ctrl_time, double* pur_swap_time);

/*		
	void synchronize(STREAM stream) {
		switch(stream) {
			case COMPUTE_STREAM: cudaStreamSynchronize(stream_singleton::get_compute_stream());
			case GPU2CPU_STREAM: cudaStreamSynchronize(stream_singleton::get_gpu2cpu_stream());
			case CPU2GPU_STREAM: cudaStreamSynchronize(stream_singleton::get_cpu2gpu_stream());
		}
	}
*/
    void init_swap_managemnet(int device_id) {
		
        query_stop = false;
        query_thread = std::make_shared<std::thread>([&]() {

#ifdef __linux__
            // must wait a little to set affinity
                std::this_thread::sleep_for(std::chrono::milliseconds(20));
                std::cout << "gpu mem query thread run on CPU " << sched_getcpu() << "\n";
#endif
			cudaSetDevice(static_cast<const size_t>(device_id));
			int deviceId;
			cudaGetDevice(&deviceId);
			printf("swap_managemnet_thread cuda device = %d\n", deviceId);
            size_t max_usage = 0;
            size_t tmp;
			
            double ts = get_cur_time();
			while(start_flag == false) {
				// printf("start_flag = %d\n", start_flag);
				std::this_thread::sleep_for(std::chrono::milliseconds(100));
			}

            // while ( ! query_stop.load() ) {
			while(start_flag == true) {
#ifdef SWAP_ON
#ifdef MULTI_SWAP_BLOCK
				// swapping_block control
				// if (training_state == FORWARD) 
				{
					
					for (size_t i = 0; i < swap_tensors.size()-SWAP_BLOCK_NUM; i++) {
					// for (size_t i = 0; i < swap_tensors.size(); i++) {
						// the last swap_tensors should not be offload
						// printf("start forward swap_ctrl tensor%d\n", swap_tensors[i]->get_tensor_id());
						swap_ctrl(swap_tensors[i], model_type, FORWARD, NULL, &forward_pur_swap_times[i]);	
						// swap_ctrl(swap_tensors[i], model_type, FORWARD, NULL, NULL);						
					}
					double pur_swap_time = 0.0;
					for (size_t i = 0; i < swap_tensors.size(); i++) {
						// swap_tensors[i]->set_swap_time(OFFLOAD);
						pur_swap_time += forward_pur_swap_times[i];
						// fpst[i] = fpst[i] + forward_pur_swap_times[i];
					}
					printf(" **** forward pur swap time = %f ****\n", pur_swap_time);
					
					while(true) {
						printf("");
						if (training_state == BACKWARD) break;	
					}
					
				}
				// else  
				{  	// training_state == BACKWARD
					
					// for (size_t i = swap_tensors.size()-1; i >= 0; i--) {
						// printf("start %d backward swap_ctrl tensor%d\n", i, swap_tensors[i]->get_tensor_id());
						// swap_ctrl(swap_tensors[i], model_type, BACKWARD, NULL, &backward_pur_swap_times[i]);	
						// printf("end %d backward swap_ctrl tensor%d\n", i, swap_tensors[i]->get_tensor_id());
						// if (i == 0) break;
					// }
					// printf("66666666666%d\n", 7);
					int i = prefetch_tensors.size();
					// while(true);
					for (auto it = prefetch_tensors.begin(); it != prefetch_tensors.end(); it++) {
						// printf("start backward swap_ctrl tensor%d for layer%d\n", (*it)->get_tensor_id(), (*it)->get_layer_id());
						swap_ctrl(*it, model_type, BACKWARD, NULL, &backward_pur_swap_times[i]);
						// swap_ctrl(*it, model_type, BACKWARD, NULL, NULL);						
						// printf("end backward swap_ctrl tensor%d for layer%d\n", (*it)->get_tensor_id(), (*it)->get_layer_id());
						// if (i == 0) break;
						i--;
					}
					
					// printf("end %d backward swap_ctrl %d\n", 666);
					double pur_swap_time = 0.0;
					for (size_t i = SWAP_BLOCK_NUM-1; i < prefetch_tensors.size(); i++) {
						// prefetch_tensors[i]->set_swap_time(FETCH);
						pur_swap_time += backward_pur_swap_times[i];
						// bpst[i] = bpst[i] + backward_pur_swap_times[i];
					}
					printf(" **** backward pur swap time = %f ****\n", pur_swap_time);
					
					while(true) {
						printf("");
						if (training_state == FORWARD) {
							// mem_controller.reset_swap_block();
							break;	
						}
					}
				}
#endif
#endif
                /*
				tmp = query_used_mem();
                if (tmp > max_usage) {
                    max_usage = tmp;
                }
                if (get_cur_time() - ts > 0.5) {
                    printf("QUERY====>max gpu usage : %f MB\n", BYTE_TO_MB(max_usage));
                    ts = get_cur_time();
                    max_usage = 0;
                }
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
				*/
            }
        });

        set_cpu_affinity(query_thread->native_handle(), -2);
    }

public:
	// 初始化一个网络会初始化一个mem_controller，一个liveness分析
    network_t(base_solver_t<value_type>* _solver):is_network_computable(false), solver(_solver), clip_gradient_limit(35.0), test_iter(0)
    {
        // google::InitGoogleLogging("");
        FLAGS_logtostderr = 1;
		
		trainning_set_flag = 0;

        reg = new registry_t<value_type>();

#ifdef LIVENESS
        printf("LIVENESS !!!!!\n");
#endif
#ifdef RECOMPUTE_ON
        printf("RECOMPUTE_ON !!!!!!!!\n");
#endif
#ifdef LARGER
        printf("LARGER !!!!\n");
#endif
#ifdef LRU_ON
        printf("LRU_ON !!!\n");
#endif
#ifdef BLASX_MALLOC
        printf("BLASX_MALLOC !!!!\n");
#endif
        //set affinity
        set_main_thread_cpu_affinity(1);

#ifdef SWAP_ON
        init_swap_managemnet(0);
#endif
        is_forward_setup  = false;
        is_backward_setup = false;
        is_testing_ready  = false;

        switch (sizeof(value_type))
        {
            case 2: cudnn_data_type = CUDNN_DATA_HALF;   break;
            case 4: cudnn_data_type = CUDNN_DATA_FLOAT;  break;
            case 8: cudnn_data_type = CUDNN_DATA_DOUBLE; break;
            default : FatalError("Unsupported data type");
        }
		printf("before createHandles, memory cost = %lf\n", BYTE_TO_MB(query_used_mem()));
        createHandles();
		printf("after createHandles, memory cost = %lf\n", BYTE_TO_MB(query_used_mem()));
    };

    ~network_t()
    {
        // we first destory data layer because the ParallelReader must be destoryed before registry
        delete train_data_layer;
        delete test_data_layer;

        delete reg;

        query_stop = true;
        query_thread->join();
        //the sequence matters
        destroyHandles();
        //all tensors will be deleted in the registry class

        // finish all computation, destroy compute stream
        stream_singleton::destory_stream();

        // destroy global blasx_malloc_t
        blasx_gpu_singleton::destroy_all_instance();
    }
	
	int get_max_layer_id() {
		return this->max_layer_id;
	}
	
	void print_layers_time(std::map<int, double>* layers_time, char* remake) {
        double sum = 0;
		std::map<int, void* >* net_layers = reg->get_net_layers_ptr();
        for (auto it = layers_time->begin(); it != layers_time->end(); it++) {
			int layer_id = it->first;
			base_layer_t<value_type>* layer = (base_layer_t<value_type>*)net_layers->find(layer_id)->second;
            LAYER layer_type = layer->get_layer_type();
			std::string type_name;
			
            if (layer_type == CONV) type_name.assign("CONV");
            else if (layer_type == POOL) type_name.assign("POOL");
            else if (layer_type == ACT) type_name.assign("ACT");
            else if (layer_type == BN) type_name.assign("BN");
            else if (layer_type == FC) type_name.assign("FC");
            else if (layer_type == LRN) type_name.assign("LRN");
            else if (layer_type == PADDING) type_name.assign("PADDING");
            else if (layer_type == DROPOUT) type_name.assign("DROPOUT");
            else if (layer_type == SOFTMAX) type_name.assign("SOFTMAX");
            else if (layer_type == CONCAT) type_name.assign("CONCAT");
            else if (layer_type == FORK_L) type_name.assign("FORK_L");
            else if (layer_type == JOIN_L) type_name.assign("JOIN_L");
			else if (layer_type == RNN) type_name.assign("RNN");
			else if (layer_type == DATA_L) type_name.assign("DATA_L");
            else { type_name.assign("555"); }
			printf("layer%d %s ", it->first, type_name.c_str());
            printf("%s ", remake);
            printf("time = %lf\n", it->second);
			
            sum += it->second;
        }
        printf("%s ", remake);
        printf("total time = %lf\n", sum);
    }
	
	void init_network_trainning(size_t batch_size, int read_flag, 
		preprocessor<value_type>* processor, parallel_reader_t<value_type> *reader,
        base_layer_t<value_type> *first_layer, base_layer_t<value_type> *last_layer,
		const char *train_image_bin, const char *train_label_bin, const char *train_mean_file) {
		this->batchsize = batch_size;
        this->reader =  reader;
        this->processor = processor;
        // this->train_data_layer = first_layer;
        this->loss_layer = last_layer;
		
		if (trainning_set_flag == 0) {
		    max_layer_id = loss_layer->get_base_id();
		    printf("\nmax_layer_id = %d\n", max_layer_id);
		    trainning_set_flag = 1;
		}
		else {
		    (this->get_registry())->delete_all_related_record();
			this->clear_data_loss_layer();
		}
		reg->print_outputs();
        printf("before setup in network.h mem usage = %zd = %f\n", query_used_mem(), BYTE_TO_MB(query_used_mem()));
		// ((data_layer_t<value_type>*)train_data_layer)->data_fake_init();
		fsetup(first_layer);
        printf("after fsetup in network.h mem usage = %f\n", BYTE_TO_MB(query_used_mem()));
		// while(1); 
		bsetup(loss_layer);
        printf("after bsetup in network.h inherent_size = %f\n", BYTE_TO_MB(query_used_mem()));       

#define DEBUG
#ifdef DEBUG
		int tensor_id;
		int tensor_id2;
		std::vector<tensor_t<value_type>* > tensors;
		base_network_layer_t<value_type>* net_layer;
		base_structure_t<value_type>* structure_layer;
        auto net_layers = reg->get_net_layers();
        auto route = reg->get_net_comp_route();
		std::vector<base_layer_t<value_type>*> next_layers;
		std::vector<base_layer_t<value_type>*> prev_layers;
        for (size_t i = 0; i < route.size(); i++) {
			if(route[i].second == FORWARD) 
			{
				base_layer_t<value_type>* layer = (base_layer_t<value_type>*)((net_layers.find(route[i].first))->second);
				LAYER layer_type = layer->get_layer_type();
				std::string type_name;
				if (layer_type == FORK_L) {
					structure_layer = (base_structure_t<value_type>*)layer;
					tensors = structure_layer->get_outputs();
					type_name.assign("FORK_L");
					next_layers = layer->get_next();
					printf("(%d, %s, %d, next is ", layer->get_base_id(), type_name.c_str(), route[i].second);
					for (int j = 0; j < next_layers.size(); j++) {
						// next_layer_id[j] = next_layers[j]->get_base_id();
						printf("%d ", next_layers[j]->get_base_id());
					}
					printf("tensors: ");
					for (int j = 0; j < tensors.size(); j++) {
						printf("%d ", tensors[j]->get_tensor_id());
					}
					printf(")->");
				}
				else if (layer_type == JOIN_L || layer_type == CONCAT) {
					if (layer_type == JOIN_L) type_name.assign("JOIN_L");
					else type_name.assign("CONCAT");
					prev_layers = layer->get_prev();
					printf("(%d, %s, %d, pre is ", layer->get_base_id(), type_name.c_str(), route[i].second);
					for (int j = 0; j < prev_layers.size(); j++) {
						// next_layer_id[j] = next_layers[j]->get_base_id();
						printf("%d ", prev_layers[j]->get_base_id());
					}
					structure_layer = (base_structure_t<value_type>*)layer;
					tensors = structure_layer->get_inputs();
					printf("ins: ");
					for (int j = 0; j < tensors.size(); j++) {
						printf("%d ", tensors[j]->get_tensor_id());
					}
					printf(")->");
				}
				else {
					net_layer = (base_network_layer_t<value_type>*)layer;
					tensor_id = net_layer->get_f_out()->get_tensor_id();
					if (i == 0) tensor_id2 = -1;
					else tensor_id2 = net_layer->get_f_in()->get_tensor_id();
					if (layer_type == CONV) type_name.assign("CONV");
					else if (layer_type == POOL) type_name.assign("POOL");
					else if (layer_type == ACT) type_name.assign("ACT");
					else if (layer_type == BN) type_name.assign("BN");
					else if (layer_type == FC) type_name.assign("FC");
					else if (layer_type == LRN) type_name.assign("LRN");
					else if (layer_type == PADDING) type_name.assign("PADDING");
					else if (layer_type == DROPOUT) type_name.assign("DROPOUT");
					else if (layer_type == SOFTMAX) type_name.assign("SOFTMAX");
					else if (layer_type == CONCAT) type_name.assign("CONCAT");
					else if (layer_type == DATA_L) type_name.assign("DATA");
					else if (layer_type == RNN) type_name.assign("RNN");
					else if (layer_type == SATTN) type_name.assign("SATTN");
					else {}
					printf("(%d, %s in:%d out:%d, %d)->", layer->get_base_id(), type_name.c_str(), tensor_id2, tensor_id, route[i].second);
				} 
			} 
        }
        printf("\n\n\n"); 
#endif 
#undef DEBUG
	}
	
    mem_controller_t<value_type>* get_mem_controller() {
        return &(this->mem_controller);
    }

    math_util<value_type>* get_math_util() {
        return &(this->math);
    }

    cudnnHandle_t* get_cudnn_handle() {
        return &(this->cudnn_handle);
    }

    cublasHandle_t* get_cublas_handle() {
        return &(this->cublas_handle);
    }

    registry_t<value_type>* get_registry() {
        return this->reg;
    }
	
	void clear_data_loss_layer() {
		this->train_data_layer = NULL;
		this->softmax_loss_layer = NULL;
	}
	
	inline base_layer_t<value_type>* get_last_layer() {
		return this->softmax_loss_layer;
	}
	
	// fsetup_kernel 可获得前向传播的执行顺序net_comp_route，以及每层所依赖   的tensor
    void fsetup(base_layer_t<value_type>* start_layer) {
        // printf("fsetup_kernel(start_layer) start\n");
        if(this->train_data_layer == NULL) {
            // printf("fsetup_kernel(start_layer) start\n");
            this->train_data_layer = start_layer;
        } else {
            printf("fsetup train data layer could only be set once!! line 12@network.cpp\n");
            exit(1);
        }
        
        fsetup_kernel(start_layer);
        this->is_forward_setup = true;
    }

    void bsetup(base_layer_t<value_type>* end_layer) {
        if(this->softmax_loss_layer == NULL) {
            this->softmax_loss_layer = end_layer;
        } else {
            printf("bsetup softmax layer could only be set once!! line 43@network.cpp\n");
            exit(1);
        }
        bsetup_kernel(end_layer);
        this->is_backward_setup = true;

        meta_setup();
    }

    void setup_test(base_layer_t<value_type>* test_data_layer, size_t iter);

	value_type forward(network_stage stage) {
        assert(this->train_data_layer != NULL);
		training_state = FORWARD;
		start_flag = true;
        base_layer_t<value_type>* n = this->train_data_layer;
        std::vector<value_type> loss;
        forward_kernel(stage, n, &loss);
        return loss[0];
    }

    void free_shared_memory() {
        mem_controller.free_tensor_shared_memory();
    }
	
	void test_output_swap_time(base_layer_t<value_type>* layer, net_comp dir);
	void test_output_swap_time_v2(base_layer_t<value_type>* layer, net_comp dir);
	
	double test_swap_time(tensor_t<value_type>* tensor, SWAP_DIR); 
	double test_swap_time_v2(tensor_t<value_type>* tensor, base_layer_t<value_type>* layer, SWAP_DIR); 

    value_type simulated_forward(network_stage stage, 
		std::map<int, double> *layers_ft) {
			
		// printf("sssssss");
        assert(this->train_data_layer != NULL);
		// printf("ddddddd\n");
        base_layer_t<value_type>* n = this->train_data_layer;
        std::vector<value_type> loss;
        // mem_controller.init_tensor_shared_memory();
        simulated_forward_kernel(stage, n, &loss, layers_ft);
        return loss[0];
    }
	
    void backward() {
        assert(this->softmax_loss_layer != NULL);
		training_state = BACKWARD;
        base_layer_t<value_type>* n = this->softmax_loss_layer;
        backward_kernel(n);
    }

	void simulated_backward(std::map<int, double> *layers_bt) {
        assert(this->softmax_loss_layer != NULL);
        base_layer_t<value_type>* n = this->softmax_loss_layer;
        simulated_backward_kernel(n, layers_bt);
    }

    void backward_with_update(size_t iter) {
        assert(this->softmax_loss_layer != NULL);

        base_layer_t<value_type>* n = this->softmax_loss_layer;
        backward_with_update_kernel(n, iter);
    }
    
    void update( size_t iter) {
        assert(this->train_data_layer != NULL);
        base_layer_t<value_type>* start = this->train_data_layer;
        update_kernel(start, iter);
    }

	void grad_zero();
	
	void select_swap_layer_by_route(std::vector<int>* swap_layers_by_route, size_t sn);
	void select_swap_layer(std::vector<int>* swap_layers, size_t sn);
	void select_recompute_layer_rnn(std::vector<int>* recompute_layers, size_t rn);
    void select_recompute_layer(std::vector<int>* recompute_layers, size_t rn);
    size_t set_recompute_layer(std::vector<int>* recompute_layers);

    size_t set_swap_layer(std::vector<int>* swap_layers);

	size_t GetLayerNum();

	bool SetRecomputeSwapTensorsbyRoute(int* code, size_t code_size);

	void simulated_train(size_t iter, std::map<int, double> *layers_ft, std::map<int, double> *layers_bt) {
		assert(is_forward_setup == true);
        assert(is_testing_ready == true);
        assert(is_backward_setup == true);
		
		int abandon = this->abandon;
        size_t curt_mem = query_used_mem();
        printf("after setup the memory used:%f\n", BYTE_TO_MB(curt_mem));
		
		value_type loss = 0;
        value_type running_std     = 0;
        value_type running_average = 0;
        value_type threshold       = 0;
        std::deque<value_type> loss_queue;
        double speed_start = get_cur_time();
		// printf("max_layer_id = %d\n", max_layer_id);
		std::map<int, double> temp_layers_ft;
		std::map<int, double> temp_layers_bt;
		/*
		for (LAYER type = CONV; type <= JOIN_L; type++) {
			// layer_type_ft[type] = 0.0;
			// layer_type_bt[type] = 0.0;
		}
		*/
		layer_type_ft[RNN] = 0.0, layer_type_bt[RNN] = 0.0, layer_type_num[RNN] = 0, layer_type_mem[RNN] = 0.0, layer_type_b_mem[RNN] = 0.0;
		layer_type_ft[CONV] = 0.0, layer_type_bt[CONV] = 0.0, layer_type_num[CONV] = 0, layer_type_mem[CONV] = 0.0, layer_type_b_mem[CONV] = 0.0;
		layer_type_ft[POOL] = 0.0, layer_type_bt[POOL] = 0.0, layer_type_num[POOL] = 0, layer_type_mem[POOL] = 0.0, layer_type_b_mem[POOL] = 0.0;
		layer_type_ft[ACT] = 0.0, layer_type_bt[ACT] = 0.0, layer_type_num[ACT] = 0, layer_type_mem[ACT] = 0.0, layer_type_b_mem[ACT] = 0.0;
		layer_type_ft[BN] = 0.0, layer_type_bt[BN] = 0.0, layer_type_num[BN] = 0, layer_type_mem[BN] = 0.0, layer_type_b_mem[BN] = 0.0;
		layer_type_ft[FC] = 0.0, layer_type_bt[FC] = 0.0, layer_type_num[FC] = 0, layer_type_mem[FC] = 0.0, layer_type_b_mem[FC] = 0.0;
		layer_type_ft[LRN] = 0.0, layer_type_bt[LRN] = 0.0, layer_type_num[LRN] = 0, layer_type_mem[LRN] = 0.0, layer_type_b_mem[LRN] = 0.0;
		layer_type_ft[PADDING] = 0.0, layer_type_bt[PADDING] = 0.0, layer_type_num[PADDING] = 0, layer_type_mem[PADDING] = 0.0, layer_type_b_mem[PADDING] = 0.0;
		layer_type_ft[DATA_L] = 0.0, layer_type_bt[DATA_L] = 0.0, layer_type_num[DATA_L] = 0, layer_type_mem[DATA_L] = 0.0, layer_type_b_mem[DATA_L] = 0.0;
		layer_type_ft[DROPOUT] = 0.0, layer_type_bt[DROPOUT] = 0.0, layer_type_num[DROPOUT] = 0, layer_type_mem[DROPOUT] = 0.0, layer_type_b_mem[DROPOUT] = 0.0;
		layer_type_ft[SOFTMAX] = 0.0, layer_type_bt[SOFTMAX] = 0.0, layer_type_num[SOFTMAX] = 0, layer_type_mem[SOFTMAX] = 0.0, layer_type_b_mem[SOFTMAX] = 0.0;
		layer_type_ft[CONCAT] = 0.0, layer_type_bt[CONCAT] = 0.0, layer_type_num[CONCAT] = 0, layer_type_mem[CONCAT] = 0.0, layer_type_b_mem[CONCAT] = 0.0;
		layer_type_ft[FORK_L] = 0.0, layer_type_bt[FORK_L] = 0.0, layer_type_num[FORK_L] = 0, layer_type_mem[FORK_L] = 0.0, layer_type_b_mem[FORK_L] = 0.0;
		layer_type_ft[JOIN_L] = 0.0, layer_type_bt[JOIN_L] = 0.0, layer_type_num[JOIN_L] = 0, layer_type_mem[JOIN_L] = 0.0, layer_type_b_mem[JOIN_L] = 0.0;

		for (int j = 1; j <= max_layer_id; j++) {
			(*layers_ft)[j] = 0.0;
			(*layers_bt)[j] = 0.0;
		}
		/*--network simulate--*/
        double start, end, f_end;
        double tttt = 0;
		double tf = 0;
		double tb = 0;
#ifdef FAKE_TRAIN
		printf("before data_fake_init %d\n", 666);
		((data_layer_t<value_type>*)train_data_layer)->data_fake_init();
		printf("after data_fake_init %d\n", 666);
#endif
        mem_controller.init_simulator_memory();
		printf("after init_simulator_memory the memory used:%f\n", BYTE_TO_MB(query_used_mem()));
		cur_swap_time_test_times = 0;
		for (int i = 1; i <= iter; i++) {
            start = get_cur_time();
			loss = simulated_forward(NET_TRAIN, &temp_layers_ft);
			f_end = get_cur_time();
			simulated_backward(&temp_layers_bt);
            end = get_cur_time();
			cur_swap_time_test_times++;
			for (int j = 1; j <= max_layer_id; j++) {
				if(this->abandon <= 0) {
					(*layers_ft)[j] += temp_layers_ft[j];
					(*layers_bt)[j] += temp_layers_bt[j];
				}
			}
            size_t curt_mem = query_used_mem();
			if (i%10 == 0) {
				printf("after iter%d the memory used:%f\n", i, BYTE_TO_MB(query_used_mem()));
				printf("iter%d time = %lf\n", i, end - start);
				printf("-----iter:%zu--lr:%f--loss:%f\n", i, solver->get_lr(), loss);
			}
			this->abandon--;
            // printf("after iter%d the memory used:%f\n", i, BYTE_TO_MB(curt_mem));
		}
        double time = 0;
		for (int j = 1; j <= max_layer_id; j++) {
            time += (*layers_ft)[j];
            time += (*layers_bt)[j];
			(*layers_ft)[j] = (*layers_ft)[j] / (double)(iter-abandon);
			(*layers_bt)[j] = (*layers_bt)[j] / (double)(iter-abandon);
			tf += (*layers_ft)[j];
			tb += (*layers_bt)[j];
		}
		time = time / (double)(iter-abandon);
        
		// mem_controller.free_tensor_shared_memory();
		for(auto it = (this->layers_ft).begin(); it != (this->layers_ft).end(); it++) {
			it->second = it->second / (double)(iter-abandon);
		}
		for(auto it = (this->layers_bt).begin(); it != (this->layers_bt).end(); it++) {
			it->second = it->second / (double)(iter-abandon);
		}
		for(auto it = (this->layers_ft2).begin(); it != (this->layers_ft2).end(); it++) {
			it->second = it->second / (double)(iter-abandon);
		}
		for(auto it = (this->layers_bt2).begin(); it != (this->layers_bt2).end(); it++) {
			it->second = it->second / (double)(iter-abandon);
		}
        // print_layers_time(&(this->layers_ft), "forward");
		// print_layers_time(&(this->layers_bt), "backward");
		// print_layers_time(&(this->layers_ft2), "forward2");
		// print_layers_time(&(this->layers_bt2), "backward2");
		printf(" average iter_time = %lf, time = %lf, memory used = %f\n", time, time, BYTE_TO_MB(curt_mem));
		printf(" average forward_time = %lf, average backward_time = %lf\n", tf, tb);
		printf("through_put = %f\n", (double)batchsize/(time/(double)(iter-abandon)));
	#ifdef DEBUG
        for (auto it = layer_type_ft.begin(); it != layer_type_ft.end(); it++) {
            printf("%lf\t%lf\n", it->second/(double)iter, it->second);
        }
        printf("layer_type_bt:\n");
        for (auto it = layer_type_bt.begin(); it != layer_type_bt.end(); it++) {
            printf("%lf\t%lf\n", it->second/(double)iter, it->second);
        }
        printf("layer_type_num:\n");
        for (auto it = layer_type_num.begin(); it != layer_type_num.end(); it++) {
            printf("%d\n", it->second/iter);
        }
        printf("layer_type_mem:\n");
        for (auto it = layer_type_mem.begin(); it != layer_type_mem.end(); it++) {
            printf("%lf\t%lf\n", it->second/(double)iter, it->second);
        }
        printf("layer_type_b_mem:\n");
        for (auto it = layer_type_b_mem.begin(); it != layer_type_b_mem.end(); it++) {
            printf("%lf\t%lf\n", it->second/(double)iter, it->second);
        }
	#endif
		this->abandon = 0;
		mem_controller.free_gpu_mem_pool();
	} 

    void train(size_t iter, size_t tracking_window, size_t test_interval, network_saver *saver=NULL) {
        assert(is_forward_setup == true);
        assert(is_testing_ready == true);
        assert(is_backward_setup == true);

#ifdef GRAD_ACC
#ifdef BIG_BATCHSIZE
		this->big_batchsize = BIG_BATCHSIZE;
#endif
#endif
		
        size_t curt_mem = query_used_mem();
        printf("after setup the memory used:%f\n", BYTE_TO_MB(query_used_mem()));
		// exit(0);
        value_type loss = 0;
		value_type avg_loss = 0.0;
        value_type running_std     = 0;
        value_type running_average = 0;
        value_type threshold       = 0;
        std::deque<value_type> loss_queue;
        double speed_start = get_cur_time();
// #define POOL_MALLOC_MODE
#ifdef POOL_MALLOC_MODE
		size_t net_total_size;
		size_t gpu_tensors_size;
		size_t gpu_pool_size;
		size_t swap_tensors_size;
		size_t swap_pool_size;
		size_t recompute_tensor_size;
		size_t recompute_pool_size;
        size_t max_grad_size;
		size_t b_data_pool_size;
		size_t QKV_buffer_size;
		size_t max_data_size;
		size_t max_tensor_size;
		size_t max_workspace_size;
		size_t max_fragment_size;
		size_t max_layer_size;
		reg->get_size_info(
			&net_total_size, &gpu_tensors_size, &gpu_pool_size, 
			&swap_tensors_size, &swap_pool_size, &recompute_tensor_size, &recompute_pool_size, &max_fragment_size,
			&max_grad_size, &max_tensor_size, &max_data_size, &max_layer_size,
			&b_data_pool_size, &QKV_buffer_size, &max_workspace_size);
		mem_controller.init_all_tensors(
			gpu_pool_size, swap_pool_size, recompute_pool_size, 
			max_grad_size, max_data_size, b_data_pool_size, QKV_buffer_size, max_workspace_size);
		printf("total_men_requirement = %zd = %f\n", net_total_size, BYTE_TO_MB(net_total_size));
#ifdef FAKE_TRAIN
		void* pool;
		size_t pool_offset;
		mem_controller.get_pool(&pool, &pool_offset);
		((data_layer_t<value_type>*)train_data_layer)->data_fake_init_pool_malloc_mode(pool, &pool_offset);
		mem_controller.set_pool(pool_offset);
		
#endif	
		
#else
		printf("before init_all_tensor_gpu_memory the memory used:%f\n", BYTE_TO_MB(query_used_mem()));
        // mem_controller.init_all_tensor_gpu_memory();
#ifdef FAKE_TRAIN
		((data_layer_t<value_type>*)train_data_layer)->data_fake_init();
#endif   
#endif
// #undef POOL_MALLOC_MODE
		printf("before trainning the memory used = %zd = %f\n", query_used_mem(), BYTE_TO_MB(query_used_mem()));
        // tensor_t<value_type>* temp_tensor  = new tensor_t<value_type>( 1, 1, 1, 1, reg->get_vector(), DATA, 0);
		// printf("tensor counter = %d\n", temp_tensor->get_tensor_base_id());
		int abandon = this->abandon;
		double start, end;
        double f_end, b_end;
        double fb_time = 0, forward_time = 0;
        double iter_start, iter_time = 0;
		double start_u;
		double update_time = 0;
		int updata_flag;
		int iter_times;
		double big_batch_start, big_batch_end;	
        double loss_avg = 0.0;
		double batch_time = 0.0;
        double avg_batch_time = 0.0;
		int big_batch_num = 0;
		std::vector<value_type> avg_loss_list;
        
		for (size_t i = 0; i < swap_tensors.size(); i++) {
			fpst.push_back(0);
			bpst.push_back(0);
		}
		
		if (big_batchsize > 0) { // 55
			iter_times = big_batchsize / batchsize;
			iter_times = (big_batchsize % batchsize) > (batchsize/2) ? iter_times+1 : iter_times;
			big_batchsize = iter_times * batchsize;
			solver->set_grad_alpha((double)1.0/(double)iter_times);
		}
        int cn = iter_times;
		big_batch_start = get_cur_time();
		// while(1);

#ifdef MULTI_SWAP_BLOCK
		if (mem_controller.PreAllocateRecompteSwapBlock(&swap_tensors, &prefetch_tensors) == false) {
			printf("FALSE to PreAllocateRecompteSwapBlock\n");
			exit(1);
		}
		// printf("swap_tensors.size() = %d, prefetch_tensors.size() = %d\n", this->swap_tensors.size(), this->prefetch_tensors.size());
		// for (size_t i = 0; i < swap_tensors.size(); i++) {
		// 	printf("layer%d swap_tensor%d swap_block_id = %d\n", 
		// 		swap_tensors[i]->get_layer_id(), swap_tensors[i]->get_tensor_id(), swap_tensors[i]->get_swap_block_id());
		// }
		// for (size_t i = 0; i < prefetch_tensors.size(); i++) {
		// 	printf("layer%d prefetch_tensor%d swap_block_id = %d\n", 
		// 		prefetch_tensors[i]->get_layer_id(), prefetch_tensors[i]->get_tensor_id(), prefetch_tensors[i]->get_prefetch_block_id());
		// }
		
		// std::vector<std::pair<int, net_comp> > net_comp = reg->get_net_comp_route();
		// std::map<int, void* > layers = reg->get_net_layers();
		// for (size_t i = 0; i < net_comp.size(); i++) {
		// 	if (net_comp[i].second == FORWARD) {
		// 		size_t layer_id = net_comp[i].first;
		// 		base_layer_t<value_type>* layer = (base_layer_t<value_type>*)layers.find(layer_id)->second;
		// 		if (layer->get_layer_position() == SWAP_LAYER) {
		// 			printf("route%d layer%d-type%d is SWAP_LAYER\n", i, layer->get_base_id(), layer->get_layer_type());
		// 		}
		// 		else if(layer->get_layer_position() == RECOMPUTE_LAYER) {
		// 			printf("route%d layer%d-type%d is RECOMPUTE_LAYER\n", i, layer->get_base_id(), layer->get_layer_type());
		// 		}
		// 		else {
		// 			printf("route%d layer%d-type%d\n", i, layer->get_base_id(), layer->get_layer_type());
		// 		}
		// 	}
		// }
#endif
        for(size_t i = 1; i <= iter; i++) {
			printf("\n iter%d \n", i);
            iter_start = get_cur_time();
			start = get_cur_time();
			loss = forward(NET_TRAIN);
			printf("\n iter%d forward done, time = %f, loss = %f\n", i, get_cur_time()-start, loss);
			// while(1);
            f_end = get_cur_time();
            backward();
			printf("\n iter%d backward done, time = %f, loss = %f\n", i, get_cur_time()-f_end, loss);
            b_end = get_cur_time();
            start_u = get_cur_time();
			avg_loss += loss;
			if (this->big_batchsize == 0) { // 
				update(i);     // 				
				grad_zero();   // 
				printf("update %d\n", i);
			}
			else {
				if (i%iter_times == 0) {  // 
					avg_loss = avg_loss / (value_type)iter_times;
					avg_loss_list.push_back(avg_loss);
					update(i/iter_times);     // 					
					grad_zero();   // 
					big_batch_end = get_cur_time();
					printf("\nGrad Acc Updata, Batchsize = %d, iter = %zd, Batch time = %lf, avg_loss = %f\n\n", 
							big_batchsize, i, big_batch_end - big_batch_start, avg_loss);
					big_batch_start = big_batch_end;
					avg_loss = 0.0;
					batch_time = big_batch_end - big_batch_start;
					if (this->abandon <= 0) {
						avg_batch_time += batch_time;
						big_batch_num++;
					}
				}
			}
            update_time += get_cur_time() - b_end;
			end = get_cur_time();
            printf("iter%d sub-batch time = %f, ft = %f, bt = %f, ut = %f, ", i, end - iter_start, f_end - iter_start, b_end - f_end, get_cur_time() - b_end);
            printf("lr = %f, loss = %f, ", i, solver->get_lr(), loss);
            printf("memory used = %f\n", i, BYTE_TO_MB(query_used_mem()));
			this->abandon--;
			if (this->abandon <= 0) {
				fb_time += end - iter_start;
				forward_time += f_end - iter_start;
			}
			// while(1);
            // continue;
            //backward_with_update(i);
            /*----loss statistics----*/  
			/*
            if(loss_queue.size() < tracking_window) {
                if (std::isfinite(loss)) {
                    loss_queue.push_back(loss);
                    running_average = ((i-1)*running_average+loss)/(i);
                }
            } else {
                value_type loss_to_go = loss_queue.front();
                running_average = (running_average*tracking_window - loss_to_go + loss)/tracking_window;
                loss_queue.pop_front();
                loss_queue.push_back(loss);
            }
            running_std = 0;
            for(unsigned i = 0; i < loss_queue.size(); i++) {
                running_std += ((loss_queue[i] - running_average)*(loss_queue[i] - running_average));
            }
            running_std        = sqrt(running_std / (value_type) loss_queue.size());
            threshold = running_average + 3*running_std;
			*/
            // if (i % 1 == 0) {
            //     double speed_end   = get_cur_time();
            //     double speed_time  = speed_end - speed_start;
            //     size_t batch_size  = ((data_layer_t<value_type>*) train_data_layer)->get_batch_size();
            //     double train_imgs  = batch_size*20.0f;
            //     double train_speed = train_imgs / speed_time;
            //     speed_start = get_cur_time();
            //     time_t tt = time(NULL);
            //     tm* t= localtime(&tt);
            //     printf("%d-%02d-%02d %02d:%02d:%02d-----iter:%zu--lr:%.10f--loss:%f--avg:%f--std:%f--threshold:%f--%f:img/s\n",
            //            t->tm_year + 1900,
            //            t->tm_mon + 1,
            //            t->tm_mday,
            //            t->tm_hour,
            //            t->tm_min,
            //            t->tm_sec,
            //            i, solver->get_lr(), loss, running_average, running_std, threshold, train_speed
			// 	);
            // }
            // double iter_end = get_cur_time();
            // if(i % test_interval == 0) {
            //     // test();
            // }
            // solver->update_lr(i, running_average);
        }
		start_flag = false;
        // file.close();
		// for(auto it = layers_ft.begin(); it != layers_ft.end(); it++) {
		// 	it->second = it->second / (double)(iter-abandon);
		// }
		// for(auto it = layers_bt.begin(); it != layers_bt.end(); it++) {
		// 	it->second = it->second / (double)(iter-abandon);
		// }
		// for(auto it = layers_ft2.begin(); it != layers_ft2.end(); it++) {
		// 	it->second = it->second / (double)(iter-abandon);
		// }
		// for(auto it = layers_bt2.begin(); it != layers_bt2.end(); it++) {
		// 	it->second = it->second / (double)(iter-abandon);
		// }
        // print_layers_time(&(this->layers_ft), "forward");
		// print_layers_time(&(this->layers_bt), "backward");
		// print_layers_time(&(this->layers_ft2), "forward2");
		// print_layers_time(&(this->layers_bt2), "backward2");
		update_time = update_time / (double)(iter-abandon);
        iter_time = iter_time / (double)(iter-abandon);
        fb_time = fb_time / (double)(iter-abandon);
        forward_time = forward_time / (double)(iter-abandon);
		printf("\nSub-batchsize = %d, memory cost = %f, ", this->batchsize, BYTE_TO_MB(query_used_mem()));
        printf("average iter time = %lf, avg_batch_time = %lf, average through_output = %lf\n", fb_time, avg_batch_time/(double)big_batch_num, (double)this->batchsize/fb_time);
        printf("average forward time = %lf, average backward time = %lf\n", forward_time, fb_time - forward_time);
		/*
		for (size_t i = 0; i < layers_ft2.size(); i++) {
			std::vector<std::pair<int, net_comp> >* net_comp_route = reg->get_net_comp_route_ptr();
			std::map<int, void* >* net_layers  = reg->get_net_layers_ptr();
			for (int i = 0; i < net_comp_route->size(); i++) {
				
			}
		}
		for (size_t i = 0; i < fpst.size(); i++) {
			fpst[i] = fpst[i] / (double)iter;
			bpst[i] = bpst[i] / (double)iter;
			printf("fpst[%d]-mem=%zd-time=%lf, bpst[%d]-mem=%zd-time=%lf\n",
				i, swap_tensors[i]->get_mem_size(), fpst[i],
				i, prefetch_tensors[i]->get_mem_size(), bpst[i]);
		}
		for (size_t i = 0; i < fpst.size(); i++) {
			fpst[i] = fpst[i] / (double)iter;
			bpst[i] = bpst[i] / (double)iter;
			printf("fpst[%d]-mem=%zd-time=%lf, bpst[%d]-mem=%zd-time=%lf\n",
				i, swap_tensors[i]->get_mem_size(), fpst[i],
				i, prefetch_tensors[i]->get_mem_size(), bpst[i]);
		}
		*/
		for (int n = 0; n < avg_loss_list.size(); n++) {
			printf("%f ", avg_loss_list[n]);
		}
		printf("\n");
	}

    void gradient_check(int layer_id);

};



}
#endif // _NETWORK_H_


