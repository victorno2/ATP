
#include <thread>
#include <tensor.h>
#include <registry.h>
#include <util/common.h>
#include <liveness.h>
// #include <recompute.h>
#include <tensor.h>
#include <mem_control.h>
#include <solver.h>

#include <deque>
#include <layer/base_layer.h>
#include <layer/base_network_layer.h>
#include <layer/base_structure_layer.h>
#include <layer/data_layer.h>
#include <util/ATP_math.h>
#include <stream_singleton.h>
#include <gpu_malloc.h>
#include <string>
#include <network.h>

#include <vector>
#include <map>

namespace ATP {

template <class value_type>
class ATPSearch {
private:
    int a;
    int b;

	// double max_tp;
	// size_t best_batch_size;
	// int no_update_times;

	/***** strage ******/
	double pcie_bandwith;
	size_t gpu_total_mem;
	size_t gpu_free_mem;
	size_t inherent_size = 0;

	size_t batch_size;
	size_t best_batchsize;
	size_t min_offloadsize;
	size_t pool_size = SIZE_MAX;
	std::vector<int> swap_layers;
	std::vector<bool> comp_route_swap;
	std::vector<double> valid_time;
	std::vector<double> occupy_time;
	int swap_num;
	
	std::map<int, int> layers_offload;  // map<layer_id, put its' out in offload_list after which layer finish in net_comp_route > 
	std::map<int, std::vector<tensor_t<value_type>* >> free_timing;  // map< id in net_comp_route, tensors >
	std::map<int, std::map<int, size_t>> outs_size;  // <batchsize, <layer_id, output_size>>
    std::map<int, std::map<int, size_t>> params_size;  // <batchsize, <layer_id, params_size>>
    std::map<int, std::map<int, size_t>> conv_buffs_size;  // <batchsize, <layer_id, conv_buffs_size>>
    // std::map<int, std::map<int, size_t>> layers_size;  // <batchsize, <layer_id, layers_size>>
	std::map<int, size_t> total_men_by_batchsize;  // map<batchsize, total_menory_cost>
	std::map< int, std::vector< size_t > > layers_size_by_batchsize;
	std::map<int, size_t> excessive_size_by_batchsize;
	std::map<int, size_t> ideal_recompute_size_by_batchsize;
	std::map<int, size_t> ideal_offload_size_by_batchsize;
	std::map<int, double> ideal_offload_time_by_batchsize;
	std::map<int, double> ideal_swapping_time_by_batchsize;
	std::map<int, double> forward_computing_time_by_batchsize;
	std::map<int, double> backward_computing_time_by_batchsize;
	std::map<int, double> ideal_computing_time_by_batchsize;
	std::map<int, double> ideal_recomputation_time_by_batchsize;
	std::map<int, double> ideal_iter_time_by_batchsize;
	std::map<int, double> ideal_throughput_by_batchsize;
	std::map<int, double> ideal_throughput_by_batchsize_without_swap;
	
	std::vector<tensor_t<value_type>* > swap_tensors;
	std::vector<tensor_t<value_type>* > prefetch_tensors;
	std::vector<tensor_t<value_type>* > alternative_swap_tensors;
	std::vector<tensor_t<value_type>* > alternative_tensors;
	std::vector<tensor_t<value_type>* > recompute_tensors;
	size_t min_swapping_size;
	size_t min_recomputing_size;

	std::map<int, double> iter_computing_time_by_batchsize;
	std::map<int, double> iter_forward_time_by_batchsize;
	std::map<int, double> iter_backward_time_by_batchsize;
	std::map<int, double> layers_ft;
    std::map<int, double> layers_bt;
	std::map<int, size_t> layers_output_size;
	
	std::vector<tensor_t<value_type>*> generation_time_order;

	/***** controller *****/
	registry_t<value_type> *reg;
	liveness_analysis_t<value_type> *liveness;
	network_t<value_type> *net;
	mem_controller_t<value_type> *mem_controller;

	/***** offload ******/
	

	/***** simulator *****/
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
    base_solver_t<value_type> *solver;
	
	typedef enum PRE_OPERATE {
		REMAINED	= 0,
		SWAPPED		= 1,
		RECOMPUTED	= 2
	}PRE_OPERATE;
	
	typedef struct MIN_GENERATE {
		PRE_OPERATE pre_operate;
		double min_generation_time;
	} MIN_GENERATE;

	typedef struct GENERATE_EFFICIENCY {
		PRE_OPERATE operation;
		double generation_efficiency;
	} GENERATE_EFFICIENCY;

public:
    ATPSearch(network_t<value_type> *net) {	
		printf("\nbefore swap done\n");	
		this->net = net;
		this->mem_controller = (this->net)->get_mem_controller();
		this->liveness = (this->mem_controller)->get_liveness_analysis_t();
		this->reg = net->get_registry();
		printf("\ncreate swap done\n");
	}

	size_t GetMaxThroughputBatchSize();

	void ResetSimulateTrainMemory();

	void ResetThroughputModelLastRecord();

	size_t GetLayerNum() {
		return reg->get_net_layers_ptr()->size();
	}

	void ResetTensorPosition();

	void GetRecomputeSwapTensorScheme();

	bool ThroughputUpperBound_v2(size_t batch_size, size_t* MR, size_t* MS, 
		std::vector<std::pair<tensor_t<value_type>*, GENERATE_EFFICIENCY*>>* gf_list, 
		std::vector<tensor_t<value_type>*>* recompute_tensors,
		std::vector<tensor_t<value_type>*>* alternative_swap_tensors,
		double* tpub, double* iter_time, double* recomp_time, double* ideal_swap_time, 
		double* tf, double* tb, double* f_time, double* b_time);

	bool IsRecomputingTensorLegal(tensor_t<value_type>* tensor);

	bool DistributeMRMS_v2(size_t batch_size, std::vector<std::pair<tensor_t<value_type>*, GENERATE_EFFICIENCY*>>* gf_list, size_t* MR, size_t* MS);

	bool DistributeMSSwapOnly(size_t batch_size, std::vector<std::pair<tensor_t<value_type>*, GENERATE_EFFICIENCY*>>* gf_list, size_t* MR, size_t* MS);

	bool SelectRecomputingTensor_v2(size_t* MR, std::vector<std::pair<tensor_t<value_type>*, GENERATE_EFFICIENCY*>>* gf_list, std::vector<tensor_t<value_type>*>* recompute_tensors);

	bool SelectRecomputingTensor(size_t* MR, std::vector<std::pair<tensor_t<value_type>*, MIN_GENERATE*>>* gt_list, std::vector<tensor_t<value_type>*>* recompute_tensors);

	void GetAlternativeSwappingTensors(std::vector<std::pair<tensor_t<value_type>*, GENERATE_EFFICIENCY*>>* gf_list, std::vector<tensor_t<value_type>*>* recompute_tensors, std::vector<tensor_t<value_type>*>* alternative_swap_tensors);

	void GetAlternativeTensors(std::vector<tensor_t<value_type>*>* alternative_tensors);

	void QuickSortGenerationTime(std::vector<std::pair<tensor_t<value_type>*, MIN_GENERATE*>>* gt_list, int low, int high);

	void QuickSortGenerationEfficiency(std::vector<std::pair<tensor_t<value_type>*, GENERATE_EFFICIENCY*>>* gf_list, int low, int high);

	void printGenerationTime();

	void GetMinGenerationTime(tensor_t<value_type>* tensor, MIN_GENERATE* min_generation);
	
	void SortGenerationTimeList(std::vector<std::pair<tensor_t<value_type>*, MIN_GENERATE*>>* gt_list);
 
	void GetMaxGenerationEfficiency(tensor_t<value_type>* tensor, GENERATE_EFFICIENCY* max_generation_efficiency);
 
	void SortGenerationEfficiencyList(std::vector<std::pair<tensor_t<value_type>*, GENERATE_EFFICIENCY*>>* gf_list);
	
	bool GetRecomputeTime(std::map<int, void* >* net_layers, tensor_t<value_type>* tensor, base_layer_t<value_type>* layer, net_comp dir, double* re_time);
	
	size_t SetSwappingTensors(bool* swapping_code, std::vector<tensor_t<value_type>*>* alternative_swap_tensors, 
		std::vector<tensor_t<value_type>*>* swap_tensors, std::vector<tensor_t<value_type>*>* prefetch_tensors);
	
	bool IsSwappingLegal(std::vector<tensor_t<value_type>*>* swap_tensors, std::vector<tensor_t<value_type>*>* prefetch_tensors);
	
	bool IterationTimeEvaluator(std::vector<tensor_t<value_type>* >* recompute_tensors, 
		std::vector<tensor_t<value_type>* >* swap_tensors, double* forward, double* sync_time_f, double* offload_time,
		std::vector<tensor_t<value_type>* >* prefetch_tensors, double* backward, double* sync_time_b, double* fetch_time);
 
	bool GetIterationTimeGASwappingCode(bool* swapping_code, double* iter_time, double* sync_time, double* swapping_size_error, bool display);
 
	void GetVDNNSchemeCode();
 
	size_t GetAlternativeSwappingTensorsNumber() {
		return this->alternative_swap_tensors.size();
	}
	
	size_t GetAlternativeTensorsNumber() {
		return this->alternative_tensors.size();
	}
 
	bool GetIterationTimePSOCode012(int* pso012_code, double* iter_time, double* sync_time, double* swapping_size_error, bool display);
 
	size_t SetTensorPolicy(int* pso012_code, std::vector<tensor_t<value_type>*>* swap_tensors, std::vector<tensor_t<value_type>*>* prefetch_tensors);
 
	bool IsPSOcode012Legal(std::vector<tensor_t<value_type>*>* swap_tensors, std::vector<tensor_t<value_type>*>* prefetch_tensors);
 
	bool MinSavedMem(size_t batch_size);
 
	void set_inherent_size(size_t x) {
		this->inherent_size = x;
	}
	
	inline double scalar_relu(double x) {
		return x > 0 ? x : 0;
	}
	
	void get_swap_layers_from_ga(bool comp_route_swap[], int size);
	
	void get_swap_layers_from_ga_v2(bool* comp_route_swap, int size);
	
	size_t get_pool_size();
	
	bool is_swap_layers_legal(bool* comp_route_swap, int size);
	
	bool is_swap_layers_legal_v2(bool* comp_route_swap, size_t* offload_size, double* offload_size_error, bool revis);
	
	bool can_layer_swap(base_layer_t<value_type>* layer) {
		return true;
	}
	
	size_t offload_size_by_swap_layers_pool_malloc_mode(bool* comp_route_swap, int size);
	size_t offload_size_by_swap_layers(bool* comp_route_swap, int size);
	
	bool forward_time_by_swap_layers_v2(
		std::vector<tensor_t<value_type>* >* swap_tensors, double* forward, double* sync_time_f, double* offload_time,
		std::vector<tensor_t<value_type>* >* prefetch_tensors, double* backward, double* sync_time_b, double* fetch_time);
	
	bool iter_time_by_swap_layers( 
		bool* comp_route_swap, 
		double* offload_size_error, double* iter_time, double* forward, double* backward, 
		double* sync_time_f, double* sync_time_b, double* offload_time, double* fetch_time);
	
	size_t set_swap_tensors(bool* comp_route_swap);
	
	double forward_time_by_swap_layers(bool* comp_route_swap, int size, double* sync_time);
	
	double invalid_time_v2(bool* comp_route_swap, int size);
	
	double invalid_time();
	
	void set_output_tensor_id();

	// 取显存
	void get_total_mem(int batchsize);

	size_t get_offload_requirement() {
		return ideal_offload_size_by_batchsize[best_batchsize];//  + pool_size;
	}

	double offload_size_error(bool comp_route_swap[], int size);

	// 在该batchsize下，超出了多少显存
	void set_excessive_size_by_batchsize(int batchsize);

	// 在该batchsize下，理想的卸载时间
	void set_ideal_offload_time_by_batchsize(int batchsize);

	// batchsize下的理想训练时间
	void set_ideal_iter_time_by_batchsize(int batchsize);

	// 特征图大小
	void get_outs_size(int batchsize);

	// 参数大小
	void get_params_size(int batchsize);

	// 取最优的batchsize
	void get_best_batchsize(void);

	// 模拟训练（测试次数，batchsize数）
	void simulate_trainning_iter(int iter, int batchsize);

	// batchsize下的总内存消耗
	void set_total_size_by_batchsize(int batchsize);

	void select_swap_layer_v2(bool *swap_flag_comp_route);

	void select_swap_layer();
	
	void print_swap_layers();
	
	std::vector<int>* get_swap_layers() {
		return &swap_layers;
	}
	
	// 模拟训练配置
	void set_simulator(size_t gpu_total_mem, double pcie_bandwith) {
		this->gpu_total_mem = gpu_total_mem;
		this->pcie_bandwith = pcie_bandwith;
	}

	void get_best_config(double* max_throughput, size_t* batchsize, size_t* min_offloadsize);

	double GetThroughput(size_t batch_size);

	// void GetThroughputPeak(size_t* batch_size, double* max_tp) {
	// 	*max_tp = this->max_tp;
	// 	*batch_size = this->best_batch_size;
	// }

	// bool UpdateThroughputPeak(size_t batch_size, int no_update_win);

	void set_simulator_trainning(
		preprocessor<value_type>* processor, parallel_reader_t<value_type> *reader, network_t<value_type> *net, 
		size_t batch_size, base_layer_t<value_type> *first_layer, base_layer_t<value_type> *last_layer,
		const char *train_image_bin, const char *train_label_bin, const char *train_mean_file
	);

	bool simulate_trainning(network_t<value_type> *net, int batchsize, int iter);

	// 模拟训练统计

	void ideal_trainning_statistics(
		char *train_image_bin,
		char *train_label_bin,
		char *train_mean_file,

		std::map<size_t*, int> *input_iter,
		int read_flag, 
		preprocessor<value_type>* processor,
		parallel_reader_t<value_type> *reader,
		base_layer_t<value_type> *first_layer,
		base_layer_t<value_type> *loss_layer
	);

    ~ATPSearch() { }
    
    void swap_test(void);
	
	void vdnn_conv(std::vector<int>* swap_layers, int* swap_num);
	size_t modnn_max_window_mem_by_batchsize(int batchsize);
	void modnn_batch_select(int start_batchsize, int end_batchsize);
    size_t modnn_offload_size_by_batchsize(int batchsize);
	
	void printTimeList();
	void printSwapInfo(const char* str);
	
	void test_code();
};

}
