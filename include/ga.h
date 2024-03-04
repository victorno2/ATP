
#ifndef ATP_GA_H
#define ATP_GA_H


#include <stdlib.h>
#include <vector>

namespace ATP {

size_t rand_seed = 0;
int ind_count = 0;

class individual   //个体结构体
{
public:
    int id;
    int id2;
	bool *chrom = NULL;     //染色体x1
	double fit = 0;             //适应度
	double prob;            //选中概率
    double t_norm;     
    int times;              // 选择次数	
    int len;	
	individual(int l) {
        this->len = l;
        id = ind_count;
        id2 = ind_count;
        ind_count++;
		chrom = (bool*)malloc(sizeof(bool)*len);
		// printf("chrom = ");
		for (int i = 0; i < len; i++) {
		 	*(chrom+i) = false;
		}
		init_v2();
	}
	~individual() {
		free(chrom);
	}

    void init() {
        *(chrom+0) = false;
        *(chrom+1) =  true;
        *(chrom+2) = false;
        *(chrom+len-3) = false;
        *(chrom+len-2) = true;
        *(chrom+len-1) = false;
    }

    void init_v2() {
        // *(chrom+0) = false;
        #ifdef HEAD_END_SWAP
            size_t mask = SWAP_BLOCK_NUM;
        #else
            size_t mask = 1;
        #endif
        for (int i = 0; i < mask; i++) {
            *(chrom+i) = true;
        }
        // *(chrom+len-1) = false;
        for (int i = this->len-mask; i < len; i++ ) {
            *(chrom+i) = true;
        }
    }

    void print_individual(bool display_chrom) {
        int sum = 0;
        printf("individual%d:\n", id);
        if (display_chrom) {
        printf("Chrom: ");
            for (int i = 0; i < len; i++) {
                printf("%d,", *(chrom+i));
                if (*(chrom+i)) {
                    sum += 1;
                }
            }
        }
        printf("\nlen = %d, %d%% bits = 1, fit = %lf, prob = %lf\n", len, (int)(((double)sum/(double)len)*100.0), fit, prob);
    }
};

class ga {
private:
    double iter_time_best = 1000;
    int pop_size;  // 种群大小，随便定
    int len;  // 染色体长度
    double mutate_prob = 0.01;  // 变异概率
    individual* best_ind;
    double sync_max;

	std::vector<individual*> population;

    double fit_sum;

    ATPSearch<float>* optim;

    size_t pool_size;
    double aver_swaping_error = 0.0;
    double aver_iter_time = 0.0;
    int good_ind_counter = 0;
    double worst_iter_time = 1000.0;
    void init_sync_max() {
        individual* temp_ind = new individual(this->len);
        mutate(temp_ind, 1.0);
        bool flag = optim->is_swap_layers_legal(temp_ind->chrom, this->len);
        double forward_time = optim->forward_time_by_swap_layers(temp_ind->chrom, this->len, &this->sync_max);
        printf("sync_max = %lf\n", this->sync_max);
        // offload_size_error = optim->offload_size_error(population[i]->chrom ,this->len);
    }

    void individual_data_copy(individual *dis, individual *src) {
        for (int i = 0; i < len; i++) {
            dis->chrom[i] = src->chrom[i];
        }
    }

    void individual_copy(individual *dis, individual *src) {
        // printf("copy src:%x to dis:%x\n", src, dis);
        dis->id2 = src->id2;
		dis->fit = src->fit;
        dis->prob = src->prob;
        dis->times = src->times;
        for (int i = 0; i < len; i++) {
            dis->chrom[i] = src->chrom[i];
        }
		// printf("copy src:%x to dis:%x down\n", src, dis);
    }

    double individual_fitness(individual *ind) {
        double forward_time;
        double offload_size_error = 0;
        double size_fit;
        printf("ind%d --- ", ind->id, forward_time, size_fit, ind->fit);
        double sync_time;
        while(1) {
            individual* test_ind = new individual(this->len);
            bool test_chome[this->len] = {0,1,0,0,1,0,0,0,0,1,0,0,1,0,0,0,1,0,0,1,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,1,0,0,1,0,0,0,0,1,0,0,0,1,0,0,1,0,0,0,0,0,0,0,0,0,1,0,0,1,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,1,0,1,0,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,0,0,0,1,0,0,1,0,0,0,0,0,0,1,0,1,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,1,0,0,1,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,1,0,1,0,0,1,0,0,0,0,0,0,0,0,0,1,0,0,0,1,0,0,1,0,0,1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,1,0,0,1,0,0,1,0,0,0,1,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,1,0,1,0,0,0,0,0,1,0,1,0,1,0,0,1,0,1,0,1,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,1,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,1,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,1,0,0,0,1,0,1,0,0,1,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,1,0,0,1,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,1,0,0,0,1,0,0,1,0,0,0,0,0,0,1,0,0,0,1,0,0,1,0,1,0,0,0,1,0,0,0,0,0,0,0,0,0,1,0,0,0,0,1,0,0,1,0,0,1,0,1,0,1,0,0,1,0,1,0,0,0,0,1,0,0,0,1,0,0,0,0,0,0,0,0,0,1,0,0,0,1,0,0,0,1,0,0,1,0,0,1,0,1,0,1,0,0,0,0,0,0,1,0,0,0,1,0,0,0,1,0,1,0,0,0,0,0,0,0,1,0,0,1,0,0,0,0,1,0,1,0,1,0,0,0,0,1,0,0,0,0,1,0,0,1,0,0,1,0,0,0,0,1,0,0,0,0,0,1,0,0,1,0,1,0,0,0,0,0,1,0,1,0,1,0,1,0,1,0,1,0,0,0,1,0,0,1,0,0,1,0,1,0,1,0,0,0,1,0,0,0,0,0,0,0,1,0,0,1,0,1,0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,1,0,0,1,0,1,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,1,0,0,1,0,0,0,0,1,0,1,0,0,0,0,1,0,0,1,0,0,0,0,0,1,0,0,0,1,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,1,0,0,0,0,0,1,0,0,0,1,0,0,1,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,1,0,1,0,0,1,0,0,0,0,0,1,0,0,1,0,1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,1,0,1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,1,0,1,0,1,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,1,0,0,1,0,1,0,0,0,0,0,0,0,0,0,1,0,0,1,0,1,0,1,0,0,0,1,0,1,0,0,1,0,0,0,1,0,0,0,0,0,1,0,1,0,0,0,0,1,0,0,1,0,0,1,0,1,0,1,0,1,0,0,0,0,0,0,0,0,0,1,0,0,1,0,0,0,1,0,0,0,0,0,0,1,0,0,0,1,0,1,0,1,0,0,0,0,0,0,0,0,1,0,0,0,0,1,0,1,0,0,0,1,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,1,0,0,0,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,1,0,0,1,0,0,0,1,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,1,0,0,1,0,0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,1,0,1,0,1,0,0,1,0,1,0,0,1,0,1,0,0,0,0,0,0,0,0,1,0,1,0,0,0,0,1,0,0,1,0,1,0,0,0,1,0,0,0,0,1,0,1,0,1,0,0,0,0,1,0,0,0,0,1,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,1,0,0,1,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,1,0,0,0,0,1,0,1,0,0,0,1,0,0,1,0,0,1,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,1,0,0,1,0,0,0,0,0,0,0,1,0,0,1,0,0,0,0,1,0,1,0,0,1,0,0,0,0,0,0,0,0,1,0,1,0,0,0,0,0,1,0,0,1,0,0,1,0,0,1,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,1,0,1,0,1,0,0,1,0,1,0,1,0,0,1,0,1,0,1,0,0,1,0,0,0,0,1,0,1,0,0,1,0,0,1,0,0,0,0,0,0,0,0,0,1,0,1,0,0,1,0,0,0,1,0,0,1,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,1,0,1,0,0,1,0,0,1,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,1,0,1,0,1,0,0,0,0,0,0,0,1,0,0,0,0,0,1,0,1,0,0,0,0,0,1,0,1,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,1,0,1,0,1,0,0,1,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,1,0,1,0,0,0,0,0,0,0,0,0,0,1,0,0,1,0,0,1,0,1,0,1,0,0,0,0,1,0,0,0,0,1,0,0,0,1,0,0,1,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,1,0,0,0,0,0,0,1,0,1,0,1,0,0,0,0,0,0,0,1,0,0,1,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,1,0,0,1,0,0,1,0,0,0,0,0,0,0,0,0,1,0,0,1,0,1,0,0,1,0,1,0,1,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,1,0,1,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,1,0,1,0,1,0,0,0,1,0,0,0,0,0,1,0,1,0,1,0,0,1,0,0,0,0,0,0,0,1,0,0,1,0,0,0,0,1,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0,1,0,0,1,0,0,0,0,1,0,0,1,0,0,1,0,0,0,1,0,0,0,0,0,0,0,0,0,0,1,0,0,0,1,0,0,0,0,0,0,1,0,1,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,1,0,1,0,0,1,0,0,0,1,0,1,0,1,0,1,0,0,1,0,0,1,0,1,0,0,0,0,0,0,0,0,0,0,0,1,0,0,1,0,0,0,0,1,0,0,0,1,0,0,1,0,1,0,0,0,0,1,0,0,1,0,0,1,0,0,1,0,0,0,0,1,0,0,0,1,0,1,0,1,0,0,0,0,1,0,1,0,1,0,0,1,0,1,0,0,0,1,0,1,0,0,1,0,1,0,0,1,0,0,0,0,0,0,0,1,0,0,0,0,0,1,0,0,1,0,0,0,0,0,1,0,0,1,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,1,0,1,0,1,0,1,0,0,0,0,1,0,1,0,0,0,0,1,0,0,1,0,0,0,0,0,0,0,0,0,1,0,0,0,1,0,1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,1,0,1,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,1,0,1,0,0,0,1,0,0,0,1,0,0,0,0,0,0,0,0,0,1,0,1,0,0,0,0,0,1,0,0,1,0,0,1,0,1,0,1,0,0,0,1,0,0,0,1,0,1,0,1,0,1,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,1,0,1,0,0,0,0,1,0,0,0,1,0,0,1,0,0,0,0,0,0,1,0,0,0,1,0,0,1,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,1,0,0,1,0,0,0,1,0,1,0,1,0,0,0,0,1,0,0,0,0,0,1,0,0,1,0,0,0,1,0,0,1,0,0,0,1,0,0,0,0,0,1,0,0,0,0,1,0,1,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,1,0,1,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,1,0,1,0,0,0,1,0,0,0,1,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,1,0,0,1,0,0,0,1,0,0,1,0,1,0,1,0,0,0,0,1,0,1,0,1,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,1,0,1,0,0,0,0,1,0,1,0,1,0,0,1,0,1,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,1,0,0,1,0,0,0,0,0,0,1,0,0,0,0,1,0,0,1,0,0,1,0,0,0,0,1,0,1,0,0,1,0,0,0,0,1,0,1,0,1,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,1,0,0,0,0,1,0,0,0,0,0,0,0,0,1,0,0,1,0,1,0,0,0,0,0,1,0,0,1,0,0,0,1,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,1,0,0,0,0,0,0,0,1,0,0,1,0,0,0,1,0,1,0,1,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,1,0,1,0,0,1,0,0,0,1,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,1,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,1,0,1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1,0,0,1,0,0,1,0,0,1,0,0,0,0,0,1,0,0,1,0,0,0,1,0,0,0,1,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,1,0,0,1,0,0,0,1,0,0,1,0,1,0,0,0,1,0,1,0,1,0,1,0,0,0,0,1,0,1,0,0,0,0,0,0,0,1,0,0,1,0};
            test_ind->chrom = test_chome;
            // optim->is_swap_layers_legal(test_chome, this->len);
            printf("test pass, is_swap_layers_legal result = %d\n", optim->is_swap_layers_legal(test_chome, this->len));
            printf("chrom = ");
            for (int i = 0; i < this->len; i++) {
                printf("%d,", test_chome[i]);
            }
            printf("\n");
            
            forward_time = optim->forward_time_by_swap_layers(test_chome ,this->len, &sync_time);
            // printf("test pass, forward_time_by_swap_layers result = %f\n", optim->forward_time_by_swap_layers(test_chome, this->len,  &sync_time));
            offload_size_error = optim->offload_size_error(test_chome, this->len);
            // printf("test pass, offload_size_error result = %f\n", optim->offload_size_error(test_chome, this->len));
            if (offload_size_error == 0) {
                    size_fit = 1;
            }
            else {
                size_fit = (10.0 - offload_size_error) / 10.0;
                    // size_fit = offload_size_error
            }
            ind->fit = 1.0/forward_time + 15*size_fit; // + 1.0/offload_size_error;
            printf("ind%d, forward_time = %lf, size_fit = %f, sync_time = %lf, fit = %lf\n", ind->id, forward_time, size_fit, sync_time, ind->fit);
            exit(-1);
        }
        
        ind->print_individual(true);
        printf("is_swap_layers_legal result = %d\n", optim->is_swap_layers_legal(ind->chrom, this->len));
        printf("forward_time_by_swap_layers result = %f\n", optim->forward_time_by_swap_layers(ind->chrom ,this->len, &sync_time));
        if (optim->is_swap_layers_legal(ind->chrom, this->len)) {
            printf("is_swap_layers_legal result = %d\n", optim->is_swap_layers_legal(ind->chrom, this->len));
            forward_time = optim->forward_time_by_swap_layers(ind->chrom ,this->len, &sync_time);
            printf("forward_time_by_swap_layers result = %f\n", forward_time);
            offload_size_error = optim->offload_size_error(ind->chrom ,this->len);
            // ind->fit = 1.0/forward_time; // + 1.0/offload_size_error;
            if (offload_size_error == 0) {
                size_fit = 1;
            }
            else {
                size_fit = (10.0 - offload_size_error) / 10.0;
                // size_fit = offload_size_error
            }
            ind->fit = 1.0/forward_time + 5*size_fit; // + 1.0/offload_size_error;
        }
        else {
            ind->fit = 0;
        }
        if (ind->fit < 0) {
            ind->fit = 0;
        }
        // printf("ind%d, forward_time = %lf, size_fit = %f, fit = %lf\n", ind->id, forward_time, size_fit, ind->fit);
        // double ind.fit = 1.0 / optim->invalid_time_v2(ind.chrom ,this->len);
        return ind->fit;
    }

    double set_prob(int g) {
        double iter_time = 0.0;
        double fit_sum = 0;
        double forward_time = 0.0;
        double backward_time = 0.0;
        double sync_time = 0.0;
        double sync_time_f = 0.0;
        double sync_time_b = 0.0;
        double offload_size_error = 0;
        double size_fit;
        double offload_time;
        double fetch_time;
        bool flag;
        double t_norm_sum = 0.0;
        
        this->aver_iter_time = 0.0;
        this->good_ind_counter = 0;
        this-> worst_iter_time = 0.0;
        #ifdef ATP_SOLUTION
        for (int i = 0; i < this->pop_size; i++) {
            // printf("population[%d] start\n", i);
            // population[i]->print_individual(true);
            // optim->GetRecomputeSwapTensorScheme();
            flag = optim->GetIterationTimeGASwappingCode(population[i]->chrom, &iter_time, &sync_time, &offload_size_error, false);

            if(flag) {
                population[i]->fit = 1.0 / (iter_time + 5.0*offload_size_error);
                this->aver_iter_time += iter_time;
                this->aver_swaping_error += offload_size_error;
                this->good_ind_counter++;
                if (this->worst_iter_time < iter_time) {
                    this->worst_iter_time = iter_time;
                }
            }
            else {
                population[i]->fit = 0;
            }
            fit_sum += population[i]->fit;
            // printf("Fitness = %lf\n", population[i]->fit);
        }
        // printf("Probability:\n");
        for (int i = 0; i < this->pop_size; i++) {
            population[i]->prob = population[i]->fit / fit_sum;
            // printf("ind%d-p:%lf, ", i, population[i]->prob);
        }
        // for (int i = 0; i < this->pop_size; i++) {
        //     population[i]->print_individual(true);
        // }

        #endif
        #ifdef SWAPADVISOR

            for (int i = 0; i < this->pop_size; i++) {
                flag = optim->GetIterationTimeGASwappingCode(population[i]->chrom, &iter_time, &sync_time, &offload_size_error, false);
                if(flag) {
                    this->aver_swaping_error += offload_size_error;
                    this->aver_iter_time += iter_time;
                    this->good_ind_counter++;
                    population[i]->fit = 1.0 / (iter_time);
                    if (iter_time_best > iter_time) {
                        iter_time_best = iter_time; 
                    }
                    if (this->worst_iter_time < iter_time) {
                        this->worst_iter_time = iter_time;
                    }
                    population[i]->t_norm = (iter_time_best-iter_time) / iter_time_best;
                }
                else {
                    population[i]->fit = 0;
                    population[i]->t_norm = -1000;
                }
                fit_sum += population[i]->fit;
                t_norm_sum += exp(population[i]->t_norm);
                for (int i = 0; i < this->pop_size; i++) {
                    population[i]->prob = exp(population[i]->t_norm) / t_norm_sum;
                    // printf("ind%d-p:%lf, ", i, population[i]->prob);
                }
                // printf("Fitness = %lf\n", population[i]->fit);
            }
            // this->aver_iter_time = this->aver_iter_time / (double)this->good_ind_counter;
            // this->aver_swaping_error = this->aver_swaping_error / (double)this->good_ind_counter;

            // for (int i = 0; i < this->pop_size; i++) {
            // flag = optim->GetIterationTimeGASwappingCode(population[i]->chrom, &iter_time, &sync_time, &offload_size_error, false);
            // if(flag) {
            //     this->aver_iter_time += iter_time;
            //     this->aver_swaping_error += offload_size_error;
            //     this->good_ind_counter++;
            //     population[i]->fit = 1.0 / (iter_time);
            //     if (iter_time_best > iter_time) {
            //         iter_time_best = iter_time; 
            //     }
            //     population[i]->t_norm = (iter_time_best-iter_time) / iter_time_best;
            // }
            // else {
            //     population[i]->fit = 0;
            //     population[i]->t_norm = -1000;
            // }
            // fit_sum += population[i]->fit;
            // t_norm_sum += exp(population[i]->t_norm);
            // for (int i = 0; i < this->pop_size; i++) {
            //     population[i]->prob = exp(population[i]->t_norm) / t_norm_sum;
            //     // printf("ind%d-p:%lf, ", i, population[i]->prob);
            // }
            // printf("Fitness = %lf\n", population[i]->fit);
            // }
        #endif
        this->aver_iter_time = this->aver_iter_time / (double)this->good_ind_counter;
        this->aver_swaping_error = this->aver_swaping_error / (double)this->good_ind_counter;
        printf("aver_iter_time = %f, good_ind_counter = %d\n", aver_iter_time, good_ind_counter);
        // printf("\n"); 
        return fit_sum;
    }

    bool constraint(individual ind) {
        return optim->is_swap_layers_legal(ind.chrom ,this->len);
    }

    void pop_iter() {
		std::vector<individual*> temp_pop;
        // individual temp_pop[pop_size];
        double prob_area[pop_size][2];
        double temp = 0;
		for (int i = 0; i < this->pop_size; i++) {
			temp_pop.push_back(new individual(this->len));
		}
        for (int i = 0; i < this->pop_size; i++) {
            // 计算轮盘区域面积
            prob_area[i][0] = temp;  
            temp += this->population[i]->prob;
            prob_area[i][1] = temp;
            // printf("area%d:[%lf, %lf) ", i, prob_area[i][0], prob_area[i][1]);
        }
        // printf("\n");
        for (int i = 0; i < this->pop_size; i++) {
            this->population[i]->times = 0;
        }
        for (int i = 0; i < this->pop_size; i++) {
            // 记录个体落到轮盘区域的次数
            srand(time(0)+rand_seed);
            rand_seed++;
            temp = (double)rand()/(RAND_MAX+1.0);
            // printf("temp = %lf\n", temp);
            for (int j = 0; j < this->pop_size; j++) {
                if ((temp >= prob_area[j][0]) && (temp < prob_area[j][1])) {
                    // printf("select population[%d]\n", j);
                    individual_copy(temp_pop[i], population[j]);
                    break;
                }
            }
        }

        for (int i = 0; i < this->pop_size; i++) {
            // 得到新的种群
            individual_copy(population[i], temp_pop[i]);
            delete temp_pop[i];
        }
        temp_pop.erase(temp_pop.begin(), temp_pop.end());
    }

    void cross(individual* ind1, individual* ind2, int cross_point) { // 1 0 1 0 1 0 | 0 1 1 1
                                                                      // 1 1 1 0 1 0 | 0 1 0 1

                                                                      // 1 0 1 0 1 0 | 0 1 0 1
                                                                      // 1 1 1 0 1 0 | 0 1 1 1
        bool temp;
        #ifdef HEAD_END_SWAP
            size_t mask = SWAP_BLOCK_NUM;
        #else
            size_t mask = 1;
        #endif
        for (int i = mask; i < this->len-mask; i++) {  // layer 0=false, 1=true, last-1=true, last=false no change, so layer 2&&(last-2) must be false 
            if (i < cross_point) {
                temp = ind1->chrom[i];
                ind1->chrom[i] = ind2->chrom[i];
                ind2->chrom[i] = temp;
            }
        }
    }

    void pop_cross() {
        int cross_point;
        #ifdef HEAD_END_SWAP
            size_t mask = SWAP_BLOCK_NUM;
        #else
            size_t mask = 1;
        #endif
        for (int i = 0; i < (this->pop_size / 2); i++) {
            srand(time(0)+rand_seed);
            rand_seed++;
            cross_point = (rand() % len-mask)+mask;  // layer 0=false, 1=true, last-1=true, last=false no change, so layer 2&&(last-2) must be false 
            // printf("individual %d(%d) and %d(%d) cross start, cross_point = %d\n", 2*i, population[2*i]->id2, 2*i+1, population[2*i+1]->id2, cross_point);
            cross(population[2*i], population[2*i+1], cross_point);
            // printf("individual %d(%d) and %d(%d) cross done\n", 2*i, population[2*i]->id2, 2*i+1, population[2*i+1]->id2);
        }
    }

    void mutate(individual *ind, double mutate_prob) {
        double temp;
        for (int i = 3; i < this->len-3; i++) {  // layer 0=false, 1=true, last-1=true, last=false no change, so layer 2&&(last-2) must be false 
            srand(time(0)+rand_seed);
            rand_seed++;
            temp = (double)rand()/(RAND_MAX+1.0);
            // printf("temp = %lf ", temp);
            if (temp < mutate_prob) {
                ind->chrom[i] = ~ind->chrom[i];
            }
        }
        // while(1);
    }

    void mutate_v2(individual *ind, double mutate_prob) {
        double temp;
        #ifdef HEAD_END_SWAP
            size_t mask = SWAP_BLOCK_NUM;
        #else
            size_t mask = 1;
        #endif
        for (int i = mask; i < this->len-mask; i++) {  // layer 0=false, 1=true, last-1=true, last=false no change, so layer 2&&(last-2) must be false 
            srand(time(0)+rand_seed);
            rand_seed++;
            temp = (double)rand()/(RAND_MAX+1.0);
            // printf("temp = %lf ", temp);
            if (temp < mutate_prob) {
                ind->chrom[i] = ~ind->chrom[i];
            }
        }
    }

    void print_population(int iter) {
        printf("generate %d, the population is:\n", iter);
        for (int i = 0; i < this->pop_size; i++) {
            population[i]->print_individual(true);
        }
    }
    
    void pop_mutate() {
        for (int i = 0; i < this->pop_size; i++) {
            mutate_v2(population[i], this->mutate_prob);
        }
    }

public:

    ga(int pop_size, int len, double mutate_prob, ATPSearch<float>* optim) { 
        this->pop_size = pop_size;
		this->len = len; 
		this->mutate_prob = mutate_prob;
		this->optim = optim;
        this->len = optim->GetAlternativeSwappingTensorsNumber();
        // optim->GetRecomputeSwapTensorScheme();
        // this->optim->printTimeList();
		printf("pop_size = %d, len = %d\n", this->pop_size, this->len);
        // while(1);
        // prob_test();
        for (int i = 0; i < this->pop_size; i++) {
            individual* temp_ind = new individual(this->len);
			// mutate(temp_ind, 1.0*((double)i)*(1.0/(double)pop_size));
            mutate_v2(temp_ind, 1.0*((double)i)*(1.0/(double)pop_size));
            // mutate(temp_ind, 1);
			population.push_back(temp_ind);
        }
        best_ind = new individual(this->len);
        // print_population(0);
        // while(1);
        this->pool_size = optim->get_pool_size();
        // init_sync_max();
        // individual_fitness(population[0]);
    }

    ~ga() {
        for (int i = 0; i < this->pop_size; i++) {
			delete population[i];
        }
		population.erase(population.begin(), population.end());
    }

    void evolution(int iter, double error) {
        // two end condition: iteration times & the error between two generations.
        
        double max_fit = 0;
        double iter_time;
        double forward_time;
        double backward_time;
        double offload_time;
        double fetch_time;
        double sync_time;
        double sync_time_f;
        double sync_time_b;
        size_t offload_size;
        size_t offload_requirement;

        double offload_size_error = 0; 
        double pop_fitness = 0.0;
        int max_fit_id;
        double time_new, time_old;
        time_old = get_cur_time();
        std::vector<double> iter_time_list;
        std::vector<double> offload_size_error_list;
        std::vector<double> sync_time_list;
        std::vector<double> fitness_list;
        std::vector<double> best_fitness;
        std::vector<double> best_iter_time_list;
        std::vector<double> aver_iter_time_list;
        std::vector<double> worst_iter_time_list;
        std::vector<double> best_swapping_error_list;
        std::vector<double> aver_swapping_error_list;
        double best_swapping_error = 0.0;
        double least_iter_time;
        double historical_best_fitness = 0.0;
        int best_g = 0;
        for (int g = 0; g <= iter; g++) {
            printf("befor pop_iter: g = %d\n", g);
            pop_fitness = set_prob(g);
            // printf("ssss\n");
            aver_iter_time_list.push_back(this->aver_iter_time);
            // printf("ssss\n");
            worst_iter_time_list.push_back(this->worst_iter_time);
            // printf("ssss\n");
            // aver_swapping_error_list.push_back(this->aver_swaping_error);
            // while(1);
            // print_population(g);
            printf("\ngenerater%d pop_fitness = %lf  ", g, pop_fitness);

            // 输出这一代的进化结果
            for (int i = 0; i < this->pop_size; i++) {
                if (max_fit < population[i]->fit) {
                    max_fit = population[i]->fit;
                    max_fit_id = i;
                }
            }
            if (population[max_fit_id]->fit > best_ind->fit) {
                individual_copy(best_ind, population[max_fit_id]);
                best_g = g;
                historical_best_fitness = population[max_fit_id]->fit;
                optim->GetIterationTimeGASwappingCode(best_ind->chrom, &iter_time, &sync_time, &offload_size_error, true);
                least_iter_time = iter_time;
                best_swapping_error = offload_size_error;   
            }
            best_iter_time_list.push_back(least_iter_time);
            best_swapping_error_list.push_back(best_swapping_error);
            
            // offload_requirement = optim->get_offload_requirement();  // 获取需要卸载的量
            printf("\n\nThe best swap_code id(%d) in this iteration:\n", max_fit_id);
            optim->GetIterationTimeGASwappingCode(population[max_fit_id]->chrom, &iter_time, &sync_time, &offload_size_error, true);
            iter_time_list.push_back(iter_time);
            offload_size_error_list.push_back(offload_size_error);
            sync_time_list.push_back(sync_time);
            fitness_list.push_back(population[max_fit_id]->fit);
            
            // optim->GetIterationTimeGASwappingCode(population[max_fit_id]->chrom, &iter_time, &sync_time, &offload_size_error, true);
            // population[max_fit_id]->print_individual(false);    // 打印这一代种群中最好的个体
            
            printf("GetLayerNum = %d\n", optim->GetLayerNum());
            // optim->GetRecomputeSwapTensorGACode(population[max_fit_id]->chrom, rs_code_layers, rs_code_route);
            // population[max_fit_id]->print_individual(true);
            optim->GetRecomputeSwapTensorScheme();
            pop_iter();
            // print_population(g);
            pop_cross();
            // printf("aftet pop_cross: g = %d\n", g);
            pop_mutate(); 
            // printf("aftet pop_mutate: g = %d\n", g);
            // pop_iter();
            double time_new = get_cur_time();
            printf("ga evolution cost time = %lf\n", time_new - time_old);
            time_old = time_new;
        }
        printf("\n\nThe global best swap_code is:\n");
        optim->GetIterationTimeGASwappingCode(best_ind->chrom, &iter_time, &sync_time, &offload_size_error, true);
        optim->GetRecomputeSwapTensorScheme();
        int step = 2;
        printf("\niter_time ");
        for (int i = 0; i < iter_time_list.size(); i+=step) {
            printf("%f ", iter_time_list[i]);
        }
        printf("\n\noffload_size_error ");
        for (int i = 0; i < offload_size_error_list.size(); i+=step) {
            printf("%f ", offload_size_error_list[i]);
        }
        printf("\n\nsync_time ");
        for (int i = 0; i < sync_time_list.size(); i+=step) {
            printf("%f ", sync_time_list[i]);
        }
        printf("\n\nfitness ");
        for (int i = 0; i < fitness_list.size(); i+=step) {
            printf("%f ", fitness_list[i]);
        }
        printf("\nBest ");
        for (int i = 0; i < best_iter_time_list.size(); i+=step) {
            printf("%f ", best_iter_time_list[i]);
        }
        printf("\nAverage ");
        for (int i = 0; i < aver_iter_time_list.size(); i+=step) {
            printf("%f ", aver_iter_time_list[i]);
        }
        printf("\nWorst ");
        for (int i = 0; i < worst_iter_time_list.size(); i+=step) {
            printf("%f ", worst_iter_time_list[i]);
        }
        printf("\n\nbest_swapping_error_list: ");
        for (int i = 0; i < best_swapping_error_list.size(); i+=step) {
            printf("%f ", best_swapping_error_list[i]);
        }
        // printf("\n\naver_swapping_error_list: ");
        // for (int i = 0; i < aver_swapping_error_list.size(); i+=step) {
        //     printf("%f ", aver_swapping_error_list[i]);
        // }
        printf("\n\nbest_g = %d\n", best_g);
        printf("%f\n%f\n%f\n%f\n", least_iter_time, aver_iter_time_list[aver_iter_time_list.size()-1], worst_iter_time_list[worst_iter_time_list.size()-1], best_swapping_error);
        // delete rs_code_layers;
        // delete rs_code_route;

    }

    void prob_test() {
        individual* ind_test = new individual(this->len);
        bool temp[this->len] = {0,1,1,1,0,1,1,1,0,1,0,0,1,1,0,1,1,1,1,1,1,0,0,1,1,1,1,1,0,0,1,0,0,1,1,0,1,1,0,0,1,1,0,0,1,0,0,0,1,1,1,1,0,1,0,1,1,1,1,1,0,0,1,0,0,1,1,0,0,1,1,1,1,1,1,0,1,1,1,1,1,1,0,1,0,1,0,1,0,0,1,0,0,0,1,0,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,1,0,1,1,1,1,0,1,1,0,0,1,1,1,1,1,1,0,1,0,1,1,1,0,0,1,1,1,0,1,1,0,1,0,0,1,1,1,1,1,0,0,0,1,0,1,1,1,0,0,1,1,0,1,1,0,0,1,0,1,1,1,1,0,1,1,0,1,1,1,1,1,1,0,1,1,1,0,1,1,1,1,1,1,1,1,1,1,1,1,0,1,1,1,0,1,1,1,1,1,0,0,0,1,0,0,1,0,1,0,1,0,0,1,1,0,1,1,0,0,1,1,0,0,1,1,0,1,1,1,1,1,1,1,1,1,1,0,1,0,1,1,1,0,0,1,1,1,0,1,0,1,1,0,1,1,1,1,1,1,1,0,0,1,0,1,1,1,1,0,1,1,0,1,1,1,1,1,1,1,1,1,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,1,0,0,1,1,1,1,1,0,0,1,1,1,0,1,1,0,1,1,1,0,1,0,0,0,1,1,1,1,1,1,1,1,0,1,1,1,1,1,1,0,0,1,1,1,1,1,0,0,1,1,1,1,1,1,1,0,1,0,0,1,0,1,0,1,0,1,1,0,1,1,1,1,1,1,0,1,1,1,0,0,1,1,1,1,1,0,1,1,1,1,1,1,1,0,1,1,0,1,1,0,0,1,1,0,1,1,1,1,1,1,0,1,1,0,0,1,1,1,0,1,0,0,0,1,0,0,1,0,1,0,1,1,0,1,1,1,1,1,0,1,1,1,1,0,1,0,1,1,1,1,1,1,1,1,1,1,1,1,1,0,1,1,1,1,1,1,1,0,1,1,0,0,1,0,0,0,1,0,0,1,0,0,0,1,0,0,1,1,0,0,1,1,0,1,1,0,0,1,1,1,1,0,0,0,1,0,0,1,1,0,0,1,1,1,1,1,1,1,1,0,0,1,1,0,1,1,0,1,1,1,1,1,1,0,0,1,1,1,0,1,1,0,0,1,0,1,1,1,1,1,1,1,0,1,0,0,0,1,0,1,1,1,1,0,1,1,1,1,1,1,1,1,0,0,1,1,0,1,1,1,1,1,1,1,1,1,1,0,0,1,1,1,1,1,1,0,1,0,1,1,1,0,1,1,0,1,1,1,1,0,1,1,0,1,0,1,1,1,0,0,1,0,0,1,1,0,1,1,1,0,0,1,0,0,1,1,0,0,1,1,0,1,1,1,1,1,1,0,0,1,1,1,1,1,1,0,1,1,0,1,0,1,0,1,1,0,1,0,1,1,1,0,1,1,0,1,1,1,1,0,1,1,0,0,1,0,0,1,1,1,0,1,0,0,1,1,1,1,1,0,0,1,1,1,0,1,1,1,1,1,0,0,1,0,0,0,1,0,1,1,0,1,0,1,1,1,1,0,0,1,1,0,0,1,1,0,1,1,0,1,1,1,1,0,1,1,1,1,1,1,0,1,1,0,1,1,1,1,1,1,0,0,1,1,1,1,1,0,1,1,1,0,1,1,1,1,1,1,0,1,0,0,0,1,1,1,1,0,0,0,1,1,0,1,0,1,0,1,1,1,1,1,0,0,1,0,1,1,1,1,1,1,1,0,1,1,0,0,1,1,0,0,1,0,1,1,0,1,0,1,1,1,1,0,0,1,1,1,1,1,0,0,1,1,1,1,1,0,1,1,1,0,0,1,1,0,0,1,0,0,1,1,0,0,1,1,1,1,1,0,0,1,1,1,1,1,1,1,1,0,1,0,1,0,0,1,1,0,0,1,1,0,1,0,1,1,1,0,1,1,0,0,1,1,1,1,1,1,1,1,1,0,0,1,1,0,0,1,1,0,1,1,0,1,1,0,1,1,1,0,0,1,0,1,1,1,0,0,1,0,1,1,1,1,1,1,1,0,1,1,1,1,1,0,1,0,1,0,1,1,0,1,0,1,0,0,1,1,0,1,1,0,1,1,1,0,1,1,0,0,1,1,0,1,1,0,1,1,1,0,1,1,1,1,0,1,1,1,1,0,0,0,1,1,0,1,1,1,1,1,0,1,1,1,0,0,1,1,1,1,1,1,0,1,0,1,1,1,1,1,1,0,1,1,1,0,1,1,0,0,1,1,1,1,1,1,0,1,1,0,1,1,0,0,1,1,0,1,1,1,0,0,1,1,1,1,0,0,1,1,1,0,1,1,1,1,1,0,0,1,0,0,0,1,0,1,1,1,0,0,1,1,0,1,1,1,0,1,1,0,1,1,1,1,1,1,0,1,1,0,0,1,1,0,1,1,1,0,1,1,0,1,1,1,0,1,0,0,0,1,1,1,1,0,1,0,1,0,0,1,0,1,0,1,1,1,1,1,0,1,1,1,1,1,1,0,0,1,0,0,1,1,1,1,1,1,1,1,1,1,0,1,1,1,0,1,0,0,1,0,0,0,1,0,0,1,1,1,1,1,1,1,1,1,0,0,1,1,0,1,1,0,1,1,0,1,1,1,1,0,1,0,0,1,1,1,1,1,0,1,1,1,0,0,1,0,0,0,1,1,0,1,1,1,0,1,1,0,1,1,1,1,1,0,1,1,1,1,1,1,1,0,1,0,0,1,1,1,1,1,1,0,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,1,1,1,1,1,1,0,1,0,0,1,1,1,0,1,0,1,0,1,0,0,1,0,0,1,1,1,0,1,1,1,0,1,0,0,1,1,0,1,1,1,0,1,1,1,0,1,0,0,1,1,0,1,1,0,0,1,1,0,1,1,1,1,1,1,1,0,1,0,1,1,1,0,0,1,1,0,1,1,0,0,1,0,1,0,1,0,1,1,1,1,1,1,1,1,1,0,0,1,1,0,0,1,1,1,1,1,0,1,1,1,0,1,1,0,1,1,1,0,1,1,0,0,1,1,0,0,1,0,1,1,1,0,1,1,1,1,0,1,0,1,1,1,1,0,1,0,1,1,1,1,0,1,0,0,1,1,1,1,1,1,1,1,1,1,0,1,1,0,1,1,1,1,1,0,1,1,1,1,0,1,0,0,1,1,0,0,1,1,0,0,1,0,0,1,0,0,0,1,0,0,1,1,0,0,1,1,1,1,1,0,0,1,1,1,1,1,1,0,1,1,1,1,1,1,1,1,1,1,1,1,0,0,1,1,1,0,1,1,1,1,0,1,1,1,1,1,1,0,0,1,1,0,1,1,0,0,1,1,1,0,1,1,0,1,1,0,1,1,1,0,0,1,1,0,1,1,1,0,1,1,0,1,1,0,1,1,1,1,0,1,0,1,1,0,0,0,1,1,1,1,0,0,1,1,1,1,1,0,1,1,1,1,1,1,1,0,1,1,0,0,1,1,1,0,1,1,0,1,1,1,1,1,0,1,1,1,1,1,1,0,0,1,1,1,1,1,0,1,1,1,0,0,1,0,1,1,1,1,0,1,0,1,1,1,0,0,1,1,1,1,1,1,0,1,1,0,0,1,1,1,1,0,1,0,1,1,1,1,1,1,0,1,0,0,1,1,0,1,1,1,0,1,1,1,1,1,0,1,1,1,0,0,1,1,1,1,1,0,0,1,0,0,1,1,0,0,1,0,0,0,1,1,0,1,1,0,0,1,1,0,1,1,1,1,1,0,1,1,1,0,1,1,1,1,1,1,0,0,1,1,0,1,1,0,1,1,0,1,0,1,1,1,1,0,0,0,1,1,1,1,1,1,0,1,0,0,1,1,0,1,1,1,1,1,0,0,1,1,1,1,1,1,1,0,1,0,0,1,1,1,0,1,0,1,1,1,0,0,1,0,0,0,1,0,0,1,0,1,0,1,0,1,1,1,1,1,1,1,1,1,0,1,1,1,0,0,1,0,0,1,1,0,1,1,0,1,0,1,1,1,1,1,0,0,1,1,0,1,1,1,0,1,1,0,0,1,1,0,1,0,0,1,1,1,0,1,1,0,1,1,0,0,1,1,0,0,1,1,1,1,1,0,0,1,1,0,1,0,0,1,1,1,1,1,0,1,0,1,1,1,1,1,1,0,1,1,0,1,1,1,0,1,0,0,1,1,0,0,1,0,1,1,1,1,1,1,0,0,0,1,0,0,1,0,0,1,1,0,1,1,1,1,0,1,0,0,1,1,0,0,1,0,0,1,1,0,1,1,0,0,1,1,1,1,1,0,0,1,1,1,1,1,0,0,1,1,1,0,1,1,1,0,1,1,0,1,0,0,0,1,1,0,1,0,0,1,1,1,0,1,1,1,0,1,1,1,1,1,1,0,1,1,1,1,1,0,0,1,0,1,1,1,0,1,1,0,0,1,1,0,0,1,0,1,0,1,1,1,1,1,1,0,1,0,1,1,0,0,1,1,0,1,1,0,1,0,1,0,0,1,0,0,1,1,1,0,1,1,0,1,1,0,1,1,1,1,0,1,1,0,1,1,1,0,1,0,0,1,1,0,0,1,1,0,0,1,1,1,1,0,1,0,1,0,0,1,1,0,0,1,0,1,1,0,1,0,1,0,0,1,0,1,0,1,0,0,1,1,0,0,1,0,0,1,1,1,0,1,1,1,1,1,0,1,1,0,1,0,1,0,0,1,1,1,0,1,0,1,1,1,0,1,1,1,0,1,0,0,1,1,0,0,1,1,0,0,1,1,0,1,1,1,0,1,1,0,1,1,0,0,1,1,0,1,1,1,0,1,0,0,1,1,1,0,1,0,1,1,1,1,1,1,0,1,0,1,1,1,1,1,0,0,1,1,1,1,1,1,1,1,0,1,1,1,0,1,1,1,0,1,1,1,0,1,1,0,1,1,1,1,1,1,0,1,1,0,1,1,0,0,0,1,0,0,1,1,0,0,1,0,1,1,0,1,1,1,0,1,1,1,0,1,1,0,0,1,1,0,0,1,0,1,1,1,1,1,1,1,0,1,1,1,1,1,0,0,1,1,0,0,1,1,0,1,1,1,0,1,0,0,1,1,0,1,1,1,0,1,1,1,0,1,1,0,1,1,0,0,1,0,0,1,1,0,1,1,0,0,0,1,0,1,1,0,1,0,1,1,0,1,1,0,0,1,0,1,1,1,1,0,1,0,0,1,1,0,0,1,0,0,1,1,1,1,1,1,1,1,1,0,0,1,1,1,0,1,1,1,1,1,1,0,1,0,0,1,0,0,1,1,1,0,1,1,1,1,1,1,0,1,1,1,0,1,0,0,1,1,1,1,1,0,0,1,1,0,0,1,1,1,0,1,0,0,1,1,0,0,1,0,0,1,0,1,1,1,0,0,1,0,1,1,1,0,0,1,0,0,0,1,0,0,1,1,1,0,1,1,0,1,1,0,0,1,1,1,1,1,0,1,1,0,1,0,1,1,0,1,0,0,0,1,0,0,1,1,0,1,1,1,1,1,0,1,1,1,0,1,1,1,1,1,1,1,1,1,0,0,1,1,1,0,1,1,0,1,1,1,0,1,1,1,0,1,0,1,1,1,0,0,1,0,0,0,1,0,0,1,0,0,0,1,1,1,1,0,1,1,1,1,0,1,0,0,1,1,1,1,1,0,1,1,1,0,0,1,1,1,0,1,0,0,1,1,1,1,1,0,1,1,1,0,1,1,1,1,0,1,0,1,1,0,1,0,1,1,0,1,0,1,1,1,0,0,1,1,0,1,1,0,0,1,0,1,1,1,0,1,1,1,0,0,1,0,1,1,1,0,0,1,0,0,1,1,0,1,1,1,0,1,1,0,1,1,1,0,1,1,0,1,1,1,1,1,1,0,1,1,0,0,1,1,0,1,1,1,1,1,1,1,1,1,1,0,1,1,0,1,1,0,0,1,1,0,0,1,1,1,1,1,0,0,1,1,1,1,1,0,1,1,1,1,0,1,1,1,0,1,1,0,1,0,1,0,1,1,0,1,1,1,1,1,0,0,1,0,0,1,1,1,0,1,0,1,1,1,0,0,1,1,0,0,1,1,0,1,1,1,0,1,1,1,1,1,1,1,1,1,0,1,1,0,0,1,1,0,1,1,0,0,1,1,0,0,1,0,1,1,1,0,0,1,0,1,1,0,1,0,1,1,1,1,0,1,1,1,0,1,1,1,0,0,1,1,1,1,1,0,1,1,1,0,1,1,1,1,1,0,1,1,1,0,1,1,0,1,0,1,0,1,1,1,0,0,1,0,0,1,0,0,1,1,0,1,1,1,1,0,1,0,1,1,0,0,0,1,1,1,1,1,1,0,1,1,1,1,1,1,1,1,0,0,1,1,1,1,1,1,1,1,1,0,1,1,1,0,1,1,0,1,1,1,0,0,1,0,0,1,0,0,0,1,1,0,1,1,1,0,1,1,0,1,1,0,0,1,0,0,1,1,0,1,1,0,0,1,1,1,0,1,0,0,1,1,1,1,1,0,0,1,1,1,0,1,0,0,1,1,1,1,1,0,1,1,1,1,1,1,0,0,0,1,0,0,1,1,1,1,1,0,0,1,0,0,1,1,1,1,1,1,1,1,1,0,1,1,1,1,0,1,0,1,1,1,1,1,1,0,0,0,1,0,0,1,1,1,0,1,0,0,1,0,0,0,1,0,1,1,1,0,1,1,0,0,1,0,0,1,1,0,1,1,1,0,1,1,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,1,1,0,1,1,1,0,0,1,0,1,0,1,0,1,1,1,1,0,1,1,0,1,1,0,0,1,0,1,1,1,0,1,1,1,0,1,1,0,0,1,0,0,1,1,0,0,1,0,1,0,1,1,1,1,0,1,1,1,1,1,1,0,0,0,1,1,0,1,0,1,0,1,0,0,1,0,1,0,1,0,0,1,1,1,1,1,1,1,1,0,0,0,1,1,0,1,1,1,1,1,1,1,1,1,1,0,1,1,1,1,1,0,1,1,0,1,0,1,1,0,1,1,1,0,1,1,0,1,0,0,0,1,0,0,1,1,0,1,1,1,0,1,0,1,0,1,1,1,1,1,1,0,1,1,1,1,1,0,0,1,0,1,1,1,1,0,1,1,1,0,1,0,0,1,0,1,1,1,0,1,1,0,1,0,1,1,1,1,0,0,1,1,1,1,1,0,0,1,1,1,1,1,1,1,1,1,0,1,1,1,0,1,1,1,0,1,1,1,0,1,0,1,1,1,1,0,1,1,1,1,1,1,1,1,1,0,1,1,1,1,1,1,0,1,1,1,1,1,0,1,0,1,0,0,1,1,0,0,1,1,0,1,0,0,1,1,0,0,1,1,0,1,1,0,0,1,1,1,0,1,1,1,1,1,1,1,1,0,1,1,1,1,0,1,0,1,0,1,1,1,1,1,0,1,1,1,0,1,1,1,1,1,0,0,1,0,1,1,1,1,1,1,0,0,0,1,0,0,1,1,0,0,1,0,0,1,1,0,1,1,0,0,1,1,0,1,1,1,0,0,1,0,0,1,1,1,1,1,1,0,1,0,1,0,1,0,1,1,0,0,1,1,0,1,1,0,0,1,1,0,0,1,1,0,0,1,0,0,1,1,0,1,1,1,0,1,1,1,0,1,0,0,1,1,0,1,1,0,0,1,1,0,1,1,0,0,1,1,1,0,1,0,0,0,1,1,0,1,1,1,1,1,0,1,1,0,0,1,1,0,1,1,1,0,1,1,0,1,1,1,1,0,1,1,0,1,1,0,1,1,1,1,1,1,1,1,1,1,0,0,1,0,0,1,0,0,1,1,1,1,1,0,1,0,1,0,1,1,0,1,0,1,1,1,1,1,0,0,1,0,1,1,1,0,0,1,1,1,1,1,1,1,1,0,0,0,1,0,0,1,0,0,1,1,0,1,1,0,0,0,1,1,1,1,1,1,1,1,0,1,1,0,0,1,1,1,0,1,1,0,0,1,1,0,1,1,0,1,1,1,1,1,1,0,1,1,0,0,1,1,1,1,1,0,1,0,1,1,0,1,1,1,1,1,1,0,1,0,1,0,1,1,1,1,1,1,1,1,1,1,1,0,1,1,1,1,1,1,0,0,1,1,0,1,1,1,1,0,1,1,0,1,1,1,1,1,1,1,1,1,0,1,1,1,0,1,1,1,1,1,1,1,0,1,0,1,1,1,1,0,1,0,1,1,0,0,0,1,1,0,1,1,1,0,1,0,0,1,1,1,1,0};
        for (int i = 1; i < this->len-1; i++) {
            temp[i] = 1;
        }
        ind_test->chrom = temp;
        double iter_time = 0.0;
        double fit_sum = 0;
        double forward_time = 0.0;
        double backward_time = 0.0;
        double sync_time = 0.0;
        double sync_time_f = 0.0;
        double sync_time_b = 0.0;
        double offload_size_error = 0;
        double size_fit;
        double offload_time;
        double fetch_time;
        bool flag;
        flag = optim->iter_time_by_swap_layers( temp, &offload_size_error, &iter_time, &forward_time, &backward_time, &sync_time_f, &sync_time_b, &offload_time, &fetch_time);
        exit(1);
    }

};

}

#endif //ATP_GA_H