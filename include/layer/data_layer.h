//
// Created by ay27 on 17/3/31.
//

#ifndef ATP_DATA_LAYER_H
#define ATP_DATA_LAYER_H

#include <tensor.h>
#include <registry.h>
#include <util/parallel_reader.h>
#include <layer/base_network_layer.h>
#include <mem_control.h>

namespace ATP {

    template<class value_type>
    class data_layer_t : public base_network_layer_t<value_type> {
    private:
		size_t seqLength, miniBatch, inputSize, labelSize;
		size_t beamDim, embedSize;
		bool is_seq = false;
		bool is_rnn_model = false;
        size_t N, C, H, W;
        const data_mode mode;
        parallel_reader_t<value_type> *reader;
		
		value_type init_data = 0.314;
		
#ifdef FAKE_TRAIN
		// tensor_t<value_type>* fake_data;
		// tensor_t<value_type>* fake_label;
#endif		
        // value_type *data_fake[2]={NULL, NULL}, *label_fake[2]={NULL, NULL};
		value_type *fake_data;
		value_type *fake_label;
		int *fake_label_int;
		int *fake_label_int_cpu;
        int choose_fake = 0;

		mem_controller_t<value_type>* mem_controller; 

        void get_batch_fake(tensor_t<value_type> *_data, tensor_t<value_type> *_label);
	
		void get_batch_fake_int(tensor_t<value_type> *_data, tensor_t<value_type> *_label);

    public:
        data_layer_t(data_mode m,
                     parallel_reader_t<value_type> *_reader) :
                reader(_reader), mode(m), base_network_layer_t<value_type>(DATA_L) {
            this->N = reader->getN();
            this->C = reader->getC();
            this->H = reader->getH();
            this->W = reader->getW();
			printf("reader: NCHW = %d %d %d %d\n", this->N, this->C, this->H, this->W);
        }
		
		// data layer for rnn test
		data_layer_t(size_t seqLength, size_t miniBatch, size_t inputSize, size_t labelSize, 
					data_mode mode = DATA_TRAIN):
			seqLength(seqLength), 
			miniBatch(miniBatch), 
			inputSize(inputSize), 
			labelSize(labelSize),
			mode(mode), base_network_layer_t<value_type>(DATA_L)
		{
			
			is_rnn_model = true;
			// is_rnn_model = false;
			this->N = seqLength;
            this->C = miniBatch;
            this->H = inputSize;
            this->W = 1;	
		}

		// data layer for seq2seq data
		// bool is_seq is set to distinguish this data_layer_t() from the data_layer_t() of rnn
		data_layer_t(size_t seqLength, size_t miniBatch, size_t beamDim, size_t embedSize, bool is_seq, 
					data_mode mode = DATA_TRAIN):
			seqLength(seqLength), 
			miniBatch(miniBatch), 
			beamDim(beamDim), 
			embedSize(embedSize),
			is_seq(is_seq),
			mode(mode), base_network_layer_t<value_type>(DATA_L)
		{
			this->N = seqLength;
            this->C = miniBatch;
            this->H = beamDim;
            this->W = embedSize;
		}
		
		void ResetBatchSize(size_t batch_size, bool is_seq) {
			if (is_seq) {
				this->miniBatch = this->C = batch_size;
			}
			else {
				this->N = batch_size;
			}
		}

		void reset_data_nchw(size_t N, size_t C, size_t H, size_t W, data_mode mode = DATA_TRAIN) {
			this->N = N;
            this->C = C;
            this->H = H;
            this->W = W;
		}

		void reset_data_seq2seq(size_t seqLength, size_t miniBatch, size_t beamDim, size_t embedSize, bool is_seq, 
					data_mode mode = DATA_TRAIN) {
			this->is_seq = is_seq;
			// is_rnn_model = false;
			this->seqLength = this->N = seqLength;
            this->miniBatch = this->C = miniBatch;
            this->inputSize = this->H = beamDim;
            this->embedSize = this->W = embedSize;
		}
		
		void reset_data_rnn(size_t seqLength, size_t miniBatch, size_t inputSize, size_t labelSize, 
					data_mode mode = DATA_TRAIN) {
			is_rnn_model = true;
			// is_rnn_model = false;
			this->seqLength = this->N = seqLength;
            this->miniBatch = this->C = miniBatch;
            this->inputSize = this->H = inputSize;
            this->W = 1;
			this->labelSize = labelSize;
		}		

        ~data_layer_t() {
            delete reader;
        }
		
		value_type* get_fake_data() {
			return fake_data;
		}
		value_type* get_fake_label() {
			return fake_label;
		}
		int* get_fake_label_int() {
			return fake_label_int;
		}
		bool get_is_rnn_model() {
			return this->is_rnn_model;
		}

		void data_fake_init_pool_malloc_mode(void* gpu_pool_ptr, size_t* offset);

		void data_fake_init_for_simulator();

		void data_fake_init();

		size_t get_fake_data_label_size();

        size_t get_batch_size() {
            return this->N;
        }



		void reset_reader(parallel_reader_t<value_type> * new_reader) {
			this->reader = new_reader;
			this->N = reader->getN();
            this->C = reader->getC();
            this->H = reader->getH();
            this->W = reader->getW();
		}

        void forward_setup(registry_t<value_type> *reg, cudnnHandle_t *cudnn_h);

        void backward_setup(registry_t<value_type> *reg, cudnnHandle_t *cudnn_h);

        std::vector<value_type> forward(network_stage stage, cublasHandle_t *cublas_h, cudnnHandle_t *cudnn_h, registry_t<value_type> *reg);

        void backward(network_stage stage, cublasHandle_t *cublas_h, cudnnHandle_t *cudnn_h, registry_t<value_type> *reg);

        void gen_description(char* buff, size_t* len_in_byte) {
            this->gen_meta_description(buff, len_in_byte);
        }

		void fake_run(net_comp dir, registry_t<value_type> *reg);
    };

	
}

#endif //ATP_DATA_LAYER_H
