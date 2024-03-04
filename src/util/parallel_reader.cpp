//
// Created by ay27 on 7/24/17.
//

#include <util/parallel_reader.h>

namespace ATP {


template<class value_type>
void parallel_reader_t<value_type>::get_batch(tensor_t<value_type> *_data, tensor_t<value_type> *_label) {
    if (data != NULL && label != NULL) {
        // we must free the tensor first, it's important when the gpu cache size is ONE !!!
        bool res = q->free_gpu_tensor(data, label);
        if (!res) {
            fprintf(stderr, "can not free tensor %p and %p!!!\n", data, label);
        }
    }
    // cout << "label ptr = " << (void*)_label << ", output ptr = " << (void*)_data << endl;
    // cout << "label device = " << (int)(_label->get_data_position()) << ", output device = " << (int)(_data->get_data_position()) << endl;

    // 将数据的地址放到 data和lable上面 如果没成功 也就是没找到空闲块来存放这两个数据
    if (!q->fetch_gpu_tensor(&data, &label)) {
		// printf("!q->fetch_gpu_tensor(&data, &label)\n");
        wait_data_m.unlock();
        wait_data_m.lock();
		// printf("wait_data_m.unlock(); wait_data_m.lock(); done\n");
        // will wait inner thread to unlock wait_data_m
        while (!wait_data_m.try_lock()) {
			// printf("!wait_data_m.try_lock()\n");
            sleep_a_while(5);
        }
        if (!q->fetch_gpu_tensor(&data, &label)) {
            fprintf(stderr, "can not fetch data !!!!\n");
			// printf("can not fetch data !!!!\n");
            exit(-1);
        }
    }
    // printf("=====================================================================================================2");
    // we don't use the gpu space alloced in tensor 如果是第一次获取batch 则释放掉原来的GPU空间 这个地方
    if (is_first_time_to_get_batch) {
        // _data->free_gpu_space();
        //  printf("=====================================================================================================4, %d\n", _label);
        // cout << "data real ptr = " << (void*)_data << " data real device ptr = " << (int)(_data->get_data_position()) << endl;
        // cout << "_label real ptr = " << (void*)_label << " _label real device ptr = " << (int)(_label->get_data_position()) << endl;
        // _label->free_gpu_space();
        // cout << "_label real ptr = " << (void*)_label << " _label real device ptr = " << (int)(_label->get_data_position()) << endl;
        //  printf("=====================================================================================================5\n");
        is_first_time_to_get_batch = false;
    }
    // 使用 tensor_queue中的GPU的空间
    // at now, we use the gpu memory alloced in tensor_queue !!!!
    // just replace the gpu ptr to avoid memcpy
    _data->replace_gpu_ptr_without_free(data->get_gpu_ptr());
    _label->replace_gpu_ptr_without_free(label->get_gpu_ptr());
}

template<class value_type>
void parallel_reader_t<value_type>::thread_entry(size_t thread_idx, size_t total_threads) {
	// cudaSetDevice(1);
    image_reader_t<value_type> reader(this->data_path, this->label_path, this->N, this->in_memory, this->shuffle);
    reader.seek(reader.get_total_data_cnt() / total_threads * thread_idx);

    tensor_t<value_type> *data_tmp = NULL, *label_tmp = NULL;
    value_type *raw = NULL;
    if (processor != NULL) {
        raw = (value_type *) malloc(N * srcC * srcH * srcW * sizeof(value_type));
    }
    while (!should_stop()) {
        if (q->fetch_free_tensor(&data_tmp, &label_tmp)) {
            // fetch free tensor success
            if (processor != NULL) {
                reader.get_batch(raw, label_tmp->get_cpu_ptr());
            } else {
                reader.get_batch(data_tmp->get_cpu_ptr(), label_tmp->get_cpu_ptr());
            }

            if (processor != NULL) {
                pre_m.lock();
                processor->process(raw, data_tmp->get_cpu_ptr());
                pre_m.unlock();
            }

            // try transfer it to gpu if has free gpu space
            if (q->transfer_cpu_to_gpu()) {
                wait_data_m.unlock();
            }

        } else {
            // we try to transfer some tensor to gpu when the thread is free
            if (q->transfer_cpu_to_gpu()) {
                wait_data_m.unlock();
            }
            sleep_a_while();
        }
    }
    if (raw != NULL) {
        free(raw);
    }
}

INSTANTIATE_CLASS(parallel_reader_t);

} // namespace ATP