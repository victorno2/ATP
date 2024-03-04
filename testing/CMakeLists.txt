
#CUDA_ADD_EXECUTABLE(alexnet alexnet.cpp)
#target_link_libraries(alexnet ATP)

#CUDA_ADD_EXECUTABLE(lenet lenet.cpp)
#target_link_libraries(lenet ATP)

#CUDA_ADD_EXECUTABLE(datareader_test datareader_test.cpp
#        ../include/util/image_reader.h
#        ../include/util/base_reader.h
#        ../include/util/parallel_reader.h
#        ../include/util/binary_dumper.h
#        ../include/util/thread_routine.h
#        ../include/util/tensor_queue.h
#        ../include/util/mem_util.h
#        ../include/liveness.h
#        ../include/recompute.h
#        ../include/mem_control.h
#        ../include/swap.h
#        ../include/swap_v2.h
#        ../include/network.h
#        ../include/bandwidth.h
#        )
#target_link_libraries(datareader_test ATP)

#CUDA_ADD_EXECUTABLE(saver_test
#        ../include/util/saver.h
#        ../include/solver.h
#        ../src/solver.cpp
#        ../include/layer/batch_normalization_layer.h
#        ../include/layer/concat_layer.h
#        ../include/layer/cudnn_activation_layer.h
#        ../include/layer/cudnn_convolution_layer.h
#        ../include/layer/cudnn_pooling_layer.h
#        ../include/layer/data_layer.h
#        ../include/layer/dropout_layer.h
#        ../include/layer/fork_layer.h
#        ../include/layer/fully_connected_layer.h
#        ../include/layer/join_layer.h
#        ../include/layer/local_response_norm_layer.h
#        ../include/layer/padding_layer.h
#        ../include/layer/softmax_layer.h
#        ../include/layer/base_layer.h
#        )
#target_link_libraries(saver_test ATP)

CUDA_ADD_EXECUTABLE(resnet512-v100-32g resnet512-v100-32g.cpp)
target_link_libraries(resnet512-v100-32g ATP)

CUDA_ADD_EXECUTABLE(resnet1022-v100-32g resnet1022-v100-32g.cpp)
target_link_libraries(resnet1022-v100-32g ATP)

CUDA_ADD_EXECUTABLE(gpt1-v100-32g gpt1-v100-32g.cpp)
target_link_libraries(gpt1-v100-32g ATP)

CUDA_ADD_EXECUTABLE(rnn64-v100-32g rnn64-v100-32g.cpp)
target_link_libraries(rnn64-v100-32g ATP)

CUDA_ADD_EXECUTABLE(gru64-v100-32g gru64-v100-32g.cpp)
target_link_libraries(gru64-v100-32g ATP)

CUDA_ADD_EXECUTABLE(lstm64-v100-32g lstm64-v100-32g.cpp)
target_link_libraries(lstm64-v100-32g ATP)

add_subdirectory(unittest)