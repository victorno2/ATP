#
# Created by ay27 on 17/3/7.
#

# link a shared library
CUDA_ADD_LIBRARY(ATP SHARED ${SOURCE_FILES} ${CUDA_FILES})
target_link_libraries(ATP ${THIRD_LIBS} -lpthread -lm)
#target_link_libraries(ATP ${THIRD_LIBS} -lm)
#target_link_libraries(ATP ${CUDA_LIBRARIES} ${CUDA_CUBLAS_LIBRARIES} ${CUDNN_LIBRARIES} ${GLOG_LIBRARIES})
