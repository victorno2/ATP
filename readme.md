### Requirements
cuda10.2
cudnn7.65
glog-0.4.0
cmake_minimum_required(VERSION 3.16)
Modify "/config.linux" lines 1~3 for the paths of libraries

### Make
mkdir bulid
cd build
cmake ..
make -j8

### Run the tests: Firstly, modify the file "/include/util/common.h", lines 31~33.

Run normal training, uncomment "#define BASELINE_TRAINING", and comment the other lines. Make.
Command: ./build/testing/<file name> <device id> <batch size>

Run The optimal training config of ATP, uncomment "#define TRAINING_CONFIGURATION_SEARCH", and comment the other lines. Make.
Command: ./build/testing/<file name> <device id> <starting batch size> <batch size step> <ending batch size> <test times> <search times> <GA population size>
Wait for the search to end, and then copy the global best solution code the corresponding file. For example, if you get a solution code for rnn with batch size 74, copy it to "/testing/rnn64-v100-32g.cpp" the lines between "#ifdef ATP_SOLUTION" and "#endif", such as:
    /* ATP: batchsize = 512 * 74 * 128 * 1 */
    int rs_code[code_size] = {0,1,1,1,0,0,2,0,1,2,0,1,2,1,0,2,0,1,2,1,1,2,1,0,2,0,0,2,0,0,2,1,0,2,0,0,2,1,1,2,0,1,2,1,1,2,0,0,0,1,1,2,0,1,2,0,0,2,1,1,2,0,1,2,0,1,2,1,1,2,0,1,0,0,0,2,1,1,2,1,0,2,0,1,2,0,1,2,0,1,2,1,1,2,1,0,2,0,1,2,0,1,2,0,0,2,1,0,2,1,1,2,1,0,2,1,0,2,0,1,2,0,0,2,1,0,2,0,0,2,1,0,2,1,0,0,0,0,2,1,1,2,0,1,2,1,0,2,0,0,2,0,0,2,1,0,2,0,0,2,1,0,2,0,0,2,1,0,2,1,0,2,0,1,2,0,0,2,1,0,2,0,1,2,1,0,2,0,0,2,0,0,2,0,0,1,1,1,0};
    /*********************************************/
Save and Make.

When you're done searching for the optimal configuration, uncomment "#define RECOPUTING_SWAPPING_TRAINING", and comment the other lines. Make. Then you can test the performance of ATP.
Command: ./build/testing/<file name> <device id> <batch size>



