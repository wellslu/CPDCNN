# set the code name for executable file 
set(L1 test1)
set(TESTUTIL util)
set(TESTSYRK syrk)
set(CMAKE_OUTPUT_TEST_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/test)
# add test resource here
add_executable(${L1}
    # add needed lib or source code here
    test1.cpp
)

add_executable(${TESTUTIL}
    ../lib/util.cu
    ../lib/cublas-norm/syrk.cu
    test-util.cu
)

add_executable(${TESTSYRK}
    ../lib/cublas-norm/syrk.cu
    test-syrk.cu
)

# set output directory
set_target_properties(${L1} PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_OUTPUT_TEST_DIRECTORY})
set_target_properties(${TESTUTIL} PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_OUTPUT_TEST_DIRECTORY})
set_target_properties(${TESTSYRK} PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_OUTPUT_TEST_DIRECTORY})

# Link Google Test obj file
target_link_libraries(${L1} gtest gtest_main)
target_link_libraries(${TESTUTIL} PRIVATE gtest gtest_main cudart /usr/local/cuda-11.7/lib64/libcublas.so torch)
target_link_libraries(${TESTSYRK} PRIVATE gtest gtest_main cudart cublas)

# add test to activate ctest -VV
add_test(NAME ${L1} COMMAND ${CMAKE_OUTPUT_TEST_DIRECTORY}/${L1})
add_test(NAME ${TESTUTIL} COMMAND ${CMAKE_OUTPUT_TEST_DIRECTORY}/${TESTUTIL})
add_test(NAME ${TESTSYRK} COMMAND ${CMAKE_OUTPUT_TEST_DIRECTORY}/${TESTSYRK})
