# set the code name for executable file 
set(L1 test1)

# add test resource here
add_executable(${L1}
    # add needed lib or source code here
    test1.cpp
)

# Link Google Test obj file
target_link_libraries(${L1} gtest gtest_main)

# add test to activate ctest -VV
add_test(NAME ${L1} COMMAND ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${L1})

