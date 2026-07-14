set(CMAKE_CXX_STANDARD 11)

find_library(TCMALLOC_LIBRARY tcmalloc)

add_executable(test_nsg_index test_nsg_index.cpp)
target_link_libraries(test_nsg_index ${PROJECT_NAME})
if(TCMALLOC_LIBRARY)
    target_link_libraries(test_nsg_index ${TCMALLOC_LIBRARY})
endif()

add_executable(test_gate_search test_gate_search.cpp)
target_link_libraries(test_gate_search ${PROJECT_NAME})

add_executable(test_gate_optimized_search test_gate_optimized_search.cpp)
target_link_libraries(test_gate_optimized_search ${PROJECT_NAME})
if(TCMALLOC_LIBRARY)
    target_link_libraries(test_gate_optimized_search ${TCMALLOC_LIBRARY})
endif()

# The Zenodo artifact's test_nsg_search.cpp has undefined training-helper
# variables in this snapshot. Keep the paper search path buildable and track
# the training-helper limitation in docs/close_baselines.md.

add_executable(test_gate_cos_navigate test_gate_cos_navigate.cpp)
target_link_libraries(test_gate_cos_navigate ${PROJECT_NAME})
if(TCMALLOC_LIBRARY)
    target_link_libraries(test_gate_cos_navigate ${TCMALLOC_LIBRARY})
endif()
