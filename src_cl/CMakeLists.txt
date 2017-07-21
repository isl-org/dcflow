find_package(OpenCL REQUIRED)


set(CL_FILE dcflow)

include_directories(${MATLAB_INCLUDE_DIR} ${OpenCL_INCLUDE_DIRS})

add_library(${CL_FILE} SHARED
      dcflow_mex.cpp
      dcflow.cpp
      sgm.cpp
      ${CMAKE_SOURCE_DIR}/matlab/Matlabdef.def)
target_link_libraries(${CL_FILE} ${MATLAB_LIBRARIES} ${OpenCL_LIBRARIES} dl)

set_target_properties(${CL_FILE} PROPERTIES PREFIX "" LINKER_LANGUAGE CXX)
set_target_properties(${CL_FILE} PROPERTIES OUTPUT_NAME "dcflow_mex")

set_property(TARGET ${CL_FILE} PROPERTY CXX_STANDARD 11)
set_property(TARGET ${CL_FILE} PROPERTY CXX_STANDARD_REQUIRED ON)


# 32-bit or 64-bit mex
if(WIN32)
    if (CMAKE_CL_64)
      set_target_properties(${CL_FILE} PROPERTIES SUFFIX .mexw64)
    else(CMAKE_CL_64)
      set_target_properties(${CL_FILE} PROPERTIES SUFFIX .mexw32)
    endif(CMAKE_CL_64)
else(WIN32)
    if (CMAKE_SIZEOF_VOID_P MATCHES "8")
      set_target_properties(${CL_FILE} PROPERTIES SUFFIX .mexa64 PREFIX "")
    else(CMAKE_SIZEOF_VOID_P MATCHES "8")
      set_target_properties(${CL_FILE} PROPERTIES SUFFIX .mexglx PREFIX "")
    endif (CMAKE_SIZEOF_VOID_P MATCHES "8")
endif(WIN32)

install(TARGETS ${CL_FILE} DESTINATION matlab)
