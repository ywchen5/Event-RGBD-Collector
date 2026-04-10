cmake_minimum_required(VERSION 3.5)
project(scanner_cpp_optim)

# set standard
set(CMAKE_CXX_STANDARD 17)

# find OrbbecSDK
if(NOT DEFINED OrbbecSDK_DIR)
    set(OrbbecSDK_DIR "${CMAKE_SOURCE_DIR}/thirdparty/OrbbecSDK/lib")
endif()
find_package(OrbbecSDK REQUIRED)

# find MetavisionSDK
if(DEFINED MetavisionSDK_DIR)
    list(APPEND CMAKE_PREFIX_PATH "${MetavisionSDK_DIR}")
endif()
find_package(MetavisionSDK COMPONENTS core driver ui REQUIRED)

# find threading library
find_package(Threads REQUIRED)

# HDF5
if(WIN32)
    # Prefer Prophesee-bundled HDF5 to keep compile-time and runtime versions aligned.
    set(PROPHESEE_THIRDPARTY_DIR "C:/Program Files/Prophesee/third_party")
    set(PROPHESEE_HDF5_INCLUDE "${PROPHESEE_THIRDPARTY_DIR}/include")
    set(PROPHESEE_HDF5_CXX_LIB "${PROPHESEE_THIRDPARTY_DIR}/lib/hdf5_cpp.lib")
    set(PROPHESEE_HDF5_C_LIB   "${PROPHESEE_THIRDPARTY_DIR}/lib/hdf5.lib")

    if(EXISTS "${PROPHESEE_HDF5_INCLUDE}/hdf5.h"
       AND EXISTS "${PROPHESEE_HDF5_CXX_LIB}"
       AND EXISTS "${PROPHESEE_HDF5_C_LIB}")
        set(HDF5_INCLUDE_DIR "${PROPHESEE_HDF5_INCLUDE}")
        set(HDF5_CXX_LIB "${PROPHESEE_HDF5_CXX_LIB}")
        set(HDF5_C_LIB   "${PROPHESEE_HDF5_C_LIB}")
        message(STATUS "Using Prophesee bundled HDF5: ${PROPHESEE_THIRDPARTY_DIR}")
    else()
        # Fallback if Prophesee path is unavailable.
        find_package(HDF5 COMPONENTS C CXX REQUIRED)
        set(HDF5_INCLUDE_DIR ${HDF5_INCLUDE_DIRS})
        set(HDF5_CXX_LIB ${HDF5_CXX_LIBRARIES})
        set(HDF5_C_LIB   ${HDF5_C_LIBRARIES})
    endif()
else()
    # Linux defaults (compatible with the original Ubuntu setup).
    if(NOT DEFINED HDF5_INCLUDE_DIR)
        set(HDF5_INCLUDE_DIR "/usr/include/hdf5/serial")
    endif()
    if(NOT DEFINED HDF5_CXX_LIB)
        set(HDF5_CXX_LIB "/usr/lib/x86_64-linux-gnu/hdf5/serial/libhdf5_cpp.so")
    endif()
    if(NOT DEFINED HDF5_C_LIB)
        set(HDF5_C_LIB   "/usr/lib/x86_64-linux-gnu/hdf5/serial/libhdf5.so")
    endif()
endif()

# include project headers
include_directories(${CMAKE_SOURCE_DIR}/include)

# find OpenCV (needed by MetavisionSDK frame generation)
find_package(OpenCV REQUIRED COMPONENTS core imgproc imgcodecs highgui)

add_executable(${PROJECT_NAME}
    src/main.cpp
    src/orbbec_processor.cpp
    src/prophesee_processor.cpp
    src/sync_processor.cpp
    src/data_recorder.cpp
)

target_include_directories(${PROJECT_NAME} PRIVATE ${HDF5_INCLUDE_DIR})
target_link_libraries(${PROJECT_NAME}
    MetavisionSDK::core MetavisionSDK::driver MetavisionSDK::ui
    ob::OrbbecSDK
    ${OpenCV_LIBS}
    ${HDF5_CXX_LIB}
    ${HDF5_C_LIB}
    Threads::Threads
)
