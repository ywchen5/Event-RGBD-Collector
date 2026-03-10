project(scanner_cpp_optim)
cmake_minimum_required(VERSION 3.5)

# set standard
set(CMAKE_CXX_STANDARD 17)

# find OrbbecSDK
set(OrbbecSDK_DIR "/home/hmy/VISUALIZE_EB_FB_DEVICE/cpp_optim/thirdparty/OrbbecSDK/")
find_package(OrbbecSDK REQUIRED)

# find MetavisionSDK
find_package(MetavisionSDK COMPONENTS core driver ui REQUIRED)

# find threading library
find_package(Threads REQUIRED)

# HDF5 (for DataRecorder) – use system serial HDF5 to match MetavisionSDK's libhdf5_cpp.so.103
# Avoid find_package(HDF5) which may pick up anaconda's incompatible version.
set(HDF5_INCLUDE_DIR "/usr/include/hdf5/serial")
set(HDF5_CXX_LIB "/usr/lib/x86_64-linux-gnu/hdf5/serial/libhdf5_cpp.so")
set(HDF5_C_LIB   "/usr/lib/x86_64-linux-gnu/hdf5/serial/libhdf5.so")

# include project headers
include_directories(${CMAKE_SOURCE_DIR}/include)

# find OpenCV (needed by MetavisionSDK frame generation)
# Only request the modules we actually use to avoid pulling in heavy transitive
# dependencies (opencv_viz → VTK/wayland/ffi, opencv_hdf → anaconda HDF5, etc.)
find_package(OpenCV REQUIRED COMPONENTS core imgproc imgcodecs highgui)

# link OrbbecSDK
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
    OrbbecSDK::OrbbecSDK
    ${OpenCV_LIBS}
    ${HDF5_CXX_LIB}
    ${HDF5_C_LIB}
    Threads::Threads
)    