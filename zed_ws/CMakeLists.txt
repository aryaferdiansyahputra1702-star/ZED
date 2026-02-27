# 设置项目
cmake_minimum_required(VERSION 3.12)
cmake_policy(SET CMP0091 NEW)

project(zed LANGUAGES CXX CUDA)

# ----------------------------------------------------------------------------
# Set C++ standard
# ----------------------------------------------------------------------------
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# ----------------------------------------------------------------------------
# Environment Variable Check
# ----------------------------------------------------------------------------
if(NOT DEFINED ENV{OFFLINE_ENV})
  message(FATAL_ERROR "OFFLINE_ENV is not set.")
else()
  set(OFFLINE_ENV $ENV{OFFLINE_ENV})
  message("OFFLINE_ENV is set to ${OFFLINE_ENV}")
endif()

set(CMAKE_INSTALL_PREFIX ${OFFLINE_ENV})

# ----------------------------------------------------------------------------
# Build Type and Optimization Flags
# ----------------------------------------------------------------------------
set(CMAKE_BUILD_TYPE Release)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Ofast -march=native -mtune=native -finline-functions -flto")

# ----------------------------------------------------------------------------
# CUDA Configuration
# ----------------------------------------------------------------------------
find_package(CUDAToolkit REQUIRED)
set(CUDA_INCLUDE_DIRS ${CUDAToolkit_INCLUDE_DIRS})
set(CUDA_LIBRARIES ${CUDAToolkit_LIBRARIES})
set(CMAKE_CUDA_ARCHITECTURES 89;86;75;70;61)

# ----------------------------------------------------------------------------
# Set TensorRT path
# ----------------------------------------------------------------------------
set(TENSORRT_PATH "/usr/local/tensorrt_real")


# ----------------------------------------------------------------------------
# Dependencies
# ----------------------------------------------------------------------------
find_package(Boost  REQUIRED COMPONENTS
             thread program_options)

find_package(ZED 3 REQUIRED)
find_package(OpenCV REQUIRED)
option(LINK_SHARED_ZED "Link with the ZED SDK shared executable" ON)

if (NOT LINK_SHARED_ZED AND MSVC)
    message(FATAL_ERROR "LINK_SHARED_ZED OFF : ZED SDK static libraries not available on Windows")
endif()

# ----------------------------------------------------------------------------
# Include Directories
# ----------------------------------------------------------------------------
include_directories(${OFFLINE_ENV}/include)
include_directories(${OFFLINE_ENV}/include/World
${OFFLINE_ENV}/include/tambahan)
include_directories(${CUDA_INCLUDE_DIRS})
include_directories(${ZED_INCLUDE_DIRS})
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(include)

# ----------------------------------------------------------------------------
# Function: Configure CUDA & TensorRT
# ----------------------------------------------------------------------------
function(configure_cuda_trt target)
    target_compile_definitions(${target} PRIVATE ${CUDA_DEFINITIONS})
    target_include_directories(${target} PRIVATE ${CUDA_INCLUDE_DIRS} ${TENSORRT_PATH}/include)
    target_link_directories(${target} PRIVATE ${TENSORRT_PATH}/lib)
    target_link_libraries(${target} PRIVATE ${CUDA_LIBRARIES} nvinfer nvinfer_plugin nvonnxparser)
endfunction()

# ----------------------------------------------------------------------------
# Function: Add Source Files
# ----------------------------------------------------------------------------
function(add_compile_files target)
    include_directories(${PROJECT_SOURCE_DIR})
    file(GLOB_RECURSE SOURCES
        ${PROJECT_SOURCE_DIR}/deploy/core/*.cpp
        ${PROJECT_SOURCE_DIR}/deploy/utils/*.cpp
        ${PROJECT_SOURCE_DIR}/deploy/infer/*.cpp
        ${PROJECT_SOURCE_DIR}/deploy/infer/*.cu
        ${PROJECT_SOURCE_DIR}/deploy/model.cpp
    )
    target_sources(${target} PRIVATE ${SOURCES})
endfunction()

# ----------------------------------------------------------------------------
# Function: Set Compilation Options
# ----------------------------------------------------------------------------
function(set_compile_options target)
    if(MSVC)
        target_compile_options(${target} PRIVATE $<$<CONFIG:Release>:-O2>)
        set_property(TARGET ${target} PROPERTY MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
    else()
        target_compile_options(${target} PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-O3 -flto=auto>)
        target_link_options(${target} PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-O3 -flto=auto>)
    endif()
endfunction()

# ----------------------------------------------------------------------------
# Define Deploy Library
# ----------------------------------------------------------------------------
add_library(ZedDeploy SHARED)
add_compile_files(ZedDeploy)
configure_cuda_trt(ZedDeploy)
set_compile_options(ZedDeploy)
set_target_properties(ZedDeploy PROPERTIES OUTPUT_NAME ZedDeploy)

# ----------------------------------------------------------------------------
# Installation for Libraries
# ----------------------------------------------------------------------------
install(TARGETS ZedDeploy
  ARCHIVE DESTINATION ${CMAKE_INSTALL_PREFIX}/lib
  LIBRARY DESTINATION ${CMAKE_INSTALL_PREFIX}/lib
  RUNTIME DESTINATION ${CMAKE_INSTALL_PREFIX}/bin
)



# 添加可执行文件
add_executable(zed src/main.cpp)

# 包含头文件目录
target_include_directories(zed PRIVATE
    ${CUDA_INCLUDE_DIRS}
    ${TENSORRT_PATH}/include
    ${OpenCV_INCLUDE_DIRS}
    ${ZED_INCLUDE_DIRS}
    ${DEPLOY_PATH}
    ${DEPLOY_PATH}/include
)

# 链接库目录
target_link_directories(zed PRIVATE
    ${TENSORRT_PATH}/lib
)

# 私有源文件
target_sources(zed PRIVATE src/main.cpp)

# 私有编译定义
target_compile_definitions(zed PRIVATE ${CUDA_DEFINITIONS})

# 私有链接库
target_link_libraries(zed PRIVATE
    ${CUDA_cudart_LIBRARY}
    ${OpenCV_LIBS}
    ${ZED_LIBRARIES}
    ${OpenCV_LIBRARY_DIRS}
    ZedDeploy
    RtDBcomm 
    RtDBrtdb 
    RtDButils
    Boost::thread
    Boost::program_options
    yaml-cpp
)

# link_directories(${OpenCV_LIBRARY_DIRS})

# ----------------------------------------------------------------------------
# TensorRT Linking
# ----------------------------------------------------------------------------
target_link_libraries(zed PRIVATE
  ${TENSORRT_PATH}/lib/libnvinfer.so
  ${TENSORRT_PATH}/lib/libnvinfer_plugin.so
  ${TENSORRT_PATH}/lib/libnvonnxparser.so
  -Wl,-rpath=${TENSORRT_PATH}/lib
)

# ----------------------------------------------------------------------------
# Installation for Executable
# ----------------------------------------------------------------------------
install(TARGETS zed
  ARCHIVE DESTINATION ${CMAKE_INSTALL_PREFIX}/lib
  LIBRARY DESTINATION ${CMAKE_INSTALL_PREFIX}/lib
  RUNTIME DESTINATION ${CMAKE_INSTALL_PREFIX}/bin
)
