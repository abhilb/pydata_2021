cmake_minimum_required(VERSION 3.16)

if(NOT CMAKE_BUILD_TYPE)
    message(STATUS "Setting build type to Release")
    set(CMAKE_BUILD_TYPE "Release")
endif()

project(DigitsDemo VERSION 0.1)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED 17)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

message(STATUS "ONNX RUNTIME DIR: ${ONNXRUNTIME_ROOTDIR}")

if(NOT ONNXRUNTIME_ROOTDIR)
  if(WIN32)
    set(ONNXRUNTIME_ROOTDIR "C:/Program Files (x86)/onnxruntime")
  else()
    set(ONNXRUNTIME_ROOTDIR "/usr/local/include/onnxruntime")
  endif()
endif()

set(ONNX_INCLUDE_DIRS "${ONNXRUNTIME_ROOTDIR}/include"
                      "${ONNXRUNTIME_ROOTDIR}/include/onnxruntime/core/session")
set(ONNX_LIB_DIR "${ONNXRUNTIME_ROOTDIR}/lib")

find_package(SDL2 REQUIRED)
find_package(OpenGL REQUIRED)
find_package(OpenCV REQUIRED)

set(SRC_FILES main.cpp
              imgui/imgui.cpp
              imgui/imgui_draw.cpp
              imgui/imgui_tables.cpp
              imgui/imgui_widgets.cpp
              imgui/backends/imgui_impl_sdl.cpp
              imgui/backends/imgui_impl_opengl2.cpp)

add_executable(${PROJECT_NAME} ${SRC_FILES})

target_include_directories(${PROJECT_NAME} PRIVATE ${CMAKE_SOURCE_DIR}/spdlog/include)
target_include_directories(${PROJECT_NAME} PUBLIC "imgui")
target_include_directories(${PROJECT_NAME} PUBLIC "imgui/backends")
target_include_directories(${PROJECT_NAME} PUBLIC ${SDL2_INCLUDE_DIRS})
target_include_directories(${PROJECT_NAME} PUBLIC ${OPENGL_INCLUDE_DIRS})
target_include_directories(${PROJECT_NAME} PUBLIC ${OpencV_INCLUDE_DIRS})
target_include_directories(${PROJECT_NAME} PUBLIC ${ONNX_INCLUDE_DIRS})

target_link_libraries(${PROJECT_NAME} PUBLIC ${SDL2_LIBRARIES})
target_link_libraries(${PROJECT_NAME} PUBLIC ${OPENGL_LIBRARIES})
target_link_libraries(${PROJECT_NAME} PUBLIC ${OpenCV_LIBS})
target_link_libraries(${PROJECT_NAME} PUBLIC onnxruntime.lib)

target_link_directories(${PROJECT_NAME} PUBLIC ${ONNX_LIB_DIR})