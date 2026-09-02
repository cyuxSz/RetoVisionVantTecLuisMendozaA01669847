cmake_minimum_required(VERSION 3.16)
project(OpenCVPruebaVantTec LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(OpenCVPruebaVantTec main.cpp)

TARGET_LINK_LIBRARIES(OpenCVPruebaVantTec ${OpenCV_LIBS})

# Copia el modelo ONNX junto al .exe cada vez que se compila
add_custom_command(TARGET OpenCVPruebaVantTec POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${CMAKE_SOURCE_DIR}/yolo26n.onnx"
        "$<TARGET_FILE_DIR:OpenCVPruebaVantTec>/yolo26n.onnx"
)

include(CTest)
enable_testing()

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)