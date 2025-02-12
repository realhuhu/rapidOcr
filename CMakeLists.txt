cmake_minimum_required(VERSION 3.27)
project(RapidOcr)
set(CMAKE_CXX_STANDARD 17)
add_definitions(-DUNICODE -D_UNICODE)

include_directories("D:/onnxruntime/onnxruntime-win-x64-1.20.1/include")
link_directories("D:/onnxruntime/onnxruntime-win-x64-1.20.1/lib")

set(OpenCV_DIR "D:/opencv/dist_vc/install")
include(D:/opencv/dist_vc/install/OpenCVConfig.cmake)
find_package(OpenCV REQUIRED)

add_executable(RapidOcr
        src/AngleNet.h
        src/AngleNet.cpp
        src/CrnnNet.h
        src/CrnnNet.cpp
        src/DbNet.h
        src/DbNet.cpp
        src/OcrUtils.h
        src/OcrUtils.cpp
        src/OcrLite.h
        src/OcrLite.cpp
        src/clipper.hpp
        src/clipper.cpp
        src/OcrStruct.h
        main.cpp
)

target_compile_definitions(RapidOcr PRIVATE __EXEC__)
target_link_libraries(RapidOcr onnxruntime ${OpenCV_LIBS})
set_target_properties(RapidOcr PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}")

install(TARGETS RapidOcr EXPORT RapidOcr)
