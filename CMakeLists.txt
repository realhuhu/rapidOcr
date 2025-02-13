#cmake option: -T "v142,host=x64" -A x64 ..

cmake_minimum_required(VERSION 3.27)
project(RapidOcr)

set(CMAKE_CXX_STANDARD 17)
add_definitions(-DUNICODE -D_UNICODE)

include_directories("D:/onnxruntime/onnxruntime-win-x64-1.20.1/include")
link_directories("D:/onnxruntime/onnxruntime-win-x64-1.20.1/lib")

set(OpenCV_DIR "D:/opencv/dist_vc/install")
find_package(OpenCV REQUIRED)

add_library(rapid_ocr SHARED
        ${CMAKE_SOURCE_DIR}/src/AngleNet.cpp
        ${CMAKE_SOURCE_DIR}/src/CrnnNet.cpp
        ${CMAKE_SOURCE_DIR}/src/DbNet.cpp
        ${CMAKE_SOURCE_DIR}/src/OcrUtils.cpp
        ${CMAKE_SOURCE_DIR}/src/OcrLite.cpp
        ${CMAKE_SOURCE_DIR}/src/clipper.cpp
)
target_include_directories(rapid_ocr PUBLIC ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(rapid_ocr PUBLIC onnxruntime ${OpenCV_LIBS})
target_compile_definitions(rapid_ocr PRIVATE OCR_EXPORTS)
set_target_properties(rapid_ocr PROPERTIES ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR})

add_executable(RapidOcr main.cpp)

target_include_directories(RapidOcr PUBLIC ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(RapidOcr PUBLIC rapid_ocr)
set_target_properties(RapidOcr PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR})
install(TARGETS RapidOcr EXPORT RapidOcr)
