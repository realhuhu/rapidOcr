cmake_minimum_required(VERSION 3.12)
project(RapidOcr)
set(CMAKE_CXX_STANDARD 17)
add_definitions(-DUNICODE -D_UNICODE)

include(D:/rapidOCR/Project_RapidOcrOnnx-1.2.2/onnxruntime-static/OnnxRuntimeWrapper.cmake)
find_package(OnnxRuntime REQUIRED)

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
target_link_libraries(RapidOcr ${OnnxRuntime_LIBS} ${OpenCV_LIBS})
set_target_properties(RapidOcr PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}")

install(TARGETS RapidOcr EXPORT RapidOcr)

foreach(flag_var
        CMAKE_C_FLAGS CMAKE_C_FLAGS_DEBUG CMAKE_C_FLAGS_RELEASE
        CMAKE_C_FLAGS_MINSIZEREL CMAKE_C_FLAGS_RELWITHDEBINFO
        CMAKE_CXX_FLAGS CMAKE_CXX_FLAGS_DEBUG CMAKE_CXX_FLAGS_RELEASE
        CMAKE_CXX_FLAGS_MINSIZEREL CMAKE_CXX_FLAGS_RELWITHDEBINFO)
    if(${flag_var} MATCHES "/MD")
        string(REGEX REPLACE "/MD" "/MT" ${flag_var} "${${flag_var}}")
    endif()
    if(${flag_var} MATCHES "/MDd")
        string(REGEX REPLACE "/MDd" "/MTd" ${flag_var} "${${flag_var}}")
    endif()
endforeach(flag_var)

#set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} /NODEFAULTLIB:atlthunk.lib")
#set(CMAKE_EXE_LINKER_FLAGS_DEBUG "${CMAKE_EXE_LINKER_FLAGS_DEBUG} /NODEFAULTLIB:libcmt.lib /NODEFAULTLIB:libcpmt.lib /NODEFAULTLIB:msvcrt.lib")
#set(CMAKE_EXE_LINKER_FLAGS_RELEASE "${CMAKE_EXE_LINKER_FLAGS_RELEASE} /NODEFAULTLIB:libcmtd.lib /NODEFAULTLIB:libcpmtd.lib /NODEFAULTLIB:msvcrtd.lib")


