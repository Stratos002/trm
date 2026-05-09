cmake_minimum_required(VERSION 3.10)

project(trm C)

# C99
set(CMAKE_C_STANDARD 99)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS OFF)

file(GLOB_RECURSE SOURCES
    src/*.c
)

add_library(${PROJECT_NAME} STATIC ${SOURCES})

target_include_directories(${PROJECT_NAME}
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include

    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        ${CMAKE_CURRENT_SOURCE_DIR}/external/volk
        ${CMAKE_CURRENT_SOURCE_DIR}/external/vulkan_headers
)

if(MSVC)
    target_compile_options(${PROJECT_NAME} PRIVATE /W4 /WX)
else()
    target_compile_options(${PROJECT_NAME} PRIVATE
        -Wall
        -Wextra
        -Wpedantic
        -Werror
        -Wshadow
        -Wconversion
        -Wsign-conversion
        -Wstrict-prototypes
        -Wmissing-prototypes
        -Wnull-dereference
        -Wdouble-promotion
        -Werror=implicit-function-declaration
        -Wformat=2
    )
endif()

if(WIN32)
    message(STATUS "platform : windows")
    target_compile_definitions(${PROJECT_NAME} PUBLIC TRM_PLATFORM_WINDOWS)
    target_compile_definitions(${PROJECT_NAME} PUBLIC VK_USE_PLATFORM_WIN32_KHR)

elseif(UNIX AND NOT APPLE)
    message(STATUS "platform : linux")
    target_compile_definitions(${PROJECT_NAME} PUBLIC TRM_PLATFORM_LINUX)
    target_compile_definitions(${PROJECT_NAME} PUBLIC VK_USE_PLATFORM_WAYLAND_KHR)

elseif(APPLE)
    message(FATAL_ERROR "unsupported platform: macOS is not supported")
else()
    message(FATAL_ERROR "unknown platform")
endif()

target_compile_definitions(${PROJECT_NAME} PUBLIC VK_NO_PROTOTYPES)
target_compile_definitions(${PROJECT_NAME} PUBLIC PROJECT_ROOT="${CMAKE_SOURCE_DIR}")