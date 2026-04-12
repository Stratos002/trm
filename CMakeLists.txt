cmake_minimum_required(VERSION 3.10)

set(PROJECT_NAME trm)
project(${PROJECT_NAME})

# C99
set(CMAKE_C_STANDARD 99)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS OFF)

# sources
file(GLOB_RECURSE SOURCES src/*.c src/*.h)

add_executable(
	${PROJECT_NAME} 
	${SOURCES}  )

target_include_directories(
	${PROJECT_NAME} 
	PRIVATE
	${CMAKE_CURRENT_SOURCE_DIR}/external/volk
	${CMAKE_CURRENT_SOURCE_DIR}/external/glfw/include
	${CMAKE_CURRENT_SOURCE_DIR}/external/vulkan_headers)

# warnings and errors
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
        -Wformat=2)
endif()

# platform detection
if(WIN32)
	message(STATUS "platform : windows")
	target_compile_definitions(${PROJECT_NAME} PRIVATE TRM_PLATFORM_WINDOWS)
elseif(UNIX AND NOT APPLE)
	message(STATUS "platform : linux")
	target_compile_definitions(${PROJECT_NAME} PRIVATE TRM_PLATFORM_LINUX)
elseif(APPLE)
	message(FATAL_ERROR "unsupported platform: macOS is not supported")
else()
	message(FATAL_ERROR "unknown platform")
endif()

target_compile_definitions(${PROJECT_NAME} PRIVATE VK_NO_PROTOTYPES)

add_subdirectory(external/glfw)

target_link_libraries(${PROJECT_NAME} PRIVATE glfw)
