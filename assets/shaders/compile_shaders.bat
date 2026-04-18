@echo off

if "%VK_SDK_PATH%"=="" (
    echo environment variable VK_SDK_PATH must be set
    exit /b 1
)

"%VK_SDK_PATH%\Bin\glslc.exe" compute.comp -o compute.spv
"%VK_SDK_PATH%\Bin\glslc.exe" vertex.vert -o vertex.spv
"%VK_SDK_PATH%\Bin\glslc.exe" fragment.frag -o fragment.spv

pause