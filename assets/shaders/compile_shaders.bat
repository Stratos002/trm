@echo off

if "%VK_SDK_PATH%"=="" (
    echo environment variable VK_SDK_PATH must be set
    exit /b 1
)

"%VK_SDK_PATH%\Bin\glslc.exe" compute.comp -o compute.spv

pause