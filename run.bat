@echo off
chcp 65001 >nul
setlocal enabledelayedexpansion

echo.
echo ========================================
echo    Image Processing Batch Tool
echo ========================================
echo.

:: Check if input folder exists
if not exist "input" (
    echo ERROR: 'input' folder does not exist!
    echo Creating 'input' folder...
    mkdir input
    echo.
    echo Please place your image files in the 'input' folder and run the script again.
    echo.
    pause
    exit /b
)

:: Check if input folder is empty
dir /b "input\*" >nul 2>&1
if errorlevel 1 (
    echo ERROR: 'input' folder is empty!
    echo.
    echo Please place your image files in the 'input' folder and run the script again.
    echo.
    pause
    exit /b
)

:: List all image files in input folder
echo Available images in input folder:
echo.
set count=0
for %%f in (input\*.*) do (
    set /a count+=1
    echo [!count!] %%~nxf
)
echo.

:: Task selection menu
:SHOW_MENU
echo Please select a processing task:
echo.
echo [1] dewarping
echo [2] deshadowing  
echo [3] appearance
echo [4] deblurring
echo [5] binarization
echo [6] end2end
echo.
set /p choice="Enter your choice (1-6): "

:: Validate input
if not defined choice goto SHOW_MENU

if "%choice%"=="1" set selected_task=dewarping & goto MEMORY_MENU
if "%choice%"=="2" set selected_task=deshadowing & goto MEMORY_MENU
if "%choice%"=="3" set selected_task=appearance & goto MEMORY_MENU
if "%choice%"=="4" set selected_task=deblurring & goto MEMORY_MENU
if "%choice%"=="5" set selected_task=binarization & goto MEMORY_MENU
if "%choice%"=="6" set selected_task=end2end & goto MEMORY_MENU

echo.
echo Invalid choice! Please enter a number between 1-6.
echo.
goto SHOW_MENU

:: Memory fix selection menu
:MEMORY_MENU
echo.
echo Select memory fix option:
echo [0] memory_fix 0
echo [1] memory_fix 1
echo [2] memory_fix 2
echo [3] memory_fix 3
echo.
set /p mem_choice="Enter your choice (0-3): "

if "%mem_choice%"=="0" set memory_fix=0 & goto TASK_SELECTED
if "%mem_choice%"=="1" set memory_fix=1 & goto TASK_SELECTED
if "%mem_choice%"=="2" set memory_fix=2 & goto TASK_SELECTED
if "%mem_choice%"=="3" set memory_fix=3 & goto TASK_SELECTED

echo Invalid choice! Please enter a number between 0-3.
goto MEMORY_MENU

:TASK_SELECTED
echo.
echo Selected task: !selected_task!
echo Memory fix option: !memory_fix!
echo.

:: Confirmation
:CONFIRM
set /p confirm="Do you want to process all images with '!selected_task!' and memory_fix=!memory_fix!? (y/n): "
if /i "%confirm%"=="y" goto RUN_PROCESSING
if /i "%confirm%"=="n" (
    echo Operation cancelled.
    pause
    exit /b
)
echo Please enter y or n.
goto CONFIRM

:RUN_PROCESSING
echo.
echo Starting processing...
echo.

:: Process all images in input folder
set success_count=0
set total_count=0

for %%f in (input\*.*) do (
    set /a total_count+=1
    echo Processing: %%~nxf
    
    call conda activate docresfork
    python inference.py --im_path "%%f" --task !selected_task! --memory_fix !memory_fix!
    
    if !errorlevel! equ 0 (
        echo ✓ Success: %%~nxf
        set /a success_count+=1
    ) else (
        echo ✗ Failed: %%~nxf
    )
    echo.
)

echo.
echo ========================================
echo Processing completed!
echo Successfully processed: !success_count! of !total_count! images
echo.
pause
