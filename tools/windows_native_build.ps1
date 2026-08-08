param(
    [string]$BuildDir = "native/build-msvc",
    [string]$Configuration = "Release",
    [switch]$SkipPythonSmoke
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$RepoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
$NativeDir = Join-Path $RepoRoot "native"
$BuildPath = Join-Path $RepoRoot $BuildDir

function Resolve-CMake {
    $command = Get-Command cmake -ErrorAction SilentlyContinue
    if ($null -ne $command) {
        return $command.Source
    }

    $defaultPath = "C:\Program Files\CMake\bin\cmake.exe"
    if (Test-Path $defaultPath) {
        return $defaultPath
    }

    throw "cmake was not found. Install Kitware.CMake with winget or add CMake to PATH."
}

function Resolve-VcVars64 {
    $vswhere = "C:\Program Files (x86)\Microsoft Visual Studio\Installer\vswhere.exe"
    if (Test-Path $vswhere) {
        $installPath = & $vswhere -latest -products * `
            -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 `
            -property installationPath
        if (-not [string]::IsNullOrWhiteSpace($installPath)) {
            $candidate = Join-Path $installPath "VC\Auxiliary\Build\vcvars64.bat"
            if (Test-Path $candidate) {
                return $candidate
            }
        }
    }

    $defaultPath = "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
    if (Test-Path $defaultPath) {
        return $defaultPath
    }

    throw "vcvars64.bat was not found. Install Visual Studio Build Tools with the C++ workload."
}

function Quote-CmdArg([string]$Value) {
    return '"' + $Value + '"'
}

$cmake = Resolve-CMake
$ctest = Join-Path (Split-Path $cmake -Parent) "ctest.exe"
if (-not (Test-Path $ctest)) {
    $ctestCommand = Get-Command ctest -ErrorAction SilentlyContinue
    if ($null -eq $ctestCommand) {
        throw "ctest was not found next to cmake or on PATH."
    }
    $ctest = $ctestCommand.Source
}
$vcvars = Resolve-VcVars64

Write-Host "Repo: $RepoRoot"
Write-Host "CMake: $cmake"
Write-Host "MSVC environment: $vcvars"
Write-Host "Build dir: $BuildPath"

$configure = "$(Quote-CmdArg $cmake) -S $(Quote-CmdArg $NativeDir) -B $(Quote-CmdArg $BuildPath) -G ""Visual Studio 17 2022"" -A x64 -DCMAKE_BUILD_TYPE=$Configuration"
$build = "$(Quote-CmdArg $cmake) --build $(Quote-CmdArg $BuildPath) --config $Configuration --parallel"
$test = "$(Quote-CmdArg $ctest) --test-dir $(Quote-CmdArg $BuildPath) -C $Configuration --output-on-failure"
$command = "$(Quote-CmdArg $vcvars) && $configure && $build && $test"

cmd.exe /s /c $command
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}

if (-not $SkipPythonSmoke) {
    $python = Join-Path $RepoRoot ".venv\Scripts\python.exe"
    if (-not (Test-Path $python)) {
        $pythonCommand = Get-Command python -ErrorAction SilentlyContinue
        if ($null -eq $pythonCommand) {
            throw "python was not found. Create .venv or add python to PATH."
        }
        $python = $pythonCommand.Source
    }

    $dll = Join-Path $BuildPath "$Configuration\vecadvisor_distance.dll"
    if (-not (Test-Path $dll)) {
        throw "Native DLL was not produced: $dll"
    }

    $oldNativeLib = $env:VECADVISOR_NATIVE_DISTANCE_LIB
    try {
        $env:VECADVISOR_NATIVE_DISTANCE_LIB = $dll
        Push-Location $RepoRoot
        & $python -m pytest tests/test_native_distance_integration.py -q
        if ($LASTEXITCODE -ne 0) {
            exit $LASTEXITCODE
        }
    }
    finally {
        Pop-Location
        $env:VECADVISOR_NATIVE_DISTANCE_LIB = $oldNativeLib
    }
}
