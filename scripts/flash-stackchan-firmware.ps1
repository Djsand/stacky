param(
    [string]$Port = "COM3",
    [switch]$NoBuild
)

$ErrorActionPreference = "Stop"

$RepoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
$FirmwarePath = Join-Path $RepoRoot "vendor\m5stack-stackchan\firmware"
$EspIdfPath = "C:\Users\nicol\esp\esp-idf-v5.5.4"

if (-not (Test-Path $FirmwarePath)) {
    throw "Firmware path not found: $FirmwarePath"
}
if (-not (Test-Path (Join-Path $EspIdfPath "export.ps1"))) {
    throw "ESP-IDF export.ps1 not found: $EspIdfPath"
}

function Ensure-SubstDrive {
    param(
        [string]$Drive,
        [string]$Target
    )

    $current = (cmd /c "subst $Drive" 2>$null)
    if ($LASTEXITCODE -eq 0 -and $current) {
        $mapped = ($current -replace "^[A-Z]:\\: => ", "").Trim()
        if ($mapped -and ((Resolve-Path $mapped).Path -ne (Resolve-Path $Target).Path)) {
            throw "$Drive is already mapped to $mapped"
        }
        return
    }
    cmd /c "subst $Drive `"$Target`""
}

Ensure-SubstDrive -Drive "S:" -Target $FirmwarePath
Ensure-SubstDrive -Drive "I:" -Target $EspIdfPath

& "I:\export.ps1"
$env:IDF_PATH = "I:\"
Set-Location "S:\"

if (-not $NoBuild) {
    idf.py build
}

idf.py -p $Port flash
