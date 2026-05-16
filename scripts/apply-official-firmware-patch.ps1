param(
    [switch]$ForceReset
)

$ErrorActionPreference = "Stop"

$repoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
$submodulePath = Join-Path $repoRoot "vendor\m5stack-stackchan"
$patchPath = Join-Path $repoRoot "patches\official-stackchan\0001-stacky-bridge.patch"

function Invoke-Git {
    param(
        [Parameter(Mandatory = $true)]
        [string[]]$Args
    )

    & git @Args
    if ($LASTEXITCODE -ne 0) {
        throw "git $($Args -join ' ') failed with exit code $LASTEXITCODE"
    }
}

if (-not (Test-Path $patchPath)) {
    throw "Missing patch: $patchPath"
}

Push-Location $repoRoot
try {
    Invoke-Git @("submodule", "update", "--init", "--recursive", "vendor/m5stack-stackchan")

    if ($ForceReset) {
        Invoke-Git @("-C", $submodulePath, "reset", "--hard", "HEAD")
        Invoke-Git @("-C", $submodulePath, "clean", "-fd")
    }

    $previousErrorActionPreference = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    & git -C $submodulePath apply --unidiff-zero --reverse --check $patchPath 2>$null
    $reverseCheckExitCode = $LASTEXITCODE
    $ErrorActionPreference = $previousErrorActionPreference

    if ($reverseCheckExitCode -eq 0) {
        Write-Host "Stacky official firmware patch is already applied."
        exit 0
    }

    $status = & git -C $submodulePath status --short
    if ($LASTEXITCODE -ne 0) {
        throw "Could not read submodule status."
    }

    if ($status) {
        Write-Host "Submodule has local changes:"
        $status | ForEach-Object { Write-Host "  $_" }
        Write-Host ""
        Write-Host "Run again with -ForceReset to discard submodule changes and reapply Stacky's patch."
        exit 2
    }

    & git -C $submodulePath apply --unidiff-zero --check $patchPath
    if ($LASTEXITCODE -ne 0) {
        throw "Patch does not apply cleanly to vendor/m5stack-stackchan."
    }

    Invoke-Git @("-C", $submodulePath, "apply", "--unidiff-zero", $patchPath)
    Write-Host "Applied Stacky official firmware patch to vendor/m5stack-stackchan."
    Write-Host "The parent repo will now show the submodule as dirty until you reset it or commit a submodule fork pointer."
}
finally {
    Pop-Location
}
