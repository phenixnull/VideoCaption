$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent (Split-Path -Parent $PSCommandPath)
$AnnotationDir = Join-Path $Root "annotations"
New-Item -ItemType Directory -Force -Path $AnnotationDir | Out-Null

function Download-File {
    param(
        [Parameter(Mandatory = $true)][string]$Url,
        [Parameter(Mandatory = $true)][string]$OutFile,
        [int]$MaxTime = 1800
    )
    if (Test-Path -LiteralPath $OutFile) {
        $len = (Get-Item -LiteralPath $OutFile).Length
        if ($len -gt 0) {
            Write-Output "SKIP $OutFile ($len bytes)"
            return
        }
    }
    Write-Output "Downloading $Url"
    & curl.exe -L --retry 5 --retry-delay 3 --connect-timeout 30 --max-time $MaxTime --fail --silent --show-error $Url -o $OutFile
    if ($LASTEXITCODE -ne 0) {
        throw "curl failed for $Url"
    }
    Write-Output "OK $OutFile ($((Get-Item -LiteralPath $OutFile).Length) bytes)"
}

$BaseUrl = "https://eric-xw.github.io/vatex-website/data"
$Files = @(
    "vatex_training_v1.0.json",
    "vatex_validation_v1.0.json",
    "vatex_public_test_english_v1.1.json",
    "vatex_private_test_without_annotations.json"
)

foreach ($file in $Files) {
    Download-File -Url "$BaseUrl/$file" -OutFile (Join-Path $AnnotationDir $file)
}

$verify = @'
import json
from pathlib import Path
root = Path.cwd().parents[0] / "annotations"
for path in sorted(root.glob("*.json")):
    data = json.load(open(path, encoding="utf-8"))
    print(f"{path.name}: {len(data)} records")
'@
Push-Location (Join-Path $Root "scripts")
try {
    $verify | python -
} finally {
    Pop-Location
}
