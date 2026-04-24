$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent (Split-Path -Parent $PSCommandPath)
$AnnotationDir = Join-Path $Root "annotations"
New-Item -ItemType Directory -Force -Path $AnnotationDir | Out-Null

function Download-File {
    param(
        [Parameter(Mandatory = $true)][string]$Url,
        [Parameter(Mandatory = $true)][string]$OutFile,
        [int]$MaxTime = 900
    )
    if (Test-Path -LiteralPath $OutFile) {
        $len = (Get-Item -LiteralPath $OutFile).Length
        if ($len -gt 0) {
            Write-Output "SKIP $OutFile ($len bytes)"
            return $true
        }
    }
    Write-Output "Downloading $Url"
    & curl.exe -L --retry 5 --retry-delay 3 --connect-timeout 30 --max-time $MaxTime --fail --silent --show-error $Url -o $OutFile
    if ($LASTEXITCODE -ne 0) {
        Write-Output "FAILED $Url"
        return $false
    }
    Write-Output "OK $OutFile ($((Get-Item -LiteralPath $OutFile).Length) bytes)"
    return $true
}

$ZipPath = Join-Path $AnnotationDir "captions.zip"
$OfficialOk = Download-File -Url "https://cs.stanford.edu/people/ranjaykrishna/densevid/captions.zip" -OutFile $ZipPath
if ($OfficialOk) {
    Expand-Archive -LiteralPath $ZipPath -DestinationPath $AnnotationDir -Force
}

$MirrorBase = "https://huggingface.co/datasets/friedrichor/ActivityNet_Captions/resolve/main/raw_data"
foreach ($file in @("train.json", "val_1.json", "val_2.json")) {
    $out = Join-Path $AnnotationDir $file
    if (!(Test-Path -LiteralPath $out) -or ((Get-Item -LiteralPath $out).Length -eq 0)) {
        $ok = Download-File -Url "$MirrorBase/$file" -OutFile $out
        if (-not $ok) {
            throw "Could not download $file from official zip or mirror."
        }
    }
}

$verify = @'
import json
from pathlib import Path
root = Path.cwd().parents[0] / "annotations"
for name in ["train.json", "val_1.json", "val_2.json"]:
    path = root / name
    data = json.load(open(path, encoding="utf-8"))
    segments = sum(len(item["sentences"]) for item in data.values())
    print(f"{name}: {len(data)} videos, {segments} segments")
'@
Push-Location (Join-Path $Root "scripts")
try {
    $verify | python -
} finally {
    Pop-Location
}

