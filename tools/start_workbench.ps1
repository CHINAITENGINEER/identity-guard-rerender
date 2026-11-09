$workspace = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $workspace
$workbenchTemp = Join-Path $workspace "ui_workspace\temp"
New-Item -ItemType Directory -Force -Path $workbenchTemp | Out-Null
$env:TEMP = $workbenchTemp
$env:TMP = $workbenchTemp
python -m strict_compositor.ui
