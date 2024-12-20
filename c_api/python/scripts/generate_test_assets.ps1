Get-ChildItem tests/assets/*.py | ForEach-Object {
    Write-Host "Generating test asset for $($_)"
    & ti module build "$($_)" -o "$($_).tcm"
}
