Set-Location $PSScriptRoot

if (Test-Path ".\.venv\Scripts\stealzalut-gui.exe") {
    Start-Process ".\.venv\Scripts\stealzalut-gui.exe" -ArgumentList $args
} else {
    uv run stealzalut-gui @args
}

