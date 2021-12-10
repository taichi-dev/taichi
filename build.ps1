$stopwatch = [system.diagnostics.stopwatch]::startNew()
python setup.py develop
$stopwatch.Elapsed
