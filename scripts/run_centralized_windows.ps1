param(
    [switch]$StartMirror
)

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ComposeFile = Join-Path $ScriptDir "docker-compose.centralized.yml"
$EnvFile = Join-Path $ScriptDir ".env"
$ImageBundle = Get-ChildItem -Path $ScriptDir -Filter "smart-gym-centralized-images-*.tar" | Select-Object -First 1

if (-not $ImageBundle) {
    throw "No smart-gym-centralized-images-*.tar archive was found in $ScriptDir"
}

docker load -i $ImageBundle.FullName
docker compose --env-file $EnvFile -f $ComposeFile up -d postgres backend

if ($StartMirror) {
    if (-not $env:DISPLAY) {
        $env:DISPLAY = "host.docker.internal:0.0"
    }
    docker compose --env-file $EnvFile -f $ComposeFile --profile mirror up -d mirror
    Write-Host "Mirror container started. This requires WSLg or an X server on Windows."
} else {
    Write-Host "Backend and database started."
    Write-Host "Run this script with -StartMirror when WSLg or an X server is available."
}
