<#
.SYNOPSIS
    Run simple-motion in the background: starts at every logon, no window,
    restarted if it ever exits. Log: motion.log in the repo folder.

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File windows\install-task.ps1
    powershell -ExecutionPolicy Bypass -File windows\install-task.ps1 -Camera 1
    powershell -ExecutionPolicy Bypass -File windows\install-task.ps1 -Camera rtsp://127.0.0.1:8554/cam
    powershell -ExecutionPolicy Bypass -File windows\install-task.ps1 -Camera 0 -ExtraArgs --hold,30
    powershell -ExecutionPolicy Bypass -File windows\install-task.ps1 -Camera rtsp://127.0.0.1:8554/cam -ExtraArgs --serve,8890
    powershell -ExecutionPolicy Bypass -File windows\install-task.ps1 -Stop
    powershell -ExecutionPolicy Bypass -File windows\install-task.ps1 -Uninstall

.NOTES
    The task runs as you, in your desktop session ("Run only when user is
    logged on"): webcams and Bluetooth are not available to background
    services or SSH sessions on Windows.
#>
param(
    [string]$Camera = '',          # index or stream URL; empty = CAMERA_INDEX
    [string[]]$ExtraArgs = @(),    # more motion_detect.py flags
    [string]$TaskName = 'SimpleMotion',
    [switch]$Stop,
    [switch]$Uninstall
)
$ErrorActionPreference = 'Stop'
# powershell -File passes "--hold,30" as one string, not a list: split it.
$ExtraArgs = @($ExtraArgs | ForEach-Object { $_ -split ',' } | Where-Object { $_ -ne '' })
$root = Split-Path -Parent $PSScriptRoot
$launcher = Join-Path $PSScriptRoot 'simple-motion.vbs'
$script = Join-Path $root 'motion_detect.py'

function Stop-Detector {
    if (Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue) {
        Stop-ScheduledTask -TaskName $TaskName
    }
    # Ending the task stops only the launcher; stop the detector it started.
    # (A venv python.exe is a stub that starts the real one: match both.)
    Get-CimInstance Win32_Process -Filter "Name = 'python.exe'" |
        Where-Object { $_.CommandLine -like "*$script*--headless*" } |
        ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }
}

if ($Stop -or $Uninstall) {
    Stop-Detector
    if ($Uninstall) {
        Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false -ErrorAction SilentlyContinue
        "Removed task $TaskName."
    } else {
        "Stopped. It starts again at next logon, or now with: Start-ScheduledTask $TaskName"
    }
    "If the strip was on, it stays on until the detector runs again."
    return
}

if (-not (Test-Path (Join-Path $root '.venv\Scripts\python.exe'))) {
    throw "No .venv in $root. First run: py -m venv .venv; .venv\Scripts\pip install -r requirements.txt"
}

# Not $env:USERDOMAIN: over SSH it reads WORKGROUP, which is not an account.
$me = [Security.Principal.WindowsIdentity]::GetCurrent().Name
$arguments = @('//B', '//NoLogo', "`"$launcher`"")
if ($Camera) { $arguments += @('--camera', "`"$Camera`"") }
$arguments += $ExtraArgs | ForEach-Object { "`"$_`"" }

$action = New-ScheduledTaskAction -Execute 'wscript.exe' -Argument ($arguments -join ' ') -WorkingDirectory $root
$trigger = New-ScheduledTaskTrigger -AtLogOn -User $me
$principal = New-ScheduledTaskPrincipal -UserId $me -LogonType Interactive -RunLevel Limited
$settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries `
    -ExecutionTimeLimit ([TimeSpan]::Zero) -MultipleInstances IgnoreNew -StartWhenAvailable `
    -RestartCount 999 -RestartInterval (New-TimeSpan -Minutes 1)

Stop-Detector  # re-install: replace a running copy instead of adding a second one
Register-ScheduledTask -TaskName $TaskName -Action $action -Trigger $trigger -Principal $principal `
    -Settings $settings -Description "simple-motion motion detector ($root)" -Force | Out-Null
Start-ScheduledTask -TaskName $TaskName
"Installed and started task $TaskName. Log: $(Join-Path $root 'motion.log')"
