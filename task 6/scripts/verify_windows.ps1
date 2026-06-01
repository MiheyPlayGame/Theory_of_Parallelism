param(
  [string]$Distro = "",
  [string]$LinuxProjectPath = "/home/mihey/Theory_of_Parallelism/task 6"
)

$ErrorActionPreference = "Stop"

$linuxCmd = "cd '$LinuxProjectPath' && make verify"

if ([string]::IsNullOrWhiteSpace($Distro)) {
  wsl.exe -- bash -lc $linuxCmd
} else {
  wsl.exe -d $Distro -- bash -lc $linuxCmd
}
