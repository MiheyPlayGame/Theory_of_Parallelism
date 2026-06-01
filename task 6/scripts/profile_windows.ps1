param(
  [ValidateSet("host", "gpu", "multicore")]
  [string]$Mode = "gpu",
  [int]$Size = 256,
  [int]$Iters = 1000000,
  [string]$Distro = "",
  [string]$LinuxProjectPath = "/home/mihey/Theory_of_Parallelism/task 6"
)

$ErrorActionPreference = "Stop"

$linuxCmd = "cd '$LinuxProjectPath' && ./scripts/profile_nsight.sh $Mode $Size $Iters"

if ([string]::IsNullOrWhiteSpace($Distro)) {
  wsl.exe -- bash -lc $linuxCmd
} else {
  wsl.exe -d $Distro -- bash -lc $linuxCmd
}
