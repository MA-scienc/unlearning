<#
.SYNOPSIS
  Live, read-only dashboard for the unlearning pilot.

.DESCRIPTION
  Run it in a second terminal while the pipeline or a training stage runs:

      powershell -ExecutionPolicy Bypass -File scripts\dashboard.ps1

  It reads runs\<run_id>\pipeline\status.json, the newest stage status.json,
  metrics.jsonl and events.jsonl, and queries nvidia-smi. It never writes
  anything, so it cannot disturb training. Works in Windows PowerShell 5.1 and
  PowerShell 7 (pwsh, also on Linux).

.PARAMETER StorageRoot
  Folder that contains runs\. Default: UNLEARNING_STORAGE_ROOT from the
  environment or .env, else the repository folder.

.PARAMETER Refresh
  Seconds between refreshes (default 5).

.PARAMETER Once
  Print a single frame and exit.
#>
param(
    [string]$StorageRoot = "",
    [int]$Refresh = 5,
    [switch]$Once
)

$ErrorActionPreference = "Continue"
try { [Console]::OutputEncoding = [System.Text.Encoding]::UTF8 } catch {}
$RepoRoot = Split-Path -Parent $PSScriptRoot

function Read-DotEnv([string]$Path) {
    $values = @{}
    if (Test-Path -LiteralPath $Path) {
        foreach ($raw in Get-Content -LiteralPath $Path) {
            $line = $raw.Trim()
            if ($line -and -not $line.StartsWith("#") -and $line.Contains("=")) {
                $parts = $line.Split("=", 2)
                $values[$parts[0].Trim()] = $parts[1].Trim()
            }
        }
    }
    return $values
}

function Resolve-StorageRoot {
    $candidate = $StorageRoot
    if (-not $candidate) { $candidate = $env:UNLEARNING_STORAGE_ROOT }
    if (-not $candidate) {
        $dotenv = Read-DotEnv (Join-Path $RepoRoot ".env")
        if ($dotenv.ContainsKey("UNLEARNING_STORAGE_ROOT")) { $candidate = $dotenv["UNLEARNING_STORAGE_ROOT"] }
    }
    if (-not $candidate) { $candidate = "." }
    if (-not [System.IO.Path]::IsPathRooted($candidate)) { $candidate = Join-Path $RepoRoot $candidate }
    return [System.IO.Path]::GetFullPath($candidate)
}

function Read-SharedText([string]$Path) {
    # Open with full sharing so the trainer can keep replacing the file.
    try {
        $share = [System.IO.FileShare]"ReadWrite, Delete"
        $stream = [System.IO.File]::Open($Path, [System.IO.FileMode]::Open, [System.IO.FileAccess]::Read, $share)
        try {
            $reader = New-Object System.IO.StreamReader($stream)
            return $reader.ReadToEnd()
        } finally { $stream.Dispose() }
    } catch { return $null }
}

function Read-JsonFile([string]$Path) {
    $text = Read-SharedText $Path
    if (-not $text) { return $null }
    try { return $text | ConvertFrom-Json } catch { return $null }
}

function Read-JsonLines([string]$Path, [int]$Last = 300) {
    $text = Read-SharedText $Path
    if (-not $text) { return @() }
    $lines = $text -split "`n" | Where-Object { $_.Trim() }
    if ($lines.Count -gt $Last) { $lines = $lines[($lines.Count - $Last)..($lines.Count - 1)] }
    $rows = @()
    foreach ($line in $lines) { try { $rows += ($line | ConvertFrom-Json) } catch {} }
    return $rows
}

function Format-Duration($Seconds) {
    if ($null -eq $Seconds) { return "?" }
    $total = [int][math]::Round([double]$Seconds)
    $hours = [math]::Floor($total / 3600)
    $minutes = [math]::Floor(($total % 3600) / 60)
    $secs = $total % 60
    if ($hours -gt 0) { return ("{0}h{1:00}m" -f $hours, $minutes) }
    if ($minutes -gt 0) { return ("{0}m{1:00}s" -f $minutes, $secs) }
    return ("{0}s" -f $secs)
}

function Get-AgeSeconds($Iso) {
    if (-not $Iso) { return $null }
    try { return ([DateTimeOffset]::UtcNow - [DateTimeOffset]::Parse($Iso)).TotalSeconds } catch { return $null }
}

function Get-Sparkline($Values, [int]$Width = 60) {
    $numbers = @($Values | Where-Object { $null -ne $_ } | ForEach-Object { [double]$_ })
    if ($numbers.Count -lt 2) { return "" }
    $bucket = [math]::Max(1, [math]::Ceiling($numbers.Count / $Width))
    $points = @()
    for ($i = 0; $i -lt $numbers.Count; $i += $bucket) {
        $end = [math]::Min($numbers.Count - 1, $i + $bucket - 1)
        $points += (($numbers[$i..$end] | Measure-Object -Average).Average)
    }
    $min = ($points | Measure-Object -Minimum).Minimum
    $max = ($points | Measure-Object -Maximum).Maximum
    $span = $max - $min
    $chars = @()
    foreach ($point in $points) {
        $level = 0
        if ($span -gt 0) { $level = [int][math]::Floor(7 * ($point - $min) / $span) }
        $chars += [char](0x2581 + $level)
    }
    return (-join $chars) + ("   min {0:N4}  max {1:N4}" -f $min, $max)
}

function Get-ProgressBar([double]$Percent, [int]$Width = 30) {
    $filled = [int][math]::Round($Width * [math]::Min(100, [math]::Max(0, $Percent)) / 100)
    return "[" + ([string][char]0x2588 * $filled) + ([string][char]0x2591 * ($Width - $filled)) + "]"
}

function Get-Gpus {
    $smi = Get-Command nvidia-smi -ErrorAction SilentlyContinue
    if (-not $smi) { return @() }
    $fields = "name,temperature.gpu,utilization.gpu,power.draw,power.limit,memory.used,memory.total,clocks.sm,clocks_event_reasons.active"
    $output = & nvidia-smi "--query-gpu=$fields" --format=csv,noheader,nounits 2>$null
    if ($LASTEXITCODE -ne 0) {
        $fields = $fields.Replace("clocks_event_reasons", "clocks_throttle_reasons")
        $output = & nvidia-smi "--query-gpu=$fields" --format=csv,noheader,nounits 2>$null
    }
    $gpus = @()
    foreach ($line in @($output)) {
        $p = $line.Split(",") | ForEach-Object { $_.Trim() }
        if ($p.Count -lt 9) { continue }
        $reasons = @()
        try {
            $mask = [Convert]::ToInt64($p[8].Replace("0x", ""), 16)
            if ($mask -band 0x4) { $reasons += "power cap" }
            if ($mask -band 0x8) { $reasons += "hw slowdown" }
            if ($mask -band 0x20) { $reasons += "SW THERMAL" }
            if ($mask -band 0x40) { $reasons += "HW THERMAL" }
            if ($mask -band 0x80) { $reasons += "power brake" }
        } catch {}
        $gpus += [pscustomobject]@{
            Name = $p[0]; Temp = $p[1]; Util = $p[2]; Power = $p[3]; PowerLimit = $p[4]
            MemUsed = $p[5]; MemTotal = $p[6]; Clock = $p[7]; Throttle = ($reasons -join ", ")
        }
    }
    return $gpus
}

function Get-Number($Value) {
    if ($null -eq $Value) { return $null }
    $parsed = 0.0
    if ([double]::TryParse([string]$Value, [System.Globalization.NumberStyles]::Float, [System.Globalization.CultureInfo]::InvariantCulture, [ref]$parsed)) { return $parsed }
    return $null
}

function Write-Line([string]$Text, [string]$Color = "Gray") { Write-Host $Text -ForegroundColor $Color }

function Show-Frame {
    $root = Resolve-StorageRoot
    $runs = Join-Path $root "runs"
    $warnings = New-Object System.Collections.Generic.List[string]
    Write-Line ("UNLEARNING DASHBOARD   {0}   (refresh {1}s, Ctrl+C to quit)" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"), $Refresh) "Cyan"
    Write-Line ("storage {0}" -f $root) "DarkGray"

    if (-not (Test-Path -LiteralPath $runs)) {
        Write-Line "No runs yet. Start with: python scripts\pilot\run_pipeline.py run" "Yellow"
        return
    }

    $pipelineFile = Get-ChildItem -Path $runs -Filter status.json -Recurse -ErrorAction SilentlyContinue |
        Where-Object { $_.Directory.Name -eq "pipeline" } | Sort-Object LastWriteTime -Descending | Select-Object -First 1
    if ($pipelineFile) {
        $pipeline = Read-JsonFile $pipelineFile.FullName
        if ($pipeline) {
            $color = "Green"
            if ($pipeline.state -eq "failed") { $color = "Red" }
            elseif ($pipeline.state -in @("waiting_for_gate", "interrupted", "stopped")) { $color = "Yellow" }
            Write-Line ("run      {0}" -f $pipeline.run_id) "White"
            Write-Line ("pipeline {0}   {1}/{2} jobs done   profile {3}" -f $pipeline.state, $pipeline.done, $pipeline.total, $pipeline.runtime_profile) $color
            if ($pipeline.current_job) { Write-Line ("         current: {0} - {1}" -f $pipeline.current_job.id, $pipeline.current_job.title) "Gray" }
            if ($pipeline.message) { Write-Line ("         {0}" -f $pipeline.message) "DarkYellow" }
        }
    }

    $stageFile = Get-ChildItem -Path $runs -Filter status.json -Recurse -ErrorAction SilentlyContinue |
        Where-Object { $_.Directory.Name -ne "pipeline" } | Sort-Object LastWriteTime -Descending | Select-Object -First 1
    Write-Line ("-" * 78) "DarkGray"
    if (-not $stageFile) {
        Write-Line "No training stage has started yet." "Gray"
    } else {
        $status = Read-JsonFile $stageFile.FullName
        $stageDir = $stageFile.Directory.FullName
        if ($status) {
            $alive = $false
            if ($status.pid) { $alive = [bool](Get-Process -Id $status.pid -ErrorAction SilentlyContinue) }
            $age = Get-AgeSeconds $status.updated_at
            $stateColor = "Green"
            if ($status.state -in @("failed")) { $stateColor = "Red" }
            elseif ($status.state -in @("interrupted", "paused_thermal", "saving", "loading")) { $stateColor = "Yellow" }
            elseif ($status.state -eq "completed") { $stateColor = "Cyan" }
            $aliveText = "not running"
            if ($alive) { $aliveText = "alive" }
            Write-Line ("STAGE {0}   state {1}   pid {2} ({3})   updated {4} ago" -f $status.stage, $status.state, $status.pid, $aliveText, (Format-Duration $age)) $stateColor
            $percent = 0.0
            if ($null -ne $status.percent) { $percent = [double]$status.percent }
            Write-Line ("progress {0} {1}/{2}  {3:N1}%   ETA {4}   {5:N2} s/step" -f (Get-ProgressBar $percent), $status.step, $status.total_steps, $percent, (Format-Duration $status.eta_seconds), (Get-Number $status.mean_step_seconds)) "White"
            $m = $status.last_metrics
            if ($m) {
                $dev = ""
                if ($status.dev_loss) { $dev = ("   dev {0:N4} @ step {1}" -f $status.dev_loss.value, $status.dev_loss.step) }
                Write-Line ("loss {0:N4} (smoothed {1:N4}){2}   lr {3:E2}   grad norm {4:N3}" -f (Get-Number $m.loss), (Get-Number $status.loss_ema), $dev, (Get-Number $m.lr), (Get-Number $m.grad_norm)) "White"
                Write-Line ("throughput {0:N0} tok/s   peak GPU memory {1:N2} GB allocated / {2:N2} GB reserved   micro-batch {3} x {4} process(es)" -f (Get-Number $m.tokens_per_second), (Get-Number $m.peak_memory_gb), (Get-Number $m.peak_reserved_gb), $status.micro_batch, $status.processes) "Gray"
                if ($null -eq $m.loss) { $warnings.Add("loss is not a finite number") }
            }
            if ($status.last_resume_point) { Write-Line ("last resume point: step {0}" -f $status.last_resume_point.step) "Gray" }
            if ($status.last_scientific_checkpoint) { Write-Line ("last scientific checkpoint: {0} (step {1})" -f $status.last_scientific_checkpoint.name, $status.last_scientific_checkpoint.step) "Gray" }
            if ($status.resume_count -gt 0) { Write-Line ("resumed {0} time(s)" -f $status.resume_count) "Gray" }
            $metrics = Read-JsonLines (Join-Path $stageDir "metrics.jsonl") 400
            if ($metrics.Count -gt 1) { Write-Line ("loss trend " + (Get-Sparkline ($metrics | ForEach-Object { $_.loss }))) "Cyan" }
            if ($status.state -in @("running", "saving", "loading") -and -not $alive) { $warnings.Add("trainer process is gone but status says '$($status.state)': it crashed or was killed. Rerun the pipeline (it resumes).") }
            if ($status.state -eq "running" -and $null -ne $age -and $age -gt 300) { $warnings.Add("no progress for $(Format-Duration $age): stalled?") }
            if ($status.state -eq "failed" -and $status.message) { $warnings.Add("FAILED: $($status.message)") }
            if ($status.state -eq "paused_thermal") { $warnings.Add("paused to let the GPU cool: $($status.message)") }

            $events = Read-JsonLines (Join-Path $stageDir "events.jsonl") 4
            if ($events.Count -gt 0) {
                Write-Line "recent events:" "DarkGray"
                foreach ($event in $events) { Write-Line ("  {0}  {1}: {2}" -f $event.time, $event.kind, $event.message) "DarkGray" }
            }
            if ($status.system) {
                $sys = $status.system
                Write-Line ("host  CPU {0}%   RAM {1}/{2} GB   disk free {3} GB" -f $sys.cpu_pct, $sys.ram_used_gb, $sys.ram_total_gb, $sys.disk_free_gb) "Gray"
                $free = Get-Number $sys.disk_free_gb
                if ($null -ne $free -and $free -lt 20) { $warnings.Add("only $free GB free on the storage disk") }
            }
        }
    }

    Write-Line ("-" * 78) "DarkGray"
    foreach ($gpu in Get-Gpus) {
        $limit = ""
        if ($gpu.PowerLimit -and $gpu.PowerLimit -ne "[N/A]") { $limit = "/" + $gpu.PowerLimit }
        $throttle = "none"
        if ($gpu.Throttle) { $throttle = $gpu.Throttle }
        $temp = Get-Number $gpu.Temp
        $color = "Green"
        if ($null -ne $temp -and $temp -ge 85) { $color = "Red" } elseif ($null -ne $temp -and $temp -ge 78) { $color = "Yellow" }
        Write-Line ("GPU  {0}   {1} C   util {2}%   {3}{4} W   mem {5}/{6} MB   SM {7} MHz   throttle: {8}" -f $gpu.Name, $gpu.Temp, $gpu.Util, $gpu.Power, $limit, $gpu.MemUsed, $gpu.MemTotal, $gpu.Clock, $throttle) $color
        if ($null -ne $temp -and $temp -ge 85) { $warnings.Add("GPU at $temp C") }
        if ($gpu.Throttle -match "THERMAL") { $warnings.Add("GPU is thermally throttling: improve cooling (stand, fan, cooler room)") }
        $used = Get-Number $gpu.MemUsed; $total = Get-Number $gpu.MemTotal
        if ($used -and $total -and ($used / $total) -gt 0.96) { $warnings.Add("GPU memory almost full: Windows may spill into shared system memory (very slow). Lower the micro-batch and --resume.") }
    }

    try {
        $battery = Get-CimInstance Win32_Battery -ErrorAction Stop | Select-Object -First 1
        if ($battery) {
            if ($battery.BatteryStatus -eq 2) {
                Write-Line ("power on AC (battery {0}%)" -f $battery.EstimatedChargeRemaining) "Gray"
            } else {
                Write-Line ("power ON BATTERY ({0}%)" -f $battery.EstimatedChargeRemaining) "Red"
                $warnings.Add("laptop is on battery: training slows sharply and the machine sleeps at critical battery. Plug in the charger.")
            }
        }
    } catch {}

    if ($warnings.Count -gt 0) {
        Write-Line ("-" * 78) "DarkGray"
        foreach ($warning in $warnings) { Write-Line ("WARNING  " + $warning) "Yellow" }
    }
}

while ($true) {
    if (-not $Once) { Clear-Host }
    Show-Frame
    if ($Once) { break }
    Start-Sleep -Seconds $Refresh
}
