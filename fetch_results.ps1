# Copy the results of a cluster run down to this machine.
#
# Usage (from the repo root):
#   .\fetch_results.ps1                                  # uses the "cluster" alias from ~/.ssh/config
#   .\fetch_results.ps1 -RemoteHost noah@login.example   # explicit user@host
#   .\fetch_results.ps1 -RemoteHost noah@login.example -Jump c7021201@jump.uibk.ac.at
#   .\fetch_results.ps1 -Summary                         # skip the per-sample images
#   .\fetch_results.ps1 -Streams 8                       # more parallelism
#   .\fetch_results.ps1 -Force                           # re-copy even files we already have
#   .\fetch_results.ps1 -Checksum                        # merge a new run into an old
#                                                        #   tree, re-fetching only what
#                                                        #   actually changed
#
# -Jump is only needed if the jump server is not already configured via
# ProxyJump in ~/.ssh/config.

param(
    [string]$RemoteHost = "cluster",
    [string]$RemotePath = "/scratch/noah/Null-Space-Networks",
    [string]$Dest = $PSScriptRoot,
    [string]$Jump = "",
    [int]$Streams = 6,
    [switch]$Force,
    [switch]$Summary,
    [switch]$Checksum
)

$ErrorActionPreference = "Stop"

$sshOpts = "-o BatchMode=yes"
if ($Jump) { $sshOpts += " -J $Jump" }

foreach ($p in @($Dest, $RemotePath)) {
    if ($p -like "*%*") { throw "Path contains '%', which cmd.exe would expand: $p" }
}

$work = Join-Path ([System.IO.Path]::GetTempPath()) ("fetch_results_" + [Guid]::NewGuid().ToString("N").Substring(0, 8))
New-Item -ItemType Directory -Force $work | Out-Null

function Write-LfFile($path, [string[]]$lines) {
    [System.IO.File]::WriteAllText($path, ($lines -join "`n") + "`n", (New-Object System.Text.UTF8Encoding $false))
}

function Format-Size([double]$bytes) {
    if ($bytes -ge 1GB) { return "{0:N2} GB" -f ($bytes / 1GB) }
    if ($bytes -ge 1MB) { return "{0:N1} MB" -f ($bytes / 1MB) }
    return "{0:N0} KB" -f ($bytes / 1KB)
}

# A figure is "per-sample" purely by its depth in the tree:
#   <run>/...                                  overview / aggregate  -> summary
#   <run>/epoch_study/...                      epoch curves          -> summary
#   <run>/init_<init>/...                      scatter / consistency -> summary
#   <run>/init_<init>/<attack>/...             per-attack figures    -> summary
#   <run>/init_<init>/<attack>/<model>/...     per-sample images     -> bulk
#   <run>/init_<init>/<attack>/<model>/worst/... worst-case images   -> bulk
# so 5 or more path segments means a per-sample image.
function Test-PerSampleFigure([string]$rel) {
    return ($rel.Split('/').Count -ge 5)
}

try {
    Write-Host "Listing ${RemoteHost}:${RemotePath}/attacks_* ..." -ForegroundColor Cyan
    $findCmd = "cd '$RemotePath' && find attacks_* -type f -printf '%s\t%p\n'"
    $raw = & ssh $sshOpts.Split(' ') $RemoteHost $findCmd
    if ($LASTEXITCODE -ne 0) {
        throw "ssh listing failed (exit $LASTEXITCODE). Check the host name, your ssh key, or whether any attacks_* directories exist under $RemotePath."
    }

    $remote = [ordered]@{}
    $rawCount = 0; $rawBytes = [int64]0
    $npzCount = 0; $npzBytes = [int64]0
    $figCount = 0; $figBytes = [int64]0
    foreach ($line in $raw) {
        if (-not $line) { continue }
        $tab = $line.IndexOf("`t")
        if ($tab -lt 1) { continue }
        $size = [int64]$line.Substring(0, $tab)
        $rel  = $line.Substring($tab + 1)
        $rawCount++; $rawBytes += $size
        $leaf = $rel.Substring($rel.LastIndexOf('/') + 1)
        # Render inputs. The cluster already turned these into figures.
        if ($leaf -like "*.npz") { $npzCount++; $npzBytes += $size; continue }
        if ($Summary -and $leaf -like "*.png" -and (Test-PerSampleFigure $rel)) {
            $figCount++; $figBytes += $size; continue
        }
        $remote[$rel] = $size
    }
    if ($rawCount -eq 0) { throw "No files found under $RemotePath/attacks_*" }

    Write-Host ("Remote: {0} files, {1}" -f $rawCount, (Format-Size $rawBytes))
    if ($npzCount) {
        Write-Host ("  excluding {0} .npz render input(s) ({1}) -- figures are rendered on the cluster" -f `
            $npzCount, (Format-Size $npzBytes)) -ForegroundColor DarkGray
    }
    if ($figCount) {
        Write-Host ("  excluding {0} per-sample image(s) ({1}) at -Summary -- drop -Summary for the full set" -f `
            $figCount, (Format-Size $figBytes)) -ForegroundColor DarkGray
    }
    if ($remote.Count -eq 0) {
        Write-Host "Nothing left to copy." -ForegroundColor Green
        return
    }

    # ---- 2. Diff against what is already here --------------------------------
    $todo = New-Object System.Collections.ArrayList
    foreach ($rel in $remote.Keys) {
        $local = Join-Path $Dest ($rel -replace '/', '\')
        if (-not $Force) {
            $fi = Get-Item -LiteralPath $local -ErrorAction SilentlyContinue
            if ($fi -and -not $fi.PSIsContainer -and $fi.Length -eq $remote[$rel]) { continue }
        }
        [void]$todo.Add([pscustomobject]@{ Rel = $rel; Size = $remote[$rel]; Local = $local })
    }

    # -Checksum: size said "same" for these; verify by content and put back any
    # that actually differ.
    if ($Checksum) {
        $todoSet = New-Object 'System.Collections.Generic.HashSet[string]'
        foreach ($t in $todo) { [void]$todoSet.Add($t.Rel) }
        $maybeSame = @($remote.Keys | Where-Object { -not $todoSet.Contains($_) })
        if ($maybeSame.Count -gt 0) {
            Write-Host ("Verifying {0} already-present file(s) by content hash ..." -f $maybeSame.Count) -ForegroundColor Cyan
            $listPath = Join-Path $work "verify-list.txt"
            $hashPath = Join-Path $work "verify-hashes.txt"
            $vcmdPath = Join-Path $work "verify-run.cmd"
            Write-LfFile $listPath $maybeSame
            Write-LfFile $vcmdPath @(
                "@echo off",
                "type ""$listPath"" | ssh $sshOpts $RemoteHost ""cd '$RemotePath' && xargs md5sum"" > ""$hashPath"""
            )
            $vp = Start-Process -FilePath "cmd.exe" -ArgumentList "/c", "`"$vcmdPath`"" `
                -NoNewWindow -PassThru -Wait
            $null = $vp.Handle

            $remoteHashes = @{}
            if (Test-Path -LiteralPath $hashPath) {
                foreach ($line in (Get-Content -LiteralPath $hashPath)) {
                    if ($line -match '^([0-9a-fA-F]{32})\s+(.+)$') {
                        $remoteHashes[$Matches[2].Trim()] = $Matches[1].ToLower()
                    }
                }
            }

            if ($remoteHashes.Count -eq 0) {
                Write-Host "Could not hash anything on the remote -- refusing to guess." -ForegroundColor Red
                Write-Host "Re-run without -Checksum to skip by size instead." -ForegroundColor Red
                exit 1
            }
            $missing = @($maybeSame | Where-Object { -not $remoteHashes.ContainsKey($_) })
            if ($missing.Count -gt 0) {
                Write-Host ("  warning: no remote hash for {0} file(s); re-fetching them rather than assuming unchanged, e.g. {1}" -f `
                    $missing.Count, $missing[0]) -ForegroundColor Yellow
            }

            $changed = 0
            foreach ($rel in $maybeSame) {
                $local = Join-Path $Dest ($rel -replace '/', '\')
                $rh = $remoteHashes[$rel]
                $lh = $null
                if ($rh) { $lh = (Get-FileHash -LiteralPath $local -Algorithm MD5).Hash.ToLower() }
                if (-not $rh -or $lh -ne $rh) {
                    [void]$todo.Add([pscustomobject]@{
                        Rel = $rel; Size = $remote[$rel]; Local = $local })
                    $changed++
                }
            }
            Write-Host ("  {0} unchanged, {1} differ and will be re-fetched" -f `
                ($maybeSame.Count - $changed), $changed) -ForegroundColor DarkGray
        }
    }

    if ($todo.Count -eq 0) {
        Write-Host "Everything is already up to date. Nothing to copy." -ForegroundColor Green
        return
    }

    $todoBytes = ($todo | Measure-Object -Property Size -Sum).Sum
    $skipped = $remote.Count - $todo.Count
    Write-Host ("To copy: {0} files, {1}{2}" -f $todo.Count, (Format-Size $todoBytes),
        $(if ($skipped) { " (skipping $skipped already present)" } else { "" })) -ForegroundColor Cyan

    # ---- 3. Keep the machine awake -------------------------------------------
    if (-not ("Win32.Power" -as [type])) {
        Add-Type -Namespace Win32 -Name Power -MemberDefinition @'
[DllImport("kernel32.dll", SetLastError = true)]
public static extern uint SetThreadExecutionState(uint esFlags);
'@
    }
    $ES_CONTINUOUS      = [uint32]2147483648
    $ES_SYSTEM_REQUIRED = [uint32]1
    [void][Win32.Power]::SetThreadExecutionState($ES_CONTINUOUS -bor $ES_SYSTEM_REQUIRED)

    # ---- 4. Transfer ---------------------------------------------------------
    function Invoke-TransferPass($files, [int]$streamCount, [string]$tag) {
        $n = [Math]::Min($streamCount, $files.Count)
        $buckets = @(); $loads = @()
        for ($i = 0; $i -lt $n; $i++) { $buckets += , (New-Object System.Collections.ArrayList); $loads += [int64]0 }

        # Longest-processing-time first: biggest files onto the lightest bucket.
        foreach ($f in ($files | Sort-Object -Property Size -Descending)) {
            $min = 0
            for ($i = 1; $i -lt $n; $i++) { if ($loads[$i] -lt $loads[$min]) { $min = $i } }
            [void]$buckets[$min].Add($f.Rel)
            $loads[$min] += $f.Size
        }

        $procs = @()
        for ($i = 0; $i -lt $n; $i++) {
            $listPath = Join-Path $work "$tag-list-$i.txt"
            $cmdPath  = Join-Path $work "$tag-run-$i.cmd"
            $errPath  = Join-Path $work "$tag-err-$i.txt"
            Write-LfFile $listPath $buckets[$i].ToArray()

            # A .cmd wrapper keeps the whole pipeline native: piping the tar
            # stream through PowerShell instead would corrupt it, because the
            # PS pipeline decodes bytes as text.
            Write-LfFile $cmdPath @(
                "@echo off",
                "type ""$listPath"" | ssh $sshOpts $RemoteHost ""cd '$RemotePath' && tar -cf - --ignore-failed-read -T -"" | tar -xf - -C ""$Dest"""
            )
            $p = Start-Process -FilePath "cmd.exe" -ArgumentList "/c", "`"$cmdPath`"" `
                -NoNewWindow -PassThru -RedirectStandardError $errPath
            # Touching Handle caches it; without this .ExitCode reads back as
            # $null once the process has gone away.
            $null = $p.Handle
            $procs += $p
        }

        # Progress: tar writes files as they land, so summing what has arrived
        # of the expected set is an honest measure of bytes moved.
        $started = Get-Date
        while ($procs | Where-Object { -not $_.HasExited }) {
            Start-Sleep -Seconds 5
            $have = [int64]0
            foreach ($f in $files) {
                $fi = Get-Item -LiteralPath $f.Local -ErrorAction SilentlyContinue
                if ($fi -and -not $fi.PSIsContainer) { $have += $fi.Length }
            }
            $elapsed = (Get-Date) - $started
            $rate = if ($elapsed.TotalSeconds -gt 0) { $have / $elapsed.TotalSeconds } else { 0 }
            $want = ($files | Measure-Object -Property Size -Sum).Sum
            $pct  = if ($want) { 100.0 * $have / $want } else { 100 }
            $eta  = if ($rate -gt 0 -and $have -lt $want) {
                [TimeSpan]::FromSeconds(($want - $have) / $rate).ToString("hh\:mm\:ss")
            } else { "--:--:--" }
            Write-Host ("`r  {0,5:N1}%  {1} / {2}  at {3}/s  ETA {4}   " -f `
                $pct, (Format-Size $have), (Format-Size $want), (Format-Size $rate), $eta) -NoNewline
        }
        Write-Host ""

        # HasExited going true is not enough for ExitCode to be readable on a
        # -PassThru process; WaitForExit caches it.
        foreach ($p in $procs) { $p.WaitForExit() }

        for ($i = 0; $i -lt $n; $i++) {
            if ($procs[$i].ExitCode -ne 0) {
                $err = Get-Content (Join-Path $work "$tag-err-$i.txt") -Raw -ErrorAction SilentlyContinue
                if ($null -eq $err) { $err = "" }
                Write-Host ("  stream {0} exited {1}: {2}" -f $i, $procs[$i].ExitCode, $err.Trim()) -ForegroundColor Yellow
            }
        }

        # Re-stat rather than trusting exit codes: cmd reports the status of the
        # last stage of the pipe, so a failed ssh can still leave local tar at 0.
        return @($files | Where-Object {
            $fi = Get-Item -LiteralPath $_.Local -ErrorAction SilentlyContinue
            -not $fi -or $fi.PSIsContainer -or $fi.Length -ne $_.Size
        })
    }

    try {
        Write-Host ("Copying with {0} parallel streams ..." -f ([Math]::Min($Streams, $todo.Count))) -ForegroundColor Cyan
        $pending = Invoke-TransferPass $todo $Streams "p0"

        $attempt = 1
        while ($pending.Count -gt 0 -and $attempt -le 2) {
            Write-Host ("Retry {0}: {1} file(s) incomplete, re-fetching ..." -f $attempt, $pending.Count) -ForegroundColor Yellow
            $pending = Invoke-TransferPass $pending $Streams "p$attempt"
            $attempt++
        }
    } finally {
        [void][Win32.Power]::SetThreadExecutionState($ES_CONTINUOUS)
    }

    # ---- 5. Report -----------------------------------------------------------
    if ($pending.Count -gt 0) {
        Write-Host ("Done with errors: {0} file(s) still missing or the wrong size, e.g." -f $pending.Count) -ForegroundColor Red
        $pending | Select-Object -First 5 | ForEach-Object { Write-Host "  $($_.Rel)" -ForegroundColor Red }
        Write-Host "Re-run the script to retry just those." -ForegroundColor Red
        exit 1
    }

    $copied = Get-ChildItem -Path $Dest -Directory -Filter "attacks_*" | Select-Object -ExpandProperty Name
    Write-Host ("Done. Copied {0} ({1} files). Local attacks_* directories: {2}" -f `
        (Format-Size $todoBytes), $todo.Count, ($copied -join ', ')) -ForegroundColor Green
} finally {
    Remove-Item -Recurse -Force $work -ErrorAction SilentlyContinue
}
