# =============================================================================
# setup_vlm_windows.ps1
# Registra el etiquetado automatico VLM (Gemini) en el Programador de Tareas
# =============================================================================

$NombreTarea = "TFM_VLM_Gemini"
$RutaBat = "C:\Users\jorge\OneDrive\Escritorio\Master CEU\TFM v2\lanzar_vlm.bat"
$IntervaloHoras = 12

Write-Host "============================================================"
Write-Host " CONFIGURACION DEL VLM AUTOMATICO - TFM"
Write-Host "============================================================"
Write-Host ""

$tareaExistente = Get-ScheduledTask -TaskName $NombreTarea -ErrorAction SilentlyContinue
if ($tareaExistente) {
    Unregister-ScheduledTask -TaskName $NombreTarea -Confirm:$false
    Write-Host "[OK] Tarea anterior eliminada para reconfigurar."
}

$accion = New-ScheduledTaskAction `
    -Execute "cmd.exe" `
    -Argument "/c `"$RutaBat`""

# Repetir cada 12 horas durante 1 anyo
$disparador = New-ScheduledTaskTrigger `
    -RepetitionInterval (New-TimeSpan -Hours $IntervaloHoras) `
    -RepetitionDuration (New-TimeSpan -Days 365) `
    -Once `
    -At (Get-Date).AddMinutes(2)

$configuracion = New-ScheduledTaskSettingsSet `
    -ExecutionTimeLimit (New-TimeSpan -Hours 2) `
    -StartWhenAvailable `
    -RunOnlyIfNetworkAvailable `
    -WakeToRun:$false

Register-ScheduledTask `
    -TaskName $NombreTarea `
    -Action $accion `
    -Trigger $disparador `
    -Settings $configuracion `
    -Description "Etiquetado automatico VLM (Gemini) cada $IntervaloHoras horas" `
    -Force | Out-Null

Write-Host ""
Write-Host "[OK] Tarea '$NombreTarea' registrada con exito."
Write-Host "Frecuencia: Cada $IntervaloHoras horas"
Write-Host "============================================================"
