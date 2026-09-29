# 西瓜蒸馏接力 v2：等教师进程退出 -> 校验 100 轮全部完成 -> 启动 s->n 蒸馏
# 学生若显存不足（OOM/cuDNN 崩溃）自动降 batch 重试：32 -> 24 -> 16
$ErrorActionPreference = "Continue"
$env:KMP_DUPLICATE_LIB_OK = "TRUE"
$root = "C:\Users\13643\Downloads\jhupload"
Set-Location $root

$runDir   = "$root\runs\watermelon\watermelon_yolov8s_ep100"
$teacher  = "$runDir\weights\best.pt"
$csv      = "$runDir\results.csv"
$totalEpochs = 100

function Test-TeacherComplete {
    if (-not (Test-Path $csv)) { return $false }
    # results.csv 每个完成（含验证）的 epoch 一行；表头 + 数据行
    $rows = (Get-Content $csv | Measure-Object -Line).Lines - 1
    return ($rows -ge $totalEpochs)
}

# 1) 等教师训练进程退出
Write-Output "WAITING_FOR_TEACHER $(Get-Date -Format o)"
while ($true) {
    $t = Get-CimInstance Win32_Process -Filter "Name='python.exe'" -ErrorAction SilentlyContinue |
        Where-Object { $_.CommandLine -match 'train_wm_teacher' }
    if (-not $t) { break }
    Start-Sleep -Seconds 60
}
Start-Sleep -Seconds 10  # 等文件句柄释放

# 2) 校验教师确实跑完 100 轮（而非中途崩溃）
$rows = if (Test-Path $csv) { (Get-Content $csv | Measure-Object -Line).Lines - 1 } else { 0 }
Write-Output "TEACHER_PROCESS_EXITED $(Get-Date -Format o) completed_epochs=$rows"
if (-not (Test-TeacherComplete)) {
    Write-Output "TEACHER_INCOMPLETE: 只有 $rows/$totalEpochs 轮，不启动学生。修复后重跑本脚本。"
    exit 2
}
if (-not (Test-Path $teacher)) { Write-Output "TEACHER_WEIGHTS_MISSING"; exit 1 }
Write-Output "TEACHER_READY $(Get-Date -Format o) -> $teacher"

# 3) 启动学生蒸馏，OOM 自动降 batch
$batches = @(32, 24, 16)
foreach ($b in $batches) {
    Write-Output "STUDENT_START batch=$b $(Get-Date -Format o)"
    python -B -u scripts/train_distill.py `
        --model src/distill/yolov8n_student.yaml `
        --pretrained runs/watermelon/watermelon_yolov8n_ep100/weights/best.pt `
        --teacher $teacher --teacher-mode online --teacher-batch 4 `
        --data watermelon_local.yaml `
        --epochs 100 --batch $b --imgsz 640 --device 0 --workers 2 --amp `
        --mosaic 1.0 --mixup 0.1 --close-mosaic 15 --seed 0 `
        --project runs/watermelon --name distill_s2n_yolov8n_ep100
    $code = $LASTEXITCODE
    if ($code -eq 0) { Write-Output "STUDENT_DONE $(Get-Date -Format o)"; exit 0 }
    Write-Output "STUDENT_FAILED batch=$b exit=$code（等待 GPU 释放后尝试更小 batch）"
    Get-Process python -ErrorAction SilentlyContinue | Stop-Process -Force -ErrorAction SilentlyContinue
    Start-Sleep -Seconds 30
}
Write-Output "STUDENT_ALL_RETRIES_FAILED"
exit 1
