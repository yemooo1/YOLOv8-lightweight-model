# 西瓜重参数化蒸馏：RepC2f 学生 + yolov8s 在线教师，OOM 自动降 batch 32->16
$ErrorActionPreference = "Continue"
$env:KMP_DUPLICATE_LIB_OK = "TRUE"
$root = "C:\Users\13643\Downloads\jhupload"
Set-Location $root

$teacher = "$root\runs\watermelon\watermelon_yolov8s_ep100\weights\best.pt"
$batches = @(32, 16)
foreach ($b in $batches) {
    Write-Output "REP_STUDENT_START batch=$b $(Get-Date -Format o)"
    python -B -u scripts/train_distill.py `
        --model src/distill/yolov8n_student.yaml `
        --pretrained runs/watermelon/watermelon_yolov8n_ep100/weights/best.pt `
        --teacher $teacher --teacher-mode online --teacher-batch 4 `
        --data watermelon_local.yaml `
        --epochs 100 --batch $b --imgsz 640 --device 0 --workers 2 --amp `
        --mosaic 1.0 --mixup 0.1 --close-mosaic 15 --seed 0 --rep `
        --project runs/watermelon --name distill_rep_s2n_yolov8n_ep100
    $code = $LASTEXITCODE
    if ($code -eq 0) { Write-Output "REP_STUDENT_DONE $(Get-Date -Format o)"; exit 0 }
    Write-Output "REP_STUDENT_FAILED batch=$b exit=$code（30s 后降配重试）"
    Get-Process python -ErrorAction SilentlyContinue | Stop-Process -Force -ErrorAction SilentlyContinue
    Start-Sleep -Seconds 30
}
Write-Output "REP_STUDENT_ALL_RETRIES_FAILED"
exit 1
