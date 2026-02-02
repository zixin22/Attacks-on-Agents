# AutoDan_webarena 启动脚本 (PowerShell版本)
# 自动设置API密钥环境变量并运行优化

Write-Host "🚀 启动 AutoDan_webarena 进化优化" -ForegroundColor Green
Write-Host "=" * 50 -ForegroundColor Yellow

# 设置API密钥环境变量
$env:OPENAI_API_KEY = "sk-O0g7ou2ojOXl9EI77pWKFeFfwLBzNQFmDw6EJ8MkHH74FRb9"

# 切换到AutoDan目录 (确保在正确目录)
Set-Location "D:\rap-main\webshop\AutoDan\AutoDan_webarena"

Write-Host "✅ API密钥已设置" -ForegroundColor Green
Write-Host "📂 工作目录: $(Get-Location)" -ForegroundColor Cyan
Write-Host "🎯 启动优化程序..." -ForegroundColor Green

# 运行优化程序
python run_optimization.py -t "web navigation optimization"
