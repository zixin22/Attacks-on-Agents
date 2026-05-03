# AutoDan_osagent 启动脚本 (PowerShell版本)
# API 密钥：使用 webshop/OpenAI_api_key.txt（代码不再读取 OPENAI_API_KEY 环境变量）

Write-Host "🚀 启动 AutoDan_osagent 进化优化" -ForegroundColor Green
Write-Host "=" * 50 -ForegroundColor Yellow

# 切换到AutoDan目录 (确保在正确目录)
Set-Location "D:\rap-main\webshop\AutoDan\AutoDan_osagent"

Write-Host "✅ 请确认 webshop/OpenAI_api_key.txt 已填写" -ForegroundColor Green
Write-Host "📂 工作目录: $(Get-Location)" -ForegroundColor Cyan
Write-Host "🎯 启动优化程序..." -ForegroundColor Green

# 运行优化程序
python run_optimization.py -t "web navigation optimization"
