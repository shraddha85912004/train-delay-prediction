@echo off
echo ========================================================
echo   Pushing Train Delay Prediction to GitHub
echo   Repository: https://github.com/shraddha85912004/train-delay-prediction
echo ========================================================
echo.
echo If prompted, click "Sign in with your browser" to authorize.
echo.
git push -u origin main
echo.
if %ERRORLEVEL% EQU 0 (
    echo [SUCCESS] Repository pushed successfully to GitHub!
) else (
    echo [NOTE] If browser login did not appear, you can use a Personal Access Token:
    echo 1. Generate token at https://github.com/settings/tokens (select 'repo')
    echo 2. Run: git remote set-url origin https://YOUR_TOKEN@github.com/shraddha85912004/train-delay-prediction.git
    echo 3. Run: git push -u origin main
)
echo.
pause
