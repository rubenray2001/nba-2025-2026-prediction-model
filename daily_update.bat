@echo off
REM NBA Model Daily Update - Runs at 3 AM Pacific Time
REM This script updates data and retrains the model

cd /d "c:\Users\ruben\OneDrive\Desktop\nba apy new"

echo ========================================
echo NBA Model Daily Update Starting...
echo %date% %time%
echo ========================================

python update_all.py --retrain

echo ========================================
echo Update Complete: %date% %time%
echo ========================================

REM Log the result
echo %date% %time% - Update completed >> daily_update_log.txt

echo.
echo Window will stay open so you can review the output.
echo Press any key to close...
pause >nul
