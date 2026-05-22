@echo off
REM Double-click entry point for the Nemo IB workflow.
REM Starts Docker + SearxNG, then launches Claude Code in this project.
REM Pin this file to the taskbar for one-click sessions.
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\nemo-launch.ps1"
