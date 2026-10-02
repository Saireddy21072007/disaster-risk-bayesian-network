@echo off
setlocal
cd /d "%~dp0"

REM ---------------------------------------------------------------------------
REM 22AIE301 project helper.
REM
REM Double-click this file and you get a menu. Or pass a command directly:
REM   run.bat setup      install the packages
REM   run.bat app        start the dashboard on http://127.0.0.1:8100
REM   run.bat all        simulate -> learn -> evaluate -> figures
REM   run.bat test       run the test suite
REM   run.bat ppt        rebuild the review slide deck
REM   run.bat feeds      one live pull from the public APIs
REM   run.bat observed   fetch 6 years of real observations, relearn weather layer
REM   run.bat live       full real-time pass against the live APIs
REM   run.bat replay     same, from the saved snapshot (works offline)
REM ---------------------------------------------------------------------------

REM --- is python even here? ---
where python >nul 2>nul
if errorlevel 1 (
    echo.
    echo   ERROR: Python was not found on your PATH.
    echo.
    echo   Install Python 3.11 or newer from python.org and tick
    echo   "Add python.exe to PATH" during installation, then run this again.
    echo.
    pause
    exit /b 1
)

set "CMD=%~1"
if "%CMD%"=="" goto menu
goto dispatch

REM ---------------------------------------------------------------------------
:menu
cls
echo ===========================================================================
echo   Explainable Multi-Hazard Disaster Prediction  --  22AIE301
echo   Sai Reddy A  .  Rohit Vardhan M  .  Jithin Reddy K  .  Sai Vandith
echo ===========================================================================
echo.
echo   FIRST TIME? Run option 1 once, then option 2.
echo.
echo     1  Install the required packages          (do this first, once)
echo     2  Start the dashboard                    (opens in your browser)
echo     3  Run the test suite                     (85 tests, ~20 seconds)
echo.
echo     4  Rebuild the review slide deck
echo     5  Rebuild all data + figures             (simulate, learn, evaluate)
echo.
echo     6  One live pull from the weather APIs    (needs internet)
echo     7  Full real-time pass, all 16 districts  (needs internet)
echo     8  Real-time pass from saved snapshot     (works offline)
echo     9  Fetch 6 years of real observations     (slow, needs internet)
echo.
echo     0  Exit
echo.
set "CHOICE="
set /p "CHOICE=  Type a number and press Enter: "

if "%CHOICE%"=="1" set "CMD=setup"    & goto dispatch
if "%CHOICE%"=="2" set "CMD=app"      & goto dispatch
if "%CHOICE%"=="3" set "CMD=test"     & goto dispatch
if "%CHOICE%"=="4" set "CMD=ppt"      & goto dispatch
if "%CHOICE%"=="5" set "CMD=all"      & goto dispatch
if "%CHOICE%"=="6" set "CMD=feeds"    & goto dispatch
if "%CHOICE%"=="7" set "CMD=live"     & goto dispatch
if "%CHOICE%"=="8" set "CMD=replay"   & goto dispatch
if "%CHOICE%"=="9" set "CMD=observed" & goto dispatch
if "%CHOICE%"=="0" exit /b 0

echo.
echo   "%CHOICE%" is not one of the options.
echo.
pause
goto menu

REM ---------------------------------------------------------------------------
:dispatch

if /i "%CMD%"=="setup" (
    echo.
    echo   Installing packages. This takes a few minutes the first time.
    echo.
    python -m pip install -r requirements.txt
    if errorlevel 1 goto failed
    echo.
    echo   Done. Now choose option 2 to start the dashboard.
    goto done
)

if /i "%CMD%"=="app" (
    echo.
    echo   Starting the dashboard...
    echo.
    REM NOTE ON PARENTHESES: never put a bare ( or ) in an echo inside a block
    REM like this one. cmd parses the whole block first, treats the bracket as
    REM block structure, and dies with "... was unexpected at this time" - which
    REM makes the window vanish the instant you choose this option. Use ^( ^) or,
    REM better, avoid brackets in these messages entirely.

    REM If a server is already listening, do not start a second one - it cannot
    REM bind to the same port and the error is confusing. Reuse the live one.
    set "ALREADY="
    python -c "import socket,sys; s=socket.socket(); r=s.connect_ex(('127.0.0.1',8100)); s.close(); sys.exit(0 if r==0 else 1)" >nul 2>nul
    if not errorlevel 1 set "ALREADY=1"
    if defined ALREADY (
        echo   A server is ALREADY running on port 8100, so we will reuse it.
        echo   To start a fresh one, close its window first and choose 2 again.
        goto openbrowser
    )

    echo   The server runs in a SECOND window. Your browser will open by itself.
    echo   To stop the server later, close that second window.
    echo.
    start "Disaster Prediction - server" cmd /k python -m uvicorn app.main:app --port 8100
    echo   Waiting for the server to answer. On a cold start this takes about
    echo   10 to 20 seconds while it loads the model...
    echo.
    REM Poll the health endpoint instead of guessing a delay. Opening the browser
    REM too early used to leave the page half-initialised with a black map.
    set "READY="
    for /l %%i in (1,1,40) do (
        if not defined READY (
            python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8100/api/health',timeout=2).status==200 else 1)" >nul 2>nul
            if not errorlevel 1 set "READY=1"
            REM ping, not timeout: timeout refuses to run when stdin is
            REM redirected and spams "Input redirection is not supported"
            if not defined READY ping -n 2 127.0.0.1 >nul
        )
    )
    if not defined READY (
        echo   The server did not answer in time. Look at the OTHER window for
        echo   the real error. The usual cause is that the packages are not
        echo   installed yet - choose option 1 - or that something else is
        echo   holding port 8100.
        goto done
    )
    goto openbrowser
)

if /i "%CMD%"=="test" (
    python -m pytest tests -q
    goto done
)

if /i "%CMD%"=="ppt" (
    python report\make_ppt.py
    if errorlevel 1 goto failed
    echo.
    echo   Deck written to report\Review_MultiHazardBN.pptx
    goto done
)

if /i "%CMD%"=="all" (
    python -m src.simulate  || goto failed
    python -m src.learn     || goto failed
    python -m src.evaluate  || goto failed
    python -m src.viz       || goto failed
    python -m src.gis monsoon_depression || goto failed
    echo.
    echo   Done. Artifacts are in artifacts\
    goto done
)

if /i "%CMD%"=="feeds" (
    python -m src.feeds
    goto done
)

if /i "%CMD%"=="observed" (
    echo.
    echo   Fetching six years of real observations for 16 districts.
    echo   This takes a few minutes and needs internet. Results are cached.
    echo.
    python -m src.observed
    goto done
)

if /i "%CMD%"=="live" (
    python -m src.stream
    goto done
)

if /i "%CMD%"=="replay" (
    python -m src.stream --snapshot
    goto done
)

echo.
echo   Unknown command "%CMD%".
echo   Try: setup ^| app ^| test ^| ppt ^| all ^| feeds ^| observed ^| live ^| replay
echo.
pause
exit /b 1

REM ---------------------------------------------------------------------------
:openbrowser
start "" http://127.0.0.1:8100
echo.
echo   Ready. Opened http://127.0.0.1:8100 in your browser.
echo   In the header, use the "Live feeds" button for the real-time pipeline.
echo   If the page ever looks blank, press Ctrl+F5 to hard-refresh it.
goto done

REM ---------------------------------------------------------------------------
:failed
echo.
echo   That step failed. The error is above.
echo   Most common cause: the packages are not installed yet - run option 1.
echo.
pause
exit /b 1

:done
echo.
REM only pause if we were double-clicked (no argument on the command line)
if "%~1"=="" pause
exit /b 0
