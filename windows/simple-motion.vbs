' simple-motion background launcher for Windows.
'
' Runs motion_detect.py --headless with no console window and restarts it
' 10 s after it exits (crash, camera unplugged, stream server not up yet).
' Output goes to motion.log in the repo folder; at 5 MB it is rotated to
' motion.log.1. Arguments are passed through to motion_detect.py, e.g.
'     wscript simple-motion.vbs --camera rtsp://127.0.0.1:8554/cam
' The scheduled task from install-task.ps1 starts this at logon.
Option Explicit

Dim sh, fso, root, py, script, logPath, extra, i, cmd, code
Set sh = CreateObject("WScript.Shell")
Set fso = CreateObject("Scripting.FileSystemObject")

root = fso.GetParentFolderName(fso.GetParentFolderName(WScript.ScriptFullName))
py = root & "\.venv\Scripts\python.exe"
script = root & "\motion_detect.py"
logPath = root & "\motion.log"

extra = ""
For i = 0 To WScript.Arguments.Count - 1
    extra = extra & " " & Quote(WScript.Arguments(i))
Next

sh.CurrentDirectory = root
sh.Environment("PROCESS")("PYTHONIOENCODING") = "utf-8"
' cmd /s strips only the outer quotes, so every inner path can be quoted.
cmd = "cmd /s /c """ & Quote(py) & " -u " & Quote(script) & " --headless" & extra & _
      " >> " & Quote(logPath) & " 2>&1"""

Do
    RotateLog
    WriteLog "=== " & Now & " starting:" & extra
    code = sh.Run(cmd, 0, True)
    WriteLog "=== " & Now & " exited with code " & code & ", restarting in 10 s"
    WScript.Sleep 10000
Loop

Function Quote(s)
    Quote = """" & s & """"
End Function

Sub WriteLog(line)
    Dim f
    Set f = fso.OpenTextFile(logPath, 8, True)  ' 8 = append, create if missing
    f.WriteLine line
    f.Close
End Sub

Sub RotateLog()
    If Not fso.FileExists(logPath) Then Exit Sub
    If fso.GetFile(logPath).Size < 5242880 Then Exit Sub
    If fso.FileExists(logPath & ".1") Then fso.DeleteFile logPath & ".1"
    fso.MoveFile logPath, logPath & ".1"
End Sub
