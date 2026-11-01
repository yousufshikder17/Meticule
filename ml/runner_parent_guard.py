"""Lightweight parent-death guard, started before any ML imports or GPU allocation."""
import os
import math
import threading
_started=False

def _windows_handles(expected,started):
    """Watch both the venv launcher and Node owner without adding a dependency to classical/TF/JAX runtimes."""
    import ctypes
    from ctypes import wintypes as w
    class Entry(ctypes.Structure):
        _fields_=[("dwSize",w.DWORD),("cntUsage",w.DWORD),("pid",w.DWORD),("heap",ctypes.c_size_t),
                  ("module",w.DWORD),("threads",w.DWORD),("parent",w.DWORD),("priority",w.LONG),
                  ("flags",w.DWORD),("name",w.WCHAR*260)]
    kernel=ctypes.WinDLL("kernel32",use_last_error=True)
    kernel.CreateToolhelp32Snapshot.argtypes=[w.DWORD,w.DWORD];kernel.CreateToolhelp32Snapshot.restype=w.HANDLE
    kernel.Process32FirstW.argtypes=[w.HANDLE,ctypes.POINTER(Entry)];kernel.Process32FirstW.restype=w.BOOL
    kernel.Process32NextW.argtypes=[w.HANDLE,ctypes.POINTER(Entry)];kernel.Process32NextW.restype=w.BOOL
    kernel.OpenProcess.argtypes=[w.DWORD,w.BOOL,w.DWORD];kernel.OpenProcess.restype=w.HANDLE
    kernel.GetProcessTimes.argtypes=[w.HANDLE,*([ctypes.POINTER(w.FILETIME)]*4)];kernel.GetProcessTimes.restype=w.BOOL
    kernel.WaitForMultipleObjects.argtypes=[w.DWORD,ctypes.POINTER(w.HANDLE),w.BOOL,w.DWORD];kernel.WaitForMultipleObjects.restype=w.DWORD
    kernel.CloseHandle.argtypes=[w.HANDLE];kernel.CloseHandle.restype=w.BOOL
    snapshot=kernel.CreateToolhelp32Snapshot(2,0)
    if snapshot in (None,ctypes.c_void_p(-1).value):raise ValueError("Parent process snapshot unavailable")
    parents={};entry=Entry();entry.dwSize=ctypes.sizeof(entry)
    try:
        more=kernel.Process32FirstW(snapshot,ctypes.byref(entry))
        while more:
            parents[entry.pid]=entry.parent;more=kernel.Process32NextW(snapshot,ctypes.byref(entry))
    finally:kernel.CloseHandle(snapshot)
    handles=[];current=os.getpid();found=False
    try:
        for _ in range(8):
            current=parents.get(current)
            if not current:break
            handle=kernel.OpenProcess(0x100000|0x1000,False,current)
            if not handle:raise ValueError("Parent process handle unavailable")
            handles.append(handle)
            if current==expected:
                creation,exit_time,kernel_time,user_time=(w.FILETIME() for _ in range(4))
                if not kernel.GetProcessTimes(handle,ctypes.byref(creation),ctypes.byref(exit_time),ctypes.byref(kernel_time),ctypes.byref(user_time)):
                    raise ValueError("Parent identity unavailable")
                born=((creation.dwHighDateTime<<32)|creation.dwLowDateTime)/10_000_000-11644473600
                if abs(born-started)>5:raise ValueError("Parent identity changed")
                found=True;break
        if not found:raise ValueError("Expected owner is not an ancestor")
        array=(w.HANDLE*len(handles))(*handles)
        if kernel.WaitForMultipleObjects(len(handles),array,False,0)!=258:raise ValueError("Parent already exited")
    except BaseException:
        for handle in handles:kernel.CloseHandle(handle)
        raise
    def watch():
        kernel.WaitForMultipleObjects(len(handles),array,False,0xffffffff)
        os._exit(1)
    return watch


def start_parent_guard():
    """Runner-owned environment only. Direct CLI calls without it keep their existing behavior."""
    global _started
    expected = os.environ.get("METICULE_RUNNER_PARENT_PID")
    if expected is None or _started:
        return
    _started=True
    try:
        expected=int(expected)
        started = float(os.environ["METICULE_RUNNER_PARENT_STARTED"])
        if expected<1 or not math.isfinite(started):raise ValueError("Invalid owner")
        if os.name=="nt":
            watch=_windows_handles(expected,started)
        else:
            if os.getppid()!=expected:raise ValueError("Parent already exited")
            def watch():
                wait=threading.Event()
                while os.getppid()==expected:wait.wait(.1)
                os._exit(1)
    except (ValueError, KeyError, OSError):
        os._exit(1)
    threading.Thread(target=watch, daemon=True, name="meticule-parent-guard").start()
