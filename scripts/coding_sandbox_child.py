#!/usr/bin/env python3
"""Child only. No candidate bytecode runs before chroot, UID drop and seccomp."""
import ctypes, errno, io, json, os, resource, sys, sysconfig

def main():
    # The root-owned launch payload is closed before confinement. Descriptor 0
    # carries only the actual test input, so open(0) and os.read(0) work normally.
    with open(sys.argv[1]) as payload:p=json.load(payload)
    stdlib=sysconfig.get_path('stdlib')
    # Resolve the kernel filter library before entering the minimal read-only root.
    lib=ctypes.CDLL('libseccomp.so.2',use_errno=True)
    lib.seccomp_init.argtypes=[ctypes.c_uint32];lib.seccomp_init.restype=ctypes.c_void_p
    lib.seccomp_syscall_resolve_name.argtypes=[ctypes.c_char_p];lib.seccomp_syscall_resolve_name.restype=ctypes.c_int
    lib.seccomp_rule_add.argtypes=[ctypes.c_void_p,ctypes.c_uint32,ctypes.c_int,ctypes.c_uint]
    lib.seccomp_load.argtypes=[ctypes.c_void_p];lib.seccomp_release.argtypes=[ctypes.c_void_p]
    ctx=lib.seccomp_init(0x00050000|errno.EPERM)
    if not ctx:raise RuntimeError('seccomp_init failed')
    # No networking, process/thread creation, exec, signals to other processes,
    # ptrace, namespace operations, device ioctls, filesystem mutations or IPC.
    allowed='read write close fstat newfstatat stat lstat statx lseek pread64 readlink readlinkat access faccessat getdents64 mmap mprotect munmap mremap brk madvise rt_sigaction rt_sigprocmask rt_sigreturn sigaltstack futex clock_gettime clock_getres gettimeofday time nanosleep clock_nanosleep getrandom getpid getppid gettid getuid geteuid getgid getegid uname getcwd sched_yield sched_getaffinity getrusage prlimit64 exit exit_group fcntl dup dup2 dup3 poll ppoll select pselect6'.split()
    for name in allowed:
        n=lib.seccomp_syscall_resolve_name(name.encode())
        if n>=0 and lib.seccomp_rule_add(ctx,0x7fff0000,n,0)!=0:raise RuntimeError('seccomp rule failed')
    # open/openat: allow only read-only flags (never create/truncate/write).
    class Cmp(ctypes.Structure):_fields_=[('arg',ctypes.c_uint),('op',ctypes.c_uint),('a',ctypes.c_uint64),('b',ctypes.c_uint64)]
    lib.seccomp_rule_add_array.argtypes=[ctypes.c_void_p,ctypes.c_uint32,ctypes.c_int,ctypes.c_uint,ctypes.POINTER(Cmp)]
    mask=os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC|os.O_APPEND|(os.O_TMPFILE & ~os.O_DIRECTORY)
    # Read-only directory enumeration is required by Python's import machinery.
    for name,arg in [('open',1),('openat',2)]:
        n=lib.seccomp_syscall_resolve_name(name.encode())
        if n>=0 and lib.seccomp_rule_add_array(ctx,0x7fff0000,n,1,ctypes.byref(Cmp(arg,7,mask,0)))!=0:raise RuntimeError('open rule failed')
    resource.setrlimit(resource.RLIMIT_CPU,(p['seconds'],p['seconds']+1))
    for r,lim in [(resource.RLIMIT_AS,4*1024**3),(resource.RLIMIT_FSIZE,2*1024**2),(resource.RLIMIT_NOFILE,32),(resource.RLIMIT_NPROC,0),(resource.RLIMIT_CORE,0)]:resource.setrlimit(r,(lim,lim))
    os.chdir(p['root']);os.chroot('.');os.chdir('/')
    os.setgroups([]);os.setgid(65534);os.setuid(65534)
    os.environ.clear();sys.path[:]=[stdlib,stdlib+'/lib-dynload']
    os.closerange(3,1024)
    if lib.seccomp_load(ctx)!=0:raise RuntimeError('seccomp_load failed')
    lib.seccomp_release(ctx)
    assert os.geteuid()==65534
    # No checker, expected output, model state, credential or host file is present here.
    ns={'__name__':'__main__'}
    sys.setrecursionlimit(10000)
    if p['fn_name']:
        # Interface support does not repair candidate code. Common type names are
        # preloaded in the same way as the official call-based harness.
        exec('from typing import *\nfrom collections import *\nimport math, heapq, bisect, itertools, functools, collections\n',ns)
        args=[json.loads(line) for line in p['input'].split('\n')]
        with __import__('contextlib').redirect_stdout(io.StringIO()):
            exec(compile(p['code'],'candidate.py','exec'),ns)
            obj=ns['Solution']() if 'Solution' in ns else None
            fn=getattr(obj,p['fn_name']) if obj is not None else ns[p['fn_name']]
            result=fn(*args)
        print(json.dumps({'result':result},allow_nan=False))
    else:
        exec(compile(p['code'],'candidate.py','exec'),ns)

if __name__=='__main__':main()
