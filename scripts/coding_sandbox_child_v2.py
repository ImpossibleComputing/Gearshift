#!/usr/bin/env python3
"""Version2 child: unchanged confinement, explicit bounded output/serialization."""
import ctypes, errno, io, json, os, resource, sys, sysconfig

class OutputLimitError(RuntimeError):
    pass

class BoundedDiscard(io.TextIOBase):
    def __init__(self, limit): self.limit=limit; self.used=0; self.buffer=self
    def write(self, value):
        n=len(value.encode("utf-8")) if isinstance(value,str) else len(value)
        self.used += n
        if self.used>self.limit: raise OutputLimitError("discarded function stdout limit")
        return len(value)
    def flush(self): pass

def send_status(fd, value):
    os.write(fd,(json.dumps(value)+"\n").encode())


def main():
    # The root-owned launch payload is closed before confinement. Descriptor 0
    # carries only the actual test input, so open(0) and os.read(0) work normally.
    with open(sys.argv[1]) as payload:p=json.load(payload)
    policy=p['policy']; status_fd=p['status_fd']
    if p.get('cpu_id') is not None: os.sched_setaffinity(0,{p['cpu_id']})
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
    resource.setrlimit(resource.RLIMIT_CPU,(policy['cpu_seconds'],policy['cpu_seconds']+1))
    for r,lim in [(resource.RLIMIT_AS,policy['address_space_bytes']),(resource.RLIMIT_FSIZE,policy['stdout_limit_bytes']),(resource.RLIMIT_NOFILE,32),(resource.RLIMIT_NPROC,0),(resource.RLIMIT_CORE,0)]:resource.setrlimit(r,(lim,lim))
    os.chdir(p['root']);os.chroot('.');os.chdir('/')
    os.setgroups([]);os.setgid(65534);os.setuid(65534)
    os.environ.clear();sys.path[:]=[stdlib,stdlib+'/lib-dynload']
    os.closerange(3,status_fd);os.closerange(status_fd+1,1024)
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
    send_status(status_fd,{'ready':True,'uid':os.geteuid(),'cpu_affinity':sorted(os.sched_getaffinity(0)),
        'confinement':'chroot_uid_seccomp','stdout_limit_bytes':policy['stdout_limit_bytes']})
    os.close(status_fd)
    if p['fn_name']:
        with __import__('contextlib').redirect_stdout(BoundedDiscard(policy['discarded_function_stdout_limit_bytes'])):
            exec(compile(p['code'],'candidate.py','exec'),ns)
            obj=ns['Solution']() if 'Solution' in ns else None
            fn=getattr(obj,p['fn_name']) if obj is not None else ns[p['fn_name']]
            result=fn(*args)
        used=0
        for chunk in json.JSONEncoder().iterencode({'result':result}):
            used+=len(chunk.encode('utf-8'))
            if used+1>policy['stdout_limit_bytes']: raise OutputLimitError('serialized function result limit')
            sys.stdout.write(chunk)
        sys.stdout.write('\n');sys.stdout.flush()
    else:
        exec(compile(p['code'],'candidate.py','exec'),ns)

if __name__=='__main__':
    try: main()
    except BaseException as exc:
        # A setup failure leaves status_fd open; after readiness it is closed.
        # Candidate code cannot emit a trusted status message.
        try:
            with open(sys.argv[1]) as f: payload=json.load(f)
            send_status(payload['status_fd'],{'ready':False,'error':type(exc).__name__,'detail':str(exc)[:1000]})
        except BaseException: pass
        raise
