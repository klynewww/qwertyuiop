"""Процесс агента. Его запускает хост (env/isolated.py), руками не нужно.

Общение с хостом только по строкам JSON:
  хост -> агент (stdin):  наблюдение (объект)
  агент -> хост (stdout): {"ready": true} при старте, затем {"action": {...}}
                          на каждое наблюдение; при падении политики {"error": "..."}.

Здесь нет ни среды, ни оценщика, ни truth: только политика. Перед стартом
политики включается «песочница» (audit hook): запрещены чтение папок из
--deny (hidden/), запуск процессов, сеть, ctypes. Это защита уровня
Python; настоящую границу даёт Docker.
"""
import importlib
import json
import os
import sys

MAX_LINE = 1_000_000
_BLOCKED_EVENTS = {
    "subprocess.Popen", "os.system", "os.exec", "os.posix_spawn", "os.spawn",
    "os.fork", "os.forkpty", "ctypes.dlopen", "ctypes.dlsym", "ctypes.cdata",
    "os.symlink", "os.link", "socket.connect", "socket.bind", "socket.getaddrinfo",
    "socket.gethostbyname", "socket.sendto",
}


def _install_sandbox(deny_paths):
    denied = [os.path.realpath(p) for p in deny_paths]

    def inside(path) -> bool:
        if isinstance(path, int):
            return False
        try:
            real = os.path.realpath(os.fsdecode(path if path is not None else "."))
        except Exception:
            return False
        return any(real == d or real.startswith(d + os.sep) for d in denied)

    def hook(event, args):
        if event in _BLOCKED_EVENTS:
            raise PermissionError("запрещено песочницей")
        if event in ("open", "os.listdir", "os.scandir", "os.chdir", "os.chmod",
                     "os.rename", "os.remove", "os.rmdir", "os.truncate") and args:
            if inside(args[0]):
                raise PermissionError("запрещено песочницей")

    sys.addaudithook(hook)


def _send(out, obj):
    out.write(json.dumps(obj, ensure_ascii=False) + "\n")
    out.flush()


def main():
    root, spec, kwargs_json = sys.argv[1:4]
    deny = sys.argv[4:]
    sys.path.insert(0, root)

    # канал протокола: копия исходного stdout; обычный print уходит в stderr
    out = os.fdopen(os.dup(1), "w", encoding="utf-8")
    devnull = os.open(os.devnull, os.O_WRONLY)
    os.dup2(devnull, 1)
    sys.stdout = sys.stderr

    module_name, class_name = spec.split(":")
    policy = getattr(importlib.import_module(module_name), class_name)(**json.loads(kwargs_json))
    _install_sandbox(deny)
    _send(out, {"ready": True})

    while True:
        line = sys.stdin.readline(MAX_LINE + 1)
        if not line:
            return
        try:
            obs = json.loads(line)
            action = policy.act(obs)
            _send(out, {"action": action})
        except BaseException as e:  # noqa: BLE001 - любое падение политики сообщаем хосту
            try:
                _send(out, {"error": type(e).__name__})
            finally:
                os._exit(1)


if __name__ == "__main__":
    main()
