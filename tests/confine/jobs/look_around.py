"""Tries to reach things a confined job must not reach, and reports as its one
output, in JSON, what happened. It reads no record.

Arguments: the path of a file that exists outside the sandbox, then the address
and the port of a listener outside the sandbox."""

import json
import os
import socket
import sys

import waterwhite_job

outside_file, host, port = sys.argv[1], sys.argv[2], int(sys.argv[3])


def attempt(action):
    try:
        return {"ok": True, "value": action()}
    except OSError as error:
        return {"ok": False, "error": type(error).__name__, "errno": error.errno}


def read_file():
    with open(outside_file, encoding="utf-8") as handle:
        return handle.read()


def connect(address):
    with socket.create_connection(address, timeout=2):
        return "connected"


def write_file(path):
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("x")
    return os.path.abspath(path)


def new_user_namespace():
    os.unshare(os.CLONE_NEWUSER)
    return "made"


def set_hostname():
    socket.sethostname("renamed")
    return "renamed"


def open_descriptors():
    found = []
    for number in range(256):
        try:
            os.fstat(number)
        except OSError:
            continue
        found.append(number)
    return found


report = {
    "file": attempt(read_file),
    "file_exists": os.path.exists(outside_file),
    "folder_exists": os.path.exists(os.path.dirname(outside_file)),
    "root": sorted(os.listdir("/")),
    "connect": attempt(lambda: connect((host, port))),
    "connect_elsewhere": attempt(lambda: connect(("192.0.2.1", 80))),
    "resolve": attempt(lambda: len(socket.getaddrinfo("example.org", 80))),
    "interfaces": [name for _, name in socket.if_nameindex()],
    "environment": sorted(os.environ),
    "write_scratch": attempt(lambda: write_file("scratch.txt")),
    "write_root": attempt(lambda: write_file("/outside.txt")),
    "new_user_namespace": attempt(new_user_namespace),
    "set_hostname": attempt(set_hostname),
    "descriptors": open_descriptors(),
    "pipes": sorted(int(part) for part in os.environ[waterwhite_job.ENV_FDS].split(",")),
}

with waterwhite_job.connect() as job:
    job.emit(json.dumps(report, sort_keys=True), used=[])
