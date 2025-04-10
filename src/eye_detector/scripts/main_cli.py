from sys import argv
from subprocess import run


def main():
    return run(
        [
            "/usr/bin/env",
            "python3",
            "-m",
            "eye_detector.scripts." + argv[1],
            *argv[2:],
        ]
    ).returncode
