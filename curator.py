import argparse
import collections
import glob
import logging
import os
import re
import shutil
import socket
import string
import subprocess

import packaging.version

from pytools.pyutils.io.file_system import get_absolute_path
from pytools.pyutils.io.pretty import load_json, loads_json
from pytools.pyutils.logging.logger import get_default_logger
from pytools.pyutils.misc.nested import AttrListDictifier, ListDictMerger
from pytools.pyutils.misc.decorator import cached_property
from pytools.pyutils.misc.string import row_pad_prefix

DEFAULT_CONFIG = AttrListDictifier().dictify({
    "dockers": {},
    "library": {
        "name": None,
        "arguments": {}
    },
    "runtime": {
        "name": None,
        "mounts": {},
        "devices": [],
        "networks": {},
        "environment": {},
        "attributes": {
            "use_gpus": False,
            "use_privileged": False,
            "use_user_permission": False,
            "use_hostname": False
        },
        "extra_args": [],
        "attach_entrypoint": "/bin/bash"
    },
    "preset": {
        "runtime": {
            "mounts": {},
            "networks": {}
        }
    }
})


class DockerCLI(object):
    DockerResult = collections.namedtuple("DockerResult", ['stdout', 'stderr', 'return_code'])

    def __init__(self, docker_executable, logger: logging.Logger):
        self._docker_executable = docker_executable
        self._logger = logger

    @classmethod
    def _expand_with_dict(cls, text: str, data: dict):
        text = string.Template(text)
        text = text.substitute(data)
        return text

    @property
    def logger(self):
        return self._logger

    @cached_property
    def version(self):
        result = self.execute("--version", use_stdout_pipe=True)
        version = re.findall(r"Docker version (\d+\.\d+\.\d+).*", result.stdout)
        if len(version) != 1:
            raise RuntimeError(f"Cannot parse docker version from \"{result.stdout}\"")
        return packaging.version.parse(version[0])

    def execute(self, *args, use_ignored_errors=False, use_dry_run=False, use_stdout_pipe=False):
        command = [self._docker_executable] + list(map(str, args))
        stringified_command = ' '.join(command)
        if use_dry_run:
            self._logger.debug(f"Execute (dry-run): \"{stringified_command}\"")
        else:
            self._logger.debug(f"Execute: \"{stringified_command}\"")

        if use_dry_run:
            return None
        pipe = subprocess.Popen(command, stdout=subprocess.PIPE if use_stdout_pipe else None, stderr=subprocess.PIPE)

        stdout, stderr = pipe.communicate()
        if stderr:
            stderr = stderr.decode('utf-8')
        if stdout:
            stdout = stdout.decode('utf-8')

        if not use_ignored_errors and pipe.returncode != 0:
            self._logger.critical(f"An error has occurred when executing docker command.\n"
                                  f"Command: {stringified_command}\n"
                                  f"Return code: {pipe.returncode}\n"
                                  f"Stderr:\n"
                                  f"{row_pad_prefix(stderr, '... ')}\n")
            raise RuntimeError("An error has occurred when executing docker command.")

        return self.DockerResult(stdout=stdout, stderr=stderr, return_code=pipe.returncode)

    def is_built(self, config):
        docker_images = self.execute("images", "--format", "{{ .Repository }}:{{ .Tag }}", use_stdout_pipe=True)
        docker_images = docker_images.stdout.split('\n')
        return config.library.name in docker_images or f"{config.library.name}:latest" in docker_images

    def get_runtime_status(self, config):
        docker_containers = self.execute(
            "ps", "-a", "--format", "{{ .Names }}|{{ .Image }}|{{ .Status }}",
            use_stdout_pipe=True
        )
        docker_containers = list(info.split('|') for info in docker_containers.stdout.split('\n'))
        for container in docker_containers:
            if container[0] == config.runtime.name:
                if container[1] == config.library.name:
                    return container[2]
                self._logger.warning(f"Find container named \"{config.runtime.name}\" "
                                     f"which is not built with library \"{config.library.name}\"")
        return None

    def get_library_variables(self, config):
        result = dict()
        info = loads_json(self.execute("image", "inspect", config.library.name, use_stdout_pipe=True).stdout)
        for environ in info[0]['Config']['Env']:
            key, *value = environ.split('=')
            result.setdefault(key, '='.join(value))
        return result

    def build(self, config):
        args = list()
        for key, value in config.library.arguments.items():
            if value is None:
                value = input(f"Please input the build argument {key}: ")
            elif value.startswith("$%"):
                value = value[2:]
                if value == "EUID":
                    value = os.geteuid()
                elif value == "EGID":
                    value = os.getegid()
                elif value == "UID":
                    value = os.getuid()
                elif value == "GID":
                    value = os.getgid()
                else:
                    raise ValueError(f"Unrecognized builtin variable \"{value}\"")
            elif value.startswith("$"):
                value = os.environ[value[1:]]
            args.extend(["--build-arg", f"{key}={value}"])
        if getattr(config, "force", False):
            args.append('--no-cache')
        self.execute("build", "-t", config.library.name, config.path, *args, use_dry_run=config.dry_run)

    def start(self, config):
        if not self.is_built(config):
            self.build(config)

        if self.get_runtime_status(config) is not None:
            self.clean(config)

        args = []

        # Handling volume mounts
        if config.runtime.mounts:
            runtime_variables = self.get_library_variables(config)
            for name, trials in config.runtime.mounts.items():
                if trials is None:
                    trials = config.preset.runtime.mounts[name]

                is_any_trial_successful = False
                for trial in trials:
                    is_trial_failed = False
                    for option in trial:
                        option.source = os.path.expanduser(os.path.expandvars(option.source))
                        option.source = get_absolute_path(option.source, use_real_path=True)
                        if not os.path.exists(option.source):
                            is_trial_failed = True
                            break

                    if not is_trial_failed:
                        is_any_trial_successful = True
                        for option in trial:
                            option.target = self._expand_with_dict(option.target, runtime_variables)

                            args.extend(["-v", f"{option.source}:{option.target}:{option.mode}"])
                        break

                if not is_any_trial_successful:
                    self._logger.warning(f"All trials on mounting collection \"{name}\" are failed")

        # Handling network mappings
        if config.runtime.networks == "host":
            args.extend(["--net", "host"])
        else:
            if isinstance(config.runtime.networks, str):
                config.runtime.networks = config.preset.runtime.networks[config.runtime.networks]

            if isinstance(config.runtime.networks, dict):
                for k, v in config.runtime.networks.items():
                    args.extend(['-p', f"{k}:{v}"])
            else:
                raise ValueError("Runtime network setting cannot be analyzed")

        # Handling device pass-through
        if isinstance(config.runtime.devices, str):
            config.runtime.devices = config.preset.runtime.devices[config.runtime.devices]
        if config.runtime.devices:
            devices = []
            for device_pattern in config.runtime.devices:
                devices.extend(glob.glob(device_pattern))
            for device in devices:
                self._logger.debug(f"Find device: {device}")
                args.extend(["--device", device])

        # Handling environment
        if config.runtime.environment:
            for key, value in config.runtime.environment.items():
                if value is None:
                    args.extend(["--env", str(key)])
                else:
                    args.append(["--env", f"{key}={value}"])

        # Handling other attributes
        if config.runtime.attributes.use_gpus:
            args.extend(["--gpus", "all"])
        if config.runtime.attributes.use_privileged:
            args.append("--privileged")
        if config.runtime.attributes.use_user_permission:
            args.extend(["-u", f"{os.geteuid()}:{os.getegid()}"])
        if config.runtime.attributes.use_hostname:
            args.extend(["--hostname", socket.gethostname()])

        args.extend(config.runtime.extra_args)

        self.execute(
            "run", "-td", "--name", config.runtime.name, *args,
            config.library.name, use_dry_run=config.dry_run
        )

    def clean(self, config):
        if getattr(config, "with_images", False) and self.is_built(config):
            self.execute("rmi", config.library.name, use_dry_run=config.dry_run)

        runtime_status = self.get_runtime_status(config)
        if runtime_status is not None:
            if runtime_status.startswith("Up"):
                self.execute("stop", config.runtime.name, use_dry_run=config.dry_run)
            self.execute("rm", config.runtime.name, use_dry_run=config.dry_run)

    def attach(self, config):
        if not self.is_built(config):
            self.build(config)

        runtime_status = self.get_runtime_status(config)
        if runtime_status is None:
            self.start(config)

        self.execute("exec", "-it", config.runtime.name, config.runtime.attach_entrypoint)


def get_config(args):
    dictifier = AttrListDictifier()
    dict_merger = ListDictMerger()
    config = DEFAULT_CONFIG
    config = dict_merger.merge(config, vars(args))
    if os.path.isfile("defaults.json"):
        config = dict_merger.merge(config, load_json("defaults.json"))
    config = dict_merger.merge(config, load_json(os.path.join(args.path, "config.json")))
    return dictifier.dictify(config)


def get_docker(config, logger):
    for name, option in config.dockers.items():
        if shutil.which(name) is not None:
            docker = DockerCLI(name, logger)

            if not config.runtime.attributes.use_gpus:
                return docker
            if isinstance(option.has_gpus_support, list):
                if len(option.has_gpus_support) != 3:
                    raise ValueError("has_gpus_support should be of the form [value, predicate, value]")
                if option.has_gpus_support[0] == "docker_version":
                    option.has_gpus_support[0] = docker.version
                else:
                    raise ValueError(f"Cannot interpret variable \"{option.has_gpus_support[0]}\"")
                option.has_gpus_support[2] = packaging.version.parse(option.has_gpus_support[2])
                option.has_gpus_support[1] = option.has_gpus_support[1].lower()
                if option.has_gpus_support[1] in (">", "larger"):
                    option.has_gpus_support = (option.has_gpus_support[0] > option.has_gpus_support[2])
                elif option.has_gpus_support[1] in ("<", "less"):
                    option.has_gpus_support = (option.has_gpus_support[0] < option.has_gpus_support[2])
                elif option.has_gpus_support[1] in ("<=", "less_equal"):
                    option.has_gpus_support = (option.has_gpus_support[0] <= option.has_gpus_support[2])
                elif option.has_gpus_support[1] in (">=", "larger_equal"):
                    option.has_gpus_support = (option.has_gpus_support[0] >= option.has_gpus_support[2])
                elif option.has_gpus_support[1] in ("==", "equal"):
                    option.has_gpus_support = (option.has_gpus_support[0] == option.has_gpus_support[2])
                elif option.has_gpus_support[1] in ('!=', "<>", "not_equal"):
                    option.has_gpus_support = (option.has_gpus_support[0] != option.has_gpus_support[2])
                else:
                    raise ValueError(f"Unrecognized predicate \"{option.has_gpus_support[1]}\"")

            if option.has_gpus_support:
                return docker

    raise RuntimeError("Unable to find any valid docker")


def main(args):
    logger = get_default_logger("Curator", logger_level="DEBUG" if args.verbose else "INFO")

    if args.action == "build":
        config = get_config(args)
        docker = get_docker(config, logger)
        docker.build(config)
    elif args.action == "start":
        config = get_config(args)
        docker = get_docker(config, logger)
        docker.start(config)
    elif args.action == "clean":
        config = get_config(args)
        docker = get_docker(config, logger)
        docker.clean(config)
    elif args.action == "attach":
        config = get_config(args)
        docker = get_docker(config, logger)
        docker.attach(config)
    else:
        raise RuntimeError(f"Unrecognized action \"{args.action}\"")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--verbose", action="store_true", help="make logger be more verbose")

    subparsers = parser.add_subparsers(dest="action", description="curator's action to manage library")

    subparser = subparsers.add_parser("build", description="build a docker library")
    subparser.add_argument("path", type=str, help="path to the docker library to be built")
    subparser.add_argument("-f", "--force", action="store_true", help="force to build and ignore cache")
    subparser.add_argument("--dry_run", action="store_true", help="only output docker commands")
    subparser.add_argument("--verbose", action="store_true", help="make logger be more verbose")

    subparser = subparsers.add_parser("start", description="start a docker runtime from library")
    subparser.add_argument("path", type=str, help="path to the docker library to be started")
    subparser.add_argument("--dry_run", action="store_true", help="only output docker commands (ignore -b)")
    subparser.add_argument("--verbose", action="store_true", help="make logger be more verbose")

    subparser = subparsers.add_parser("clean", description="start a docker runtime from library")
    subparser.add_argument("path", type=str, help="path to the docker library for cleaning docker runtime")
    subparser.add_argument("-i", "--with_images", action="store_true", help="clean the docker images")
    subparser.add_argument("--dry_run", action="store_true", help="only output docker commands")
    subparser.add_argument("--verbose", action="store_true", help="make logger be more verbose")

    subparser = subparsers.add_parser("attach", description="attach to a docker runtime from library")
    subparser.add_argument("path", type=str, help="path to the docker library for attaching docker runtime")
    subparser.add_argument("--dry_run", action="store_true", help="only output docker commands")
    subparser.add_argument("--verbose", action="store_true", help="make logger be more verbose")

    main(parser.parse_args())
