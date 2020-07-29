# Configuration

There are two types of the configuration, namely the global configuration and the per-image configuration. The global configuration is mainly used to specify machine-related configurations, such as docker executables. The image configuration contains other options, including whether to enable GPU supports, privileged dockers, handling user permissions, and so on. 

Though we have stated that the global configuration is machine-related, you can, however, choose to offload the options to the image configuration and not to use global configuration at all. However, you will have to manually twist the configuration for each machine and for each image.

The global configuration is located at `defaults.json` (an example is shown in `defaults-example.json`). The image is located at the same directory of `Dockerfile` but named `config.json`.      

## Options

### Docker Executables

Specifying docker executables, as shown in the example, are like:

```json
{
  "dockers": {
    "docker": {
      "has_gpus_support": ["docker_version", ">=", "19.03"]
    },
    "nvidia-docker2": {
      "has_gpus_support": true
    },
    "nvidia-docker": {
      "has_gpus_support": true
    }
  }
}
```

The key (`docker`, `nvidia-docker2`, `nvidia-docker`) is the path to the docker executable. The option `has_gpus_support` specifies whether the docker has the GPU support. You could just write `true` or `false` to say that docker has/has no GPU supports or use a simple predicate to automatically determine whether that docker has GPU supports. 

### Library Options

The library options contain the essential information to build a docker image:

```json
{
  "library": {
    "name": "<docker image name>",
    "arguments": {
      "<name>": "<value | environ | builtin values | null>" 
    }
  }
}
```

For arguments:

- If an argument is specified as `null` (such as passwords), it will be asked interactively when you try to build the docker. 
- Currently, the builtin values are `$%EUID` (user effective id), `$%EGID` (group effective id), `$%UID` (user id), and `$%GID` (group id).
- An environ value should be started with `$` (for example, `$USER`, `$PWD`). 

### Runtime Options

The runtime options contain the information to start a runtime docker container:

```json
{
  "runtime": {
    "name": "<docker container name>",
    "mounts": {
      "<mount name>": [
        [
          {
            "source": "<path on the host machine for trial 1>",
            "target": "<path on the container for trial 1>",
            "mode": "<ro | rw>"
          },
          {
            "source": "<path on the host machine for trial 1>",
            "target": "<path on the container for trial 1>",
            "mode": "<ro | rw>"
          }
        ],
        [          
          {
            "source": "<path on the host machine for trial 2>",
            "target": "<path on the container for trial 2>",
            "mode": "<ro | rw>"
          }
        ]
      ],
      "<preset mount name>": null
    },
    "networks": {
      "<port on the host machine>": "<port on the container>"
    },
    "attributes": {
      "use_gpus": "<true | false>",
      "use_privileged": "<true | false>",
      "use_user_permission": "<true | false>",
      "use_hostname": "<true | false>"
    },
    "extra_args": [],
    "attach_entrypoint": "<command to be executed when you attach to the container>"
  }
}
```

### Preset Values

Oftentimes, we have different place on different machine to store the same content. For example, one may store ImageNet at `/mnt/dataset/imagenet` on machine A while on machine B the ImageNet is located at `/data/dataset/imagenet`. These options should be considered as machine-related and we provides a way to specify them as preset values.
