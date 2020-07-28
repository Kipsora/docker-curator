# docker-curator
A Curator for Managing Docker Library.

Many lab clusters are managed by `docker` for sharing resources. However, sometimes we may find that using `docker` can be annoying since we need to figure out which is the volume/port to be mounted, whether we need to use privileged containers, which `docker` version to use (`docker` has nvidia GPU supports only after `19.03`), and so on. 

This project aims at providing a united interface for `docker` container configuration. So you can configure once, and deploy everywhere.     

## Configurations

Please refer to [configurations](CONFIGURATION.md).

## Commands

To build a library:

```shell script
python3 curator.py build /path/to/the/library
```

To start a container from a library:

```shell script
python3 curator.py start /path/to/the/library
```

To attach to a library:

```shell script
python3 curator.py attach /path/to/the/library
```

To clean the existing containers:

```shell script
python3 curator.py clean /path/to/the/library
```

## License

The code is released under the [MIT License](LICENSE).
