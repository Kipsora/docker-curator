from pytools.legacy.pytorch import curator


if __name__ == "__main__":
    for name in curator.registered_registry_names():
        print("Module class:", name)
        for key in curator.registered_registry_keys(name):
            print("  Key:", key)
