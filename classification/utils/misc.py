import time


def get_time_str():
    return time.strftime("%Y%m%d_%H%M%S", time.localtime())


def mkdir_unique(parent_dir, dir_template):
    """create a uniquely named directory in the parent_dir, and return a path to the created directory.
    Returns the unique path that was created.

    Args:
            parent_dir: directory in which to make this new directory
            dir_template: name of the directory to create, but with curly braces somewhere,
                            indicating where to insert the uniquifying string.
                            e.g. 'foobar{}'
    """
    path = parent_dir / dir_template.format("")
    version_number = 0
    while path.exists():
        version_number += 1
        uniquifying_string = f"_v{version_number}"
        path = parent_dir / dir_template.format(uniquifying_string)
    path.mkdir(parents=True)
    return path


def mkdir_and_rename(path):
    """mkdirs. If path exists, rename it with timestamp and create a new one.

    Args:
        path (pathlib.Path): Folder path.
    """
    if path.exists():
        new_name = path.with_stem(path.stem + "_archived_" + get_time_str())
        print(f"Path already exists. Rename it to {new_name}", flush=True)
        path.rename(new_name)
    path.mkdir(exist_ok=True)
