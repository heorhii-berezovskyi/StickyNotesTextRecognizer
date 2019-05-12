import os


def get_file_name(path: str) -> str:
    name, extension = os.path.splitext(os.path.basename(path))
    return name


def get_list_of_files(folder: str) -> list:
    return [os.path.join(folder, f) for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))]
