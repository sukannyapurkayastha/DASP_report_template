import os


def get_project_root(current_file: str = __file__, project_name: str = "DASP_report_template") -> str:
    """
    Traverse upwards from the current file until the project root is found.

    :param current_file: The path of the current file (__file__).
    :param project_name: The name of the project root directory.
    :returns: The absolute path to the project root directory.
    """
    path = os.path.abspath(current_file)
    while True:
        path, folder = os.path.split(path)
        if folder == project_name:
            return os.path.join(path, folder)
        if folder == '':
            # Reached the root of the filesystem without finding the project directory
            raise Exception(f"Project root '{project_name}' not found in path.")
