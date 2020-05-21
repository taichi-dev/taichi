import os


def clear_directory_with_suffix(directory, suffix):
    files = os.listdir(directory)
    assert suffix[0] != '.', "No '.' needed."
    for f in files:
        if f.endswith('.' + suffix):
            os.remove(os.path.join(directory, f))
