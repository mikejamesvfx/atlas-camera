import os
import glob
import nuke

def get_latest_file(directory, extensions):
    latest_file = None
    latest_time = 0
    for ext in extensions:
        for file_path in glob.glob(os.path.join(directory, f"*{ext}")):
            mtime = os.path.getmtime(file_path)
            if mtime > latest_time:
                latest_time = mtime
                latest_file = file_path
    return latest_file

def import_latest_atlas():
    # Store settings in ~/.nuke/atlas_bridge_config.txt (simple)
    config_path = os.path.join(os.path.expanduser("~"), ".nuke", "atlas_bridge_config.txt")
    output_dir = ""

    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            output_dir = f.read().strip()

    if not output_dir or not os.path.isdir(output_dir):
        output_dir = nuke.getFilename("Select Atlas Camera Output Directory (e.g. atlas_exports)")
        if output_dir:
            # getFilename sometimes appends filenames, ensure directory
            if os.path.isfile(output_dir):
                output_dir = os.path.dirname(output_dir)
            with open(config_path, "w") as f:
                f.write(output_dir)
        else:
            return

    latest = get_latest_file(output_dir, ['.nk'])
    if latest:
        nuke.nodePaste(latest)
        print(f"Imported: {latest}")
    else:
        nuke.message(f"No .nk files found in {output_dir}")

def add_menu():
    menubar = nuke.menu("Nuke")
    atlas_menu = menubar.addMenu("Atlas Camera")
    atlas_menu.addCommand("Import Latest Export", import_latest_atlas, "Ctrl+Shift+A")

# Add menu automatically when loaded via menu.py
add_menu()
