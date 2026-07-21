import os
import glob
import maya.cmds as cmds

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
    output_dir = cmds.optionVar(q="atlas_output_dir") if cmds.optionVar(exists="atlas_output_dir") else ""

    if not output_dir or not os.path.isdir(output_dir):
        prompt = cmds.fileDialog2(fileMode=3, dialogStyle=2, caption="Select Atlas Camera Output Directory (e.g. atlas_exports)")
        if prompt:
            output_dir = prompt[0]
            cmds.optionVar(stringValue=("atlas_output_dir", output_dir))
        else:
            cmds.warning("No directory selected.")
            return

    latest = get_latest_file(output_dir, ['.ma', '.usd', '.usda'])
    if latest:
        cmds.file(latest, i=True)
        print(f"Imported: {latest}")
    else:
        cmds.warning(f"No .ma or .usd files found in {output_dir}")

def show_ui():
    window_name = "AtlasBridgeUI"
    if cmds.window(window_name, exists=True):
        cmds.deleteUI(window_name)

    window = cmds.window(window_name, title="Atlas Bridge", widthHeight=(300, 100))
    cmds.columnLayout(adjustableColumn=True)
    cmds.text(label="Atlas Camera Bridge", align="center", font="boldLabelFont", height=30)
    cmds.button(label="Import Latest Export", command=lambda x: import_latest_atlas(), height=40)
    cmds.button(label="Set Output Directory", command=lambda x: cmds.optionVar(remove="atlas_output_dir") or import_latest_atlas())
    cmds.showWindow(window)

# If run directly from script editor
if __name__ == "__main__":
    show_ui()
