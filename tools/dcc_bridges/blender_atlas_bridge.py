bl_info = {
    "name": "Atlas Camera Bridge",
    "blender": (4, 0, 0),
    "category": "Import-Export",
    "description": "Import the latest Atlas Camera USD export.",
}

import bpy
import os
import glob

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

class ATLAS_OT_import_latest(bpy.types.Operator):
    bl_idname = "atlas.import_latest"
    bl_label = "Import Latest USD"
    bl_description = "Find and import the most recently generated USD file in the Atlas output directory"

    def execute(self, context):
        prefs = context.preferences.addons[__name__].preferences
        output_dir = prefs.atlas_output_dir

        if not output_dir or not os.path.isdir(output_dir):
            self.report({'ERROR'}, "Please configure a valid Atlas output directory in Addon Preferences.")
            return {'CANCELLED'}

        latest = get_latest_file(output_dir, ['.usd', '.usda', '.usdc'])
        if latest:
            bpy.ops.wm.usd_import(filepath=latest)
            self.report({'INFO'}, f"Imported: {latest}")
            return {'FINISHED'}
        else:
            self.report({'WARNING'}, f"No USD files found in {output_dir}")
            return {'CANCELLED'}

class AtlasBridgePreferences(bpy.types.AddonPreferences):
    bl_idname = __name__

    atlas_output_dir: bpy.props.StringProperty(
        name="Atlas Output Directory",
        description="Path to the ComfyUI atlas_exports directory",
        subtype='DIR_PATH',
    )

    def draw(self, context):
        layout = self.layout
        layout.prop(self, "atlas_output_dir")

class ATLAS_PT_bridge_panel(bpy.types.Panel):
    bl_label = "Atlas Bridge"
    bl_idname = "ATLAS_PT_bridge_panel"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'Atlas'

    def draw(self, context):
        layout = self.layout
        prefs = context.preferences.addons[__name__].preferences

        layout.prop(prefs, "atlas_output_dir")
        layout.separator()
        layout.operator("atlas.import_latest", text="Import Latest Export", icon='IMPORT')

classes = (
    ATLAS_OT_import_latest,
    AtlasBridgePreferences,
    ATLAS_PT_bridge_panel,
)

def register():
    for cls in classes:
        bpy.utils.register_class(cls)

def unregister():
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)

if __name__ == "__main__":
    register()
