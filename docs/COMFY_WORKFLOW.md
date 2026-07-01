# Comfy Workflow

The ComfyUI adapter is a wrapper around Atlas Camera core behavior. The core
package must not depend on ComfyUI.

Initial planned nodes:

- Atlas Load Image / Solve Camera
- Atlas Export Review Package
- Atlas Export Solve JSON
- Atlas Export Maya Review Scene
- Atlas USD Camera Loader

The current node module provides scaffolds and direct wrappers for metadata-only
solve creation, JSON export, review package export, Maya review scene export,
and USD camera loading.

## Maya2Comfy Relationship

Maya2Comfy proved that camera estimation, USD camera loading, Kornia conversion,
JSON export, generated Maya validation scenes, Docker testing, and Comfy nodes
can work. Atlas Camera keeps those ideas but moves the shared model into a
DCC-agnostic core.

The next migration step is to move stable vanishing-point detection and camera
conversion logic into `atlas_camera.core`, then keep Comfy-specific node code in
`atlas_camera.comfy`.

