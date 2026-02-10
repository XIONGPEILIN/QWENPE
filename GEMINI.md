# AI Assistant Guidelines

## Language
- **Final answers**: Chinese

## Code Editing
- Narrow context scope; make incremental changes
- Validate syntax with Python compile check

## Environment
```bash
.venv/bin/python
```
python use .venv 


## Dataset Schema
| Field | Description |
|-------|-------------|
| `edit_image` | Source/input image |
| `image` | Ground truth (target) |

## Dataset Directory
```
pico-banana-400k-subject_driven/openimages
```