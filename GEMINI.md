# AI Assistant Guidelines

## Language
- **Final answers**: Chinese
- **Reasoning/other**: English

## Code Editing
- Narrow context scope; make incremental changes
- Validate syntax with Python compile check

## Environment
```bash
.venv/bin/python
```

## Dataset Schema
| Field | Description |
|-------|-------------|
| `edit_image` | Source/input image |
| `image` | Ground truth (target) |

## Dataset Directory
```
pico-banana-400k-subject_driven/openimages
```