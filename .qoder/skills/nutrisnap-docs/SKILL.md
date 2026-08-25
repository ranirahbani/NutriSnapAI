---
name: nutrisnap-docs
description: Update NutriSnap AI README.md with documentation images from docs/images/. Integrates screenshots (upload_tab.png, dashboard_tab.png) into the Screenshots section and architecture diagram (architecture_flow.png) into the Architecture section. Keeps file structure listing complete. Use when updating README images, adding new screenshots, or syncing docs/images/ references.
---

# NutriSnap Docs Image Integration

## Overview

This skill manages image references in the NutriSnap AI README.md, ensuring documentation images from `docs/images/` are properly embedded in the correct sections.

## Available Images

| File | Section | Description |
|------|---------|-------------|
| `docs/images/upload_tab.png` | Screenshots & UI Description | Upload & Analyze tab screenshot |
| `docs/images/dashboard_tab.png` | Screenshots & UI Description | Dashboard tab screenshot |
| `docs/images/architecture_flow.png` | Architecture > Visual Architecture | Architecture flow diagram |

## Update Workflow

### 1. Scan for images

```bash
ls docs/images/
```

Check which `.png` files exist. The canonical set is:
- `upload_tab.png`
- `dashboard_tab.png`
- `architecture_flow.png`

### 2. Verify README image references

Ensure these markdown image references exist in `README.md`:

**Screenshots section** (after `## Screenshots & UI Description`):
```markdown
### Upload & Analyze Tab
![Upload & Analyze Tab](docs/images/upload_tab.png)

### Dashboard
![Dashboard Tab](docs/images/dashboard_tab.png)
```

**Architecture section** (before `### Detection Pipeline`):
```markdown
### Visual Architecture

![Architecture Flow](docs/images/architecture_flow.png)
```

### 3. Verify File Structure listing

The `## File Structure` section must include:

```
├── start.sh                # macOS/Linux startup script
├── start.bat               # Windows startup script
├── README.md               # This file
├── docs/
│   └── images/             # Documentation images (screenshots, architecture diagram)
```

### 4. Add new images

When a new image is added to `docs/images/`:

1. Add a markdown subsection in the appropriate README section
2. Use the format: `![Alt Text](docs/images/filename.png)`
3. Update the File Structure `docs/` listing if needed
4. Preserve all existing README content — only insert image references

## Rules

- **Never rewrite the full README** — use targeted insertions only
- **Use relative paths** (`docs/images/...`) not absolute paths
- **Place screenshots before** the "Interface Theme" subsection
- **Place architecture image before** the "Detection Pipeline" subsection
- Keep alt text descriptive and matching the subsection heading
