---
name: Python package tooling caveat
description: Python dependency installation and cleanup behavior in this workspace.
---

Use the managed Python package installer to add dependencies. In this workspace it used `uv` and installed packages into the project Python environment. The package-removal helper attempted global `pip uninstall`, but the Nix-managed interpreter had no pip and rejected changes as externally managed.

**Why:** The base Python runtime is immutable; global pip operations can fail even when project-scoped package installation succeeds.

**How to apply:** Keep `requirements.txt` and `pyproject.toml` as the source of truth, use `uv lock` after manifest changes, and avoid forcing changes into the base Nix interpreter.
