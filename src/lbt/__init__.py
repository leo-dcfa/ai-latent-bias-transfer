"""lbt — Latent Bias Transfer (LBT-2) measurement harness.

Scripts under scripts/ are thin CLIs; all science-bearing logic lives here so it
is unit-testable and shared identically across every model/condition/seed cell.
See SPEC.md for the experimental design this package implements.
"""

__all__ = ["config", "seeds", "runmeta", "io", "chat"]
