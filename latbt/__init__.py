"""latbt — latent (framing-only) bias transfer measurement harness.

This package holds the shared plumbing used by the top-level scripts
(validate_data.py, train_lora.py, eval_stance.py, analyze.py). The scripts
are thin CLIs; the science-bearing logic lives here so it is unit-testable
and shared identically across every model in the config.
"""

__all__ = [
    "config",
    "seeds",
    "blocklist",
    "dataio",
    "scoring",
    "modeling",
]
