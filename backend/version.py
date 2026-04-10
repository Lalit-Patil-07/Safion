"""
Safion — semantic version

Bump rules
----------
  MAJOR  breaking API change or complete subsystem replacement
  MINOR  new feature, backward-compatible
  PATCH  bug fix, no new functionality

Current history
---------------
  0.1.0  initial monolith (app_server.py)
  1.0.0  modular rewrite — InsightFace embeddings, identity clustering,
         JWT auth scaffold, async task queue
"""

MAJOR = 1
MINOR = 0
PATCH = 0
STAGE = "beta"           # alpha | beta | rc | stable

__version__ = f"{MAJOR}.{MINOR}.{PATCH}"
VERSION_STRING = f"v{__version__}-{STAGE}"
