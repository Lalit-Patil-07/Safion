"""
Safion — semantic version

Bump rules
----------
  MAJOR  breaking API change or full subsystem replacement
  MINOR  new feature, backward-compatible
  PATCH  bug fix, no new functionality

History
-------
  0.1.0  initial monolith (app_server.py)
  1.0.0  modular rewrite — InsightFace, identity system, JWT, async queue
  1.1.0  multi-prototype identity model, merge suggestions, review queue UI
  1.2.0  IoU face tracker (temporal consistency), delayed identity creation
  1.3.0  cross-track identity continuity — match_and_store_track(),
         fixed singleton similarity formula, recalibrated thresholds
"""

MAJOR = 1
MINOR = 3
PATCH = 0
STAGE = "beta"

__version__    = f"{MAJOR}.{MINOR}.{PATCH}"
VERSION_STRING = f"v{__version__}-{STAGE}"
