"""edge-containers benchmark harness (spec §1-9 core).

Datasets and engines are plugins behind fixed interfaces; the runner and metric
code never contain engine- or dataset-specific branches. Every metric is
computed the same way regardless of engine, dataset, language or backend.

Layout (spec §2): datasets/ engines/ probes/ metrics/ runner.py report.py env.py
CLI: python -m bench.cli {prepare,selfcheck,run,report,compare}
"""

__version__ = "0.1.0"
