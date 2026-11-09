"""Neutral environment-selected entry; no scientific identity in process argv."""
import importlib.util
import json
import os
import sys
from pathlib import Path

root = Path(os.environ['Q_SOURCE']).resolve()
spec = importlib.util.spec_from_file_location('q', root / '__init__.py', submodule_search_locations=[str(root)])
package = importlib.util.module_from_spec(spec)
sys.modules['q'] = package
spec.loader.exec_module(package)
module = __import__('q.' + os.environ['Q_MODULE'], fromlist=['run'])
module.run(json.loads(Path(os.environ['Q_CONFIG']).read_text()))
