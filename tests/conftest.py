import os
import tempfile

# Point the artifact store at a throwaway SQLite file before app import.
os.environ["CALIBRATION_DB"] = os.path.join(tempfile.mkdtemp(), "test_artifacts.db")
