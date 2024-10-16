"""
Exposes common paths useful for manipulating datasets and generating figures.

"""
# import os
from pathlib import Path
# os.environ["PATH"] += os.pathsep + str(Path.home() / "bin")

# Local path to understand where to move from
local = Path(__file__).resolve().name

# Absolute path to the top level of the repository
if local == 'SHAP':
    root = Path(__file__).resolve().parents[3].absolute()
else:
    root = Path(__file__).resolve().parents[2].absolute()

# Absolute path to the `src` folder
src = root / "src"

# Absolute path to the `src/data` folder (contains datasets)
data = src / "data"

# Absolute path to the `src/data/models` folder (contains prediction models)
models = data / "models"

# Absolute path to the `src/static` folder (contains static images)
static = src / "static"

# Absolute path to the `src/scripts` folder (contains figure/pipeline scripts)
scripts = src / "scripts"

# Absolute path to the `src/tex` folder (contains the manuscript)
tex = src / "tex"

# Absolute path to the `src/tex/figures` folder (contains figure output)
figures = tex / "figures"

# Absolute path to the `src/tex/output` folder (contains other user-defined output)
output = tex / "output"
