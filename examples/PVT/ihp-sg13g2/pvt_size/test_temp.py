from spicexplorer.optimization.orchestrator import Circuit_Optimizer_Orchestrator_with_SPICE, Optimizer_Type_Enum
from spicexplorer.logging.logger_setup import setup_loggers, setup_loggers_with_spicelib_suppression

from pathlib import Path
import os
#logger = setup_loggers_with_spicelib_suppression()
'''
from spicelib.raw.raw_read import RawRead

raw = RawRead("/home/hoanga12/Documents/Course_Project/Xplore_Danial_PVT/SpiceXplorer/examples/PVT/ihp-sg13g2/spice/rawspice.raw")

# List ALL plot names in the file
print(raw.get_plot_names())
# Output: ['ac1', 'ac2', 'ac3', ...]
'''
path_to_project_setup = Path("./project_setup.yaml")
optimizer_type = Optimizer_Type_Enum.NEVERGRAD_CONSTRAINT
orchestrator = Circuit_Optimizer_Orchestrator_with_SPICE(
    project_setup_path=path_to_project_setup,
    optimizer_type=optimizer_type,
    verbose=True
)

optimizer = orchestrator.get_optimizer()

optimizer.parameterize()
os.environ['PATH'] = '/home/hoanga12/local/bin:' + os.environ.get('PATH', '')
os.environ['SPICE_USERINIT_DIR'] = '/home/hoanga12/local/pdks/ihp-sg13g2/libs.tech/ngspice'

# Verify
import shutil
print("ngspice:", shutil.which('ngspice'))
print("SPICE_USERINIT_DIR:", os.environ.get('SPICE_USERINIT_DIR'))

optimizer.optimize()

optimizer.plot_score(show=True)