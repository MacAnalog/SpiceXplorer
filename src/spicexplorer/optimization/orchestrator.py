"""This Module implements the user endpoint for selecting optimizers types based on an input project_setup yaml file and optimization engine type"""
import logging
import numpy        as np

# Third-party imports
from    enum        import Enum
from    typing      import Dict, Tuple, Any, List, Type
from    pathlib     import Path
from    abc         import ABC, abstractmethod

# Symxplorer Specific Imports
from    spicexplorer.spice_engine    import NGSpice_Wrapper, Sim_Execution_Type
from    spicexplorer.core.domains    import Project_Setup, TestbenchParams

from    .base           import Spice_Base_Optimizer, Base_Optimizer
from    .stochastic.nevergrad      import Nevergrad_Spice_Bode_Optimizer, Nevergrad_Spice_Constraint_Satisfaction,  Nevergrad_Spice_Single_Objective
from    .stochastic.bayesian_ax    import Ax_Spice_Constraint_Satisfaction, Ax_Spice_Single_Objective

# ------------------ Module Logger ------------------

logger = logging.getLogger("spicexplorer.optimization.orchestrator")


# ------------------ Enums ------------------
class Optimizer_Type_Enum(Enum):
    BAYESIAN = "bayesian"
    EVOLUTIONARY = "evolutionary"
    NEVERGRAD_BODE = "nevergrad_bode"
    NEVERGRAD_CONSTRAINT = "nevergrad_constraint"
    NEVERGRAD_SINGLE = "nevergrad_single"
    AX_CONSTRAINT = "ax_constraint"
    AX_SINGLE = "ax_single"

SPICE_OPTIMIZER_CLASSES : Dict[Optimizer_Type_Enum, Type[Spice_Base_Optimizer]] = {
    Optimizer_Type_Enum.NEVERGRAD_BODE: Nevergrad_Spice_Bode_Optimizer,
    Optimizer_Type_Enum.NEVERGRAD_CONSTRAINT: Nevergrad_Spice_Constraint_Satisfaction,
    Optimizer_Type_Enum.NEVERGRAD_SINGLE: Nevergrad_Spice_Single_Objective,
    Optimizer_Type_Enum.AX_CONSTRAINT: Ax_Spice_Constraint_Satisfaction,
    Optimizer_Type_Enum.AX_SINGLE: Ax_Spice_Single_Objective,
}

# ------------------ Classes ------------------

class Circuit_Optimizer_Orchestrator_Base(ABC):
    def __init__(self, project_setup_path: str | Path, optimizer_type: Optimizer_Type_Enum, verbose: bool = False):
        self.project_setup_path = project_setup_path
        self.optimizer_type = optimizer_type
        self.verbose = verbose

        self.__post_init__()

    def __post_init__(self):
        # Order matters here
        self.project_setup:     Project_Setup   = self.read_project_setup()
        logger.debug(f"created the project setup for {self.project_setup.name}")

        self.spicelib_wrappers:  Dict[str, NGSpice_Wrapper] = self.create_spicelib_wrappers()
        logger.debug(f"created the spicelib_wrapper.")


    def read_project_setup(self) -> Project_Setup:
        # Load the project setup information
        try:
            PROJECT_SETUP = Project_Setup.from_yaml(yaml_path=self.project_setup_path)
        except Exception as e:
            logger.critical(
                f"Failed to load project setup from '{self.project_setup_path}': {e.__class__.__name__}: {e}"
            )
            raise RuntimeError(
                f"Error loading project setup from '{self.project_setup_path}'. "
                f"Check if the file exists, is valid YAML, and has correct structure."
            ) from e
        return PROJECT_SETUP
    
    def create_spicelib_wrappers(self) -> Dict[str, NGSpice_Wrapper]:
        PROJECT_SETUP = self.project_setup

        netlist_filename = Path(PROJECT_SETUP.ws_root) / Path(PROJECT_SETUP.netlist)
        output_folder    = Path(PROJECT_SETUP.ws_root) / Path(PROJECT_SETUP.outdir)
        sim_execution_t  = Sim_Execution_Type.RUN_AND_WAIT  # only RUN_AND_WAIT is supported as of now...
        path_to_simulator=Path(PROJECT_SETUP.simulator)
        
        logger.info("=============================================================================")
        logger.info(f"project: {PROJECT_SETUP.name} has ({len(PROJECT_SETUP.testbenches)}) testbenches.")
        logger.debug(f"\tpath_to_simulator {PROJECT_SETUP.simulator}")
        logger.debug(f"\tDUT netlist {netlist_filename}")
        logger.debug(f"\toutput_folder {output_folder}")


        # Create the Spice Simulator Wrapper for each testbench
        logger.debug(f"Creating spicelib_wrappers for each testbench...")
        logger.debug("-----------------------------------------------------------------------------")
        spicelib_wrappers : Dict[str, NGSpice_Wrapper] = {}

        for i, tb in enumerate(PROJECT_SETUP.testbenches):
            if not tb.enable:
                logger.info(f"({i+1}) Skipping disabled testbench: {tb.name} - {tb.description}")
                continue

            logger.debug(f"({i+1}) Creating spicelib_wrapper for testbench: {tb.name} - {tb.description}")
            logger.debug(f"\tspicelib_wrapper will use the following configs:")
            logger.debug(f"\t- testbench_name {tb.name}")
            logger.debug(f"\t- netlist_filename {tb.netlist}")

            wrapper = NGSpice_Wrapper(
                testbench_name=tb.name,
                netlist_filename=Path(PROJECT_SETUP.ws_root) / Path(tb.netlist),
                output_folder=output_folder,
                sim_execution_t=sim_execution_t,
                path_to_simulator=path_to_simulator,
                verbose=self.verbose
                )

            spicelib_wrappers[tb.name] = wrapper
            logger.debug(f"Created spicelib_wrapper for testbench: {tb.name}")
        logger.debug("-----------------------------------------------------------------------------")
        logger.info(f"Created ({len(spicelib_wrappers)}) spicelib_wrappers for project: {PROJECT_SETUP.name}")
        logger.info("=============================================================================")
        return spicelib_wrappers
    
    def get_project_setup(self) -> Project_Setup:
        return self.project_setup
    
    def get_spicelib_wrapper(self) -> Dict[str, NGSpice_Wrapper]:
        return self.spicelib_wrappers

    @abstractmethod
    def get_optimizer(self) -> Base_Optimizer:
        pass  


class Circuit_Optimizer_Orchestrator_with_SPICE(Circuit_Optimizer_Orchestrator_Base):
    def get_optimizer(self) -> Spice_Base_Optimizer:
        logger.info(f"creating the circuit_optimizer of type {self.optimizer_type.value}")
        circuit_optimizer = SPICE_OPTIMIZER_CLASSES[self.optimizer_type](
            spicelib_wrappers=self.spicelib_wrappers,
            setup_obj=self.project_setup
        )
        logger.info(f"created the circuit_optimizer; type {type(circuit_optimizer)}")
        return circuit_optimizer
    
    def run_sanity_on_spicelib_wrapper(self, use_editor: bool = True)-> bool:
        for tb_name, spicelib_wrapper in self.spicelib_wrappers.items():
            logger.info(f"running sanity check on spicelib_wrapper for testbench: {tb_name}")
            if not spicelib_wrapper.run_sanity_check(use_editor=use_editor, sim_execution_t=Sim_Execution_Type.RUN_NOW):
                logger.warning(f"sanity check failed for spicelib_wrapper for testbench: {tb_name}")
                return False
        return True