"""This Module implements the nevergrad-based (evolutionary algorithms) optimizers """
import logging
import numpy        as np
import nevergrad    as ng

from    typing      import Dict, Tuple, Any, Optional

# Symxplorer Specific Imports
from   spicexplorer.spice_engine    import NGSpice_Wrapper
from   spicexplorer.core.domains    import Project_Setup, TestbenchParams

from   spicexplorer.optimization.base         import Spice_Constraint_Satisfaction, Spice_Single_Objective, Spice_Bode_Optimizer, Base_Optimizer

logger = logging.getLogger("spicexplorer.optimization.stochastic.nevergrad")
logger.debug(f'imported {__name__}')


# ----------------------------
# --- Global Constants ---
# ----------------------------


# ----------------------------
# --- Function Definitions ---
# ---------------------------

def create_optimizer(
    optimizer_name: str,
    parametrization: ng.p.Dict,
    budget: int,
    optimizer_kwargs: Optional[Dict[str, Any]] = None,
    random_seed: Optional[int] = None
) -> ng.optimizers.base.Optimizer:
    """
    Factory function to instantiate a Nevergrad optimizer from configuration.
    
    Handles two cases:
    1. Families: Configurable classes (e.g. DifferentialEvolution, ParametrizedCMA).
       These require a two-step init: Family(**kwargs) -> Optimizer(params, budget).
    2. Registry: Pre-configured strings (e.g. 'NGOpt', 'TwoPointsDE').
       These are instantiated directly: RegistryKey(params, budget, **kwargs).
    """
    if optimizer_kwargs is None:
        optimizer_kwargs = {}
    
    # 1. Set Random Seed (Global for the parametrization)
    if random_seed is not None:
        parametrization.random_state = np.random.RandomState(random_seed)

    # Make a copy of kwargs to avoid mutating the original config
    kwargs = optimizer_kwargs.copy()
    
    # Extract 'num_workers' (common to all, defaults to 1)
    num_workers = kwargs.pop('num_workers', 1)

    # -------------------------------------------------------------------------
    # CASE A: CONFIGURABLE FAMILIES
    # Check if the name exists in ng.families (e.g., "DifferentialEvolution")
    # -------------------------------------------------------------------------
    if hasattr(ng.families, optimizer_name):
        try:
            # 1. Get the Family Class
            family_class = getattr(ng.families, optimizer_name)
            
            # 2. Configure the Family (Pass algorithmic settings like 'popsize', 'crossover')
            #    Any argument that the Family constructor doesn't accept will raise a TypeError here.
            optimizer_factory = family_class(**kwargs)
            
            # 3. Instantiate the Optimizer (Pass execution settings)
            optimizer = optimizer_factory(
                parametrization=parametrization, 
                budget=budget, 
                # num_workers=num_workers # FIXME: Do not enforce num_workers for now. 
            )
            
            logger.info(f"Initialized Family '{optimizer_name}' with config: {kwargs}")
            return optimizer

        except TypeError as e:
            logger.error(f"Invalid argument provided for Family '{optimizer_name}': {e}")
            raise

    # -------------------------------------------------------------------------
    # CASE B: REGISTRY PRESETS
    # Check if the name exists in the registry (e.g., "NGOpt", "TwoPointsDE")
    # -------------------------------------------------------------------------
    registry = ng.optimizers.registry.get(optimizer_name)
    if registry is not None:
        # Registry optimizers are instantiated directly.
        # Note: They typically do NOT accept algorithmic kwargs (like 'crossover').
        # If kwargs contains something the registry opt doesn't support, it might crash or ignore it.
        try:
            optimizer = registry(
                parametrization=parametrization, 
                budget=budget, 
                num_workers=num_workers,
                **kwargs # Passing remaining kwargs (rarely used for registry items)
            )
            logger.info(f"Initialized Registry Optimizer '{optimizer_name}'")
            return optimizer
        except TypeError as e:
             logger.warning(
                 f"Optimizer '{optimizer_name}' rejected extra arguments {kwargs}. "
                 "Registry presets usually do not accept algorithmic kwargs. "
                 "Use the Family Name instead if you want to configure it."
             )
             raise e
            
    # -------------------------------------------------------------------------
    # CASE C: FAILURE
    # -------------------------------------------------------------------------
    raise ValueError(
        f"Optimizer '{optimizer_name}' not found in 'ng.families' or 'ng.optimizers.registry'.\n"
        f"Available Families: {[x for x in dir(ng.families) if not x.startswith('_')]}\n"
    )

# ----------------------------
# --- Class Definitions ---
# ----------------------------

# ------------------------------------------------
# A [ABSTRACT] Nevergrad-based Optimizers
# ------------------------------------------------
class NevergradMixin(Base_Optimizer):
    """Reusable mixin for all Nevergrad-based optimizers."""
    # --- Overwriting Some Abstract Methods ---
    def parameterize(self) -> ng.p.Dict:        
        parameters: Dict[str, ng.p.Scalar] = {}

        for param in self.setup_obj.dut_params:
            if param.log_scale:
                 p_obj = ng.p.Log(
                    lower=self.optimizer_config.log_variable_bounds.min, 
                    upper=self.optimizer_config.log_variable_bounds.max)
            else:
                p_obj = ng.p.Scalar(
                    lower=self.optimizer_config.lin_variable_bounds.min, 
                    upper=self.optimizer_config.lin_variable_bounds.max)
            
            if param.is_integer:
                p_obj.set_integer_casting()

            parameters[param.name] = p_obj
                
        self.parametrization = ng.p.Dict(**parameters)
        return self.parametrization
    
    def _create_optimizer_obj(self) -> bool:
        if self.parametrization is None:
            logger.critical("NEED TO CALL self.parameterize")
            return False

        try:
            self.optimizer = create_optimizer(
                optimizer_name=self.optimizer_config.name,
                parametrization=self.parametrization,
                budget=self.optimizer_config.budget,
                optimizer_kwargs=self.optimizer_config.optimizer_kwargs,
                random_seed=self.optimizer_config.random_seed
            )
            
            # FIXME: Handle Initial Points Manually
            # if self.optimizer_config.initial_points:
            #     for point in self.optimizer_config.initial_points:
            #          self.optimizer.suggest(self.parametrization.spawn_child(new_value=point).value)
            
            return True
            
        except Exception as e:
            logger.critical(f"Failed to create optimizer: {e}")
            return False

    def optimization_step(self) -> Tuple[Dict[str, np.floating] , np.floating , Dict[str, Any]]:
        # Get a new candidate
        candidate : ng.p.Parameter = self.optimizer.ask()
        # Evaluate function
        denorm_params: Dict[str, float] = self.denormalize_params(parameterization=candidate.value)
        curr_score, metadata = self.evaluate(parameterization=denorm_params)
        # Provide feedback to optimizer (The negative of the fitness score is used because the optimizer is set to minimize this value... this way the optimizer will maximize the fitness score.
        self.optimizer.tell(candidate, -1 * curr_score)
        return candidate.value, curr_score, metadata

# ------------------------------------------------
# B [USER-ENDPOINT] Nevergrad-based Bode Fitter
# ------------------------------------------------
class Nevergrad_Spice_Bode_Optimizer(NevergradMixin, Spice_Bode_Optimizer):
    pass

# ------------------------------------------------
# B [USER-ENDPOINT] Nevergrad-based Constraint Satisfaction
# ------------------------------------------------
class Nevergrad_Spice_Constraint_Satisfaction(NevergradMixin, Spice_Constraint_Satisfaction):
    def __init__(self,
                 setup_obj: Project_Setup,
                 spicelib_wrappers : Dict[TestbenchParams, NGSpice_Wrapper]):
        super().__init__(setup_obj = setup_obj, spicelib_wrappers = spicelib_wrappers)
        self.parametrization: ng.p.Dict | None = None
        logger.info(f"started the {__class__} optimizer class")
# ------------------------------------------------
# B [USER-ENDPOINT] Nevergrad-based Single Objective Optimizer
# ------------------------------------------------
class Nevergrad_Spice_Single_Objective(NevergradMixin, Spice_Single_Objective):
    def __init__(self,
                 setup_obj: Project_Setup,
                 spicelib_wrappers : Dict[TestbenchParams, NGSpice_Wrapper]):
        super().__init__(setup_obj = setup_obj, spicelib_wrappers = spicelib_wrappers)
        self.parametrization: ng.p.Dict | None = None
        logger.info(f"started the {__class__} optimizer class")