"""TRACE-GRPO patches over verl. Importing this module registers the
``trace_l3_mask`` advantage estimator with verl's adv-estimator registry
and (lazily) the ``capa`` attention implementation with transformers.

Side-effecting imports — kept intentional so that callers can simply::

    import trace_grpo.patches  # noqa: F401

before invoking ``verl.trainer.main_ppo``.
"""

from . import level3_patch  # noqa: F401  (registers trace_l3_mask)
from . import critique_conditioned_provider  # noqa: F401  (registers critique_v1)
from . import counterfactual_provider  # noqa: F401  (registers counterfactual_mask_v1)
from .capa import register_capa_attention

register_capa_attention()
