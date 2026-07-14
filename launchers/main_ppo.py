"""TRACE-GRPO wrapper around verl's PPO launcher.

The stock ``python -m verl.trainer.main_ppo`` entry point does not import
project-local registration modules. This wrapper imports the TRACE-GRPO
side-effect modules first, then delegates to verl's Hydra entry point.
"""

from __future__ import annotations

import trace_grpo.agent_loops.alfworld_agent_loop  # noqa: F401
import trace_grpo.agent_loops.sciworld_agent_loop  # noqa: F401
import trace_grpo.patches  # noqa: F401
from verl.trainer.main_ppo import main


if __name__ == "__main__":
    main()
