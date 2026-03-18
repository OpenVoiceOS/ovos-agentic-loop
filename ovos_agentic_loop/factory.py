# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""OPM entry-point factory classes for ovos-agentic-loop."""
from ovos_agentic_loop.chain_of_thought import ChainOfThoughtEngine
from ovos_agentic_loop.critic import CRITICEngine
from ovos_agentic_loop.plan_execute import PlanAndExecuteEngine
from ovos_agentic_loop.react import ReActLoopEngine
from ovos_agentic_loop.reflexion import ReflexionEngine
from ovos_agentic_loop.self_ask import SelfAskEngine
from ovos_agentic_loop.tree_of_thoughts import TreeOfThoughtsEngine

__all__ = [
    "ReActLoopEnginePlugin",
    "PlanAndExecuteEnginePlugin",
    "ReflexionEnginePlugin",
    "SelfAskEnginePlugin",
    "ChainOfThoughtEnginePlugin",
    "CRITICEnginePlugin",
    "TreeOfThoughtsEnginePlugin",
]


class ReActLoopEnginePlugin(ReActLoopEngine):
    """
    OPM-registered plugin class for the ReAct loop engine.

    Entry point group: ``opm.agents.chat``
    Entry point name:  ``ovos-react-loop``
    """


class PlanAndExecuteEnginePlugin(PlanAndExecuteEngine):
    """
    OPM-registered plugin class for the Plan-and-Execute engine.

    Entry point group: ``opm.agents.chat``
    Entry point name:  ``ovos-plan-execute-loop``
    """


class ReflexionEnginePlugin(ReflexionEngine):
    """
    OPM-registered plugin class for the Reflexion engine.

    Entry point group: ``opm.agents.chat``
    Entry point name:  ``ovos-reflexion-loop``
    """


class SelfAskEnginePlugin(SelfAskEngine):
    """
    OPM-registered plugin class for the Self-Ask engine.

    Entry point group: ``opm.agents.chat``
    Entry point name:  ``ovos-self-ask-loop``
    """


class ChainOfThoughtEnginePlugin(ChainOfThoughtEngine):
    """
    OPM-registered plugin class for the Chain-of-Thought engine.

    Entry point group: ``opm.agents.chat``
    Entry point name:  ``ovos-chain-of-thought-loop``
    """


class CRITICEnginePlugin(CRITICEngine):
    """
    OPM-registered plugin class for the CRITIC engine.

    Entry point group: ``opm.agents.chat``
    Entry point name:  ``ovos-critic-loop``
    """


class TreeOfThoughtsEnginePlugin(TreeOfThoughtsEngine):
    """
    OPM-registered plugin class for the Tree-of-Thoughts engine.

    Entry point group: ``opm.agents.chat``
    Entry point name:  ``ovos-tree-of-thoughts-loop``
    """
