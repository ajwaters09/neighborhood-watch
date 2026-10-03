"""The chat agent's wiring: every tool the model is told about is dispatched to a logged
agent_tools function whose parameters match the spec. Offline; nothing is called.

    python -m pytest tests/test_agent_wiring.py
"""

import inspect

from src import agent_chat, agent_tools


def test_every_spec_has_a_logged_tool_with_matching_params():
    specs = {s["function"]["name"]: s["function"] for s in agent_chat.TOOL_SPECS}
    assert set(specs) == set(agent_chat._DISPATCH)
    for name, spec in specs.items():
        fn = agent_chat._DISPATCH[name]
        assert fn is getattr(agent_tools, name)
        assert hasattr(fn, "__wrapped__"), f"{name} isn't @_logged"
        params = inspect.signature(fn).parameters
        assert set(spec["parameters"]["properties"]) <= set(params), name
        required = {p for p, v in params.items() if v.default is inspect.Parameter.empty}
        assert required <= set(spec["parameters"].get("required", [])), name
