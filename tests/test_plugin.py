"""The tool body stays local. No llama-server and no Ollama in this file."""

from __future__ import annotations

import json
import unittest

from ukar.format import USED_LINE, finish_reply
from ukar.plugin import TOOL_NAME, execute, tool_schemas


class PluginTests(unittest.TestCase):
    def test_schema_is_one_local_tool(self):
        tools = tool_schemas()
        self.assertEqual(len(tools), 1)
        self.assertEqual(tools[0]["function"]["name"], TOOL_NAME)

    def test_acquire_uses_a_supplied_fact(self):
        needs = {
            "confidence": 0.2,
            "can_answer_now": False,
            "known": [],
            "needs": [
                {
                    "id": "stool_count",
                    "statement": "How many stools are in the room",
                    "why": "The count has to come from the user",
                    "scope": "personal",
                    "freshness": "static",
                    "keys": ["stool_count"],
                }
            ],
        }
        text = execute(
            TOOL_NAME,
            {"question": "How many stools fit?", "needs_json": json.dumps(needs), "facts": "stool_count=4"},
        )
        self.assertIn("4", text)
        self.assertIn("on this machine", text)
        self.assertIn("The count has to come from the user", text)
        self.assertIn("Confidence before acquisition  0.2", text)
        self.assertIn(USED_LINE, text)
        self.assertNotIn("Customer problem", text)
        self.assertTrue(execute("send_away", {}).startswith("ERROR:"))
        marked = finish_reply("About 8 kW.", used=True)
        self.assertTrue(marked.rstrip().endswith(USED_LINE))
        self.assertEqual(finish_reply(marked, used=True).count(USED_LINE), 1)
        self.assertNotIn(USED_LINE, finish_reply("plain answer", used=False))


if __name__ == "__main__":
    unittest.main()
