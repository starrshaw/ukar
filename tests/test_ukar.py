"""Offline tests. No model server, no GPU, no network."""

from __future__ import annotations

import json
import unittest

from ukar.backends.ollama import OllamaCompleter
from ukar.backends.ollama import build_payload as ollama_payload
from ukar.backends.ollama import chat_url as ollama_url
from ukar.backends.openai_compat import OpenAICompatCompleter, build_payload, chat_url, message_text
from ukar.backends.scripted import ScriptedCompleter
from ukar.parse import GapError, parse_gap
from ukar.planner import acquire
from ukar.router import Router
from ukar.schema import KnowledgeNeed
from ukar.sources import Quote


QUESTION = "How many stools fit in the studio?"


class ParseTests(unittest.TestCase):
    def test_rejects_model_handoff(self):
        raw = json.dumps({"delegate_to": "gpt-4", "confidence": 0.1, "can_answer_now": False, "needs": []})
        with self.assertRaises(GapError) as caught:
            parse_gap(raw, question=QUESTION)
        self.assertIn("hand", str(caught.exception).lower())

    def test_rejects_a_need_that_asks_for_the_answer(self):
        raw = json.dumps(
            {
                "confidence": 0.2,
                "can_answer_now": False,
                "known": [],
                "needs": [
                    {
                        "id": "expert",
                        "statement": "An expert opinion on the whole job",
                        "why": "I want the answer",
                        "scope": "public",
                        "freshness": "static",
                        "keys": ["expert_take"],
                    }
                ],
            }
        )
        with self.assertRaises(GapError):
            parse_gap(raw, question=QUESTION)

    def test_accepts_fenced_json(self):
        raw = """```json
        {"confidence": 1, "can_answer_now": true, "known": ["enough"], "needs": []}
        ```"""
        gap = parse_gap(raw, question="anything")
        self.assertTrue(gap.can_answer_now)
        self.assertEqual(gap.needs, [])


class PlannerTests(unittest.TestCase):
    def _need(self, need_id, scope, keys, statement="A fact"):
        return KnowledgeNeed(need_id, statement, "because", scope, "static", keys)

    def test_off_machine_source_is_never_called(self):
        class Remote:
            id = "remote-table"
            privacy = "leaves_device"
            fetches = 0

            def cover(self, needs, known):
                del known
                return [need.id for need in needs]

            def quote(self, need_ids):
                return Quote(0.0, self.privacy, 1 if need_ids else 0)

            def fetch(self, needs, known):
                del needs, known
                self.fetches += 1
                return []

        remote = Remote()
        result = acquire(
            [self._need("user_fact", "personal", ["user_fact"], "A fact only the user knows")],
            [remote],
            {},
        )
        self.assertEqual(remote.fetches, 0)
        self.assertEqual(result.facts, [])
        self.assertTrue(result.questions)
        self.assertTrue(
            any("on this machine" in item.reason for item in result.considerations)
        )

    def test_supplied_fact_fills_the_need(self):
        from ukar.catalog import default_sources

        result = acquire(
            [
                KnowledgeNeed(
                    "stool_count",
                    "How many stools are in the room",
                    "The count has to come from the user",
                    "personal",
                    "static",
                    ["stool_count"],
                )
            ],
            default_sources(),
            {"stool_count": "4"},
        )
        self.assertEqual(result.facts[0].source_id, "given")
        self.assertEqual(result.facts[0].value, "4")
        self.assertEqual(result.facts[0].privacy, "local")


class RouterTests(unittest.TestCase):
    def test_supplied_facts_are_repeated_and_not_extended(self):
        trace = Router(ScriptedCompleter()).run(QUESTION, {"stool_count": "4"})
        self.assertEqual(trace.status, "answered")
        self.assertIn("stool_count = 4", trace.answer)
        self.assertNotIn("gpt", trace.answer.lower())
        self.assertTrue(any("on this machine" in note for note in trace.notes))
        self.assertTrue(all(fact.privacy == "local" for fact in trace.facts))

    def test_no_facts_does_not_guess(self):
        trace = Router(ScriptedCompleter()).run(QUESTION)
        self.assertEqual(trace.status, "waiting_on_user")
        self.assertEqual(trace.answer, "")
        self.assertIn("user_fact", " ".join(trace.questions))

    def test_any_other_question_uses_the_same_rule(self):
        trace = Router(ScriptedCompleter()).run("Who wrote the manual?")
        self.assertEqual(trace.status, "waiting_on_user")
        self.assertIn("user_fact", " ".join(trace.questions))

    def test_handoff_completion_is_refused(self):
        class Handoff:
            name = "scripted"

            def complete(self, messages, *, json_mode=False, max_tokens=700):
                del messages, max_tokens
                if json_mode:
                    return json.dumps({"route_to": "claude", "confidence": 0, "can_answer_now": False, "needs": []})
                return "Ask a bigger model."

        trace = Router(Handoff()).run(QUESTION, {"stool_count": "4"})
        self.assertEqual(trace.status, "refused")
        self.assertEqual(trace.answer, "")

    def test_cut_off_gap_json_is_retried_with_a_larger_budget(self):
        class CutOff:
            name = "scripted"

            def __init__(self):
                self.budgets: list[int] = []

            def complete(self, messages, *, json_mode=False, max_tokens=700):
                self.budgets.append(max_tokens)
                if len(self.budgets) == 1:
                    return '{"confidence": 0.2, "can_answer_now": false, "needs": [{"id": "user'
                return ScriptedCompleter().complete(messages, json_mode=json_mode, max_tokens=max_tokens)

        completer = CutOff()
        trace = Router(completer).run(QUESTION)
        self.assertEqual(completer.budgets, [1400, 2800])
        self.assertEqual(trace.status, "waiting_on_user")
        self.assertIn("user_fact", " ".join(trace.questions))


class BackendShapeTests(unittest.TestCase):
    def test_llamacpp_payload_is_chat_on_the_local_route(self):
        messages = [{"role": "user", "content": QUESTION}]
        payload = build_payload(messages, model="local", max_tokens=50, json_mode=True)
        self.assertEqual(payload["messages"][0]["content"], QUESTION)
        self.assertEqual(payload["response_format"]["type"], "json_object")
        self.assertEqual(chat_url("127.0.0.1:8080/v1"), "http://127.0.0.1:8080/v1/chat/completions")
        text = message_text({"choices": [{"message": {"content": "{\"ok\": true}"}}]})
        self.assertIn("ok", text)

    def test_ollama_json_format_is_only_for_the_gap_turn(self):
        messages = [{"role": "user", "content": "hi"}]
        gap = ollama_payload(messages, model="llama3.2", json_mode=True)
        prose = ollama_payload(messages, model="llama3.2", json_mode=False)
        self.assertEqual(gap["format"], "json")
        self.assertNotIn("format", prose)
        self.assertEqual(ollama_url("127.0.0.1:11434"), "http://127.0.0.1:11434/api/chat")
        self.assertTrue(gap["stream"] is False)

    def test_remote_model_hosts_are_refused(self):
        with self.assertRaises(ValueError):
            OpenAICompatCompleter("https://api.openai.com/v1")
        with self.assertRaises(ValueError):
            OllamaCompleter("http://192.168.1.20:11434", "llama3.2")
        self.assertTrue(OpenAICompatCompleter("http://127.0.0.1:8080").base_url.startswith("http://127.0.0.1"))


if __name__ == "__main__":
    unittest.main()
