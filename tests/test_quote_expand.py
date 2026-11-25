import os
import unittest
from pathlib import Path

from policybook.answer import Answerer, Citation

R = Path(__file__).resolve().parent.parent


class QuoteExpandTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.a = Answerer(str(R / ".index"), str(R / "models" / "all-MiniLM-L6-v2"),
                         str(R / "data" / "corpus"))

    def test_expanded_quote_is_verbatim_superset(self):
        sec = next(s for s in self.a.corpus.sections if len(s.text) > 400)
        first = sec.text.strip().split(". ")[0][:200]
        c = Citation(doc_id=sec.passage_id.split("#")[0], section=sec.passage_id, quote=first)
        ans = sec.text[:600]
        out = self.a._expand_citations(ans, [c])[0]
        self.assertTrue(self.a._verify_quote(out.doc_id, out.section, out.quote))
        self.assertIn(first.lower()[:40], out.quote.lower())

    def test_flag_off_is_identity(self):
        os.environ["OP05_QUOTE_EXPAND"] = "0"
        c = [Citation(doc_id="d", section="d#s1", quote="q")]
        self.assertEqual(self.a._maybe_expand("x", c), c)


if __name__ == "__main__":
    unittest.main()


class TrimFlagTest(unittest.TestCase):
    def test_trim_default_off_and_budget(self):
        from policybook import llm
        os.environ.pop("OP05_TRIM_INPUT", None)
        os.environ.pop("OP05_GEN_EV_CHAR_BUDGET", None)
        class H:  # minimal hit
            def __init__(s, i): s.doc_id="d"; s.section=f"d#s{i}"; s.text="x"*400
        hits=[H(i) for i in range(6)]
        n_off=len(llm.pack_evidence(hits, {}))
        os.environ["OP05_TRIM_INPUT"]="1"
        try:
            n_on=len(llm.pack_evidence(hits, {}))
        finally:
            os.environ.pop("OP05_TRIM_INPUT", None)
        self.assertLess(n_on, n_off)
        self.assertGreaterEqual(n_on, 3)


class FlagDefaultsTest(unittest.TestCase):
    def test_retry_default_is_insufficient_only(self):
        os.environ.pop("OP05_GEN_RETRY_BROAD", None)  # default ON since 2026-11-25
        self.assertTrue(Answerer._retry_reason_ok("model said INSUFFICIENT"))
        self.assertTrue(Answerer._retry_reason_ok("no verifiable citations after retry"))
        os.environ["OP05_GEN_RETRY_BROAD"] = "0"
        try:
            self.assertTrue(Answerer._retry_reason_ok("model said INSUFFICIENT"))
            self.assertFalse(Answerer._retry_reason_ok("no verifiable citations after retry"))
        finally:
            os.environ.pop("OP05_GEN_RETRY_BROAD", None)

    def test_effort_flag_off_and_on(self):
        a = QuoteExpandTest.a if hasattr(QuoteExpandTest, "a") else None
        os.environ.pop("OP05_GEN_REASONING_NUM", None)
        a = Answerer(str(R / ".index"), str(R / "models" / "all-MiniLM-L6-v2"), str(R / "data" / "corpus"))
        self.assertEqual(a._effort_for("How many days within the deadline?"), "medium")  # default ON
        os.environ["OP05_GEN_REASONING_NUM"] = "0"
        try:
            self.assertIsNone(a._effort_for("How many days within the deadline?"))
            os.environ["OP05_GEN_REASONING_NUM"] = "medium"
            self.assertEqual(a._effort_for("How many days within the deadline?"), "medium")
            self.assertIsNone(a._effort_for("Who regulates mutual funds"))
        finally:
            os.environ.pop("OP05_GEN_REASONING_NUM", None)
