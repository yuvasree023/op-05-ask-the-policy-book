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
        os.environ.pop("OP05_QUOTE_EXPAND", None)
        c = [Citation(doc_id="d", section="d#s1", quote="q")]
        self.assertEqual(self.a._maybe_expand("x", c), c)


if __name__ == "__main__":
    unittest.main()
