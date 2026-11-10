"""Offline span construction: no fabricated joins or cross-source merges."""
import unittest
from scripts.offline_citation_spans import consolidate


class SpanTests(unittest.TestCase):
    def citation(self, text, section='d#s1'):
        return {'doc_id':'d', 'section':section, 'quote':text}

    def test_intervening_text_is_preserved(self):
        text = 'Old rule. Important exception. New rule.'
        row = {'answer':'compare', 'citations':[self.citation('Old rule.'), self.citation('New rule.')]}
        new = consolidate(row, {('d','d#s1'):text})
        self.assertEqual(new['citations'], [self.citation(text)])
        self.assertEqual(len(row['citations']), 2)
        self.assertEqual(new['answer'], row['answer'])

    def test_missing_exact_quote_leaves_group_unchanged(self):
        row = {'citations':[self.citation('Old rule.'), self.citation('invented')]}
        self.assertEqual(consolidate(row, {('d','d#s1'):'Old rule.'}), row)

    def test_different_passages_never_merged(self):
        row = {'citations':[self.citation('Old.'),self.citation('New.','d#s2')]}
        self.assertEqual(consolidate(row, {('d','d#s1'):'Old.',('d','d#s2'):'New.'}),row)
