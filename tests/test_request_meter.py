"""Request-local error state and all-attempt usage, entirely offline."""
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from unittest import mock
from policybook import llm
from tests import test_experimental_flags as fixtures
from tests.test_experimental_flags import SeqStub, INSUFFICIENT, VALID


class RequestStateTests(unittest.TestCase):
    def test_errors_are_isolated_between_threads(self):
        barrier = threading.Barrier(12)
        def worker(i):
            with llm.request_context() as meter:
                llm._set_error(str(i))
                barrier.wait(timeout=5)
                return llm.get_last_error(), meter['attempts']
        with ThreadPoolExecutor(max_workers=12) as pool:
            self.assertEqual(list(pool.map(worker, range(12))),
                             [(str(i), 0) for i in range(12)])

    def test_unknown_quota_usage_and_safe_category(self):
        with llm.request_context() as meter:
            with mock.patch.object(llm, '_post_impl', side_effect=llm.QuotaExceeded('private detail')):
                with self.assertRaises(llm.QuotaExceeded):
                    llm._post('unused', 'unused', {})
            self.assertEqual(meter['attempts'], 1)
            self.assertEqual(meter['usage_unknown_attempts'], 1)
            self.assertEqual(meter['errors'], {'quota': 1})
            self.assertNotIn('private detail', str(meter))


class MeterIntegrationTests(unittest.TestCase):
    setUp = fixtures.FlagTests.setUp
    tearDown = fixtures.FlagTests.tearDown
    _answer = fixtures.FlagTests._answer

    def test_both_insufficient_and_success_tokens_counted(self):
        import os
        os.environ['OP05_GEN_INSUFFICIENT_RETRY'] = '1'
        os.environ['OP05_PRICE_IN_PER_M'] = '1'
        os.environ['OP05_PRICE_OUT_PER_M'] = '1'
        SeqStub.queue = [INSUFFICIENT, VALID]
        self.ans._price_in = 1
        self.ans._price_out = 1
        result = self._answer()
        meter = result.debug['generation_meter']
        self.assertEqual(meter['attempts'], 2)
        self.assertEqual(meter['input_tokens'], 20)
        self.assertEqual(meter['output_tokens'], 10)
        self.assertTrue(result.debug['cost_complete'])
        self.assertAlmostEqual(result.cost_usd, .00003)

    def test_all_insufficient_fallback_still_metered(self):
        import os
        os.environ['OP05_GEN_INSUFFICIENT_RETRY'] = '1'
        SeqStub.queue = [INSUFFICIENT, INSUFFICIENT]
        result = self._answer()
        self.assertEqual(result.generator, 'extractive')
        self.assertEqual(result.debug['generation_meter']['attempts'], 2)
        self.assertEqual(result.debug['generation_meter']['input_tokens'], 20)
