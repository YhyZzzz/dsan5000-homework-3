"""Offline checks for the actual notebook utility cell; no API or credentials used."""

import contextlib
import copy
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest


NOTEBOOK = Path(__file__).resolve().parents[1] / "HW-3.1-crawl.ipynb"


class FakeClient:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def search(self, params):
        self.calls.append(dict(params))
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return copy.deepcopy(response)


def response(number=1, token=None, jobs=True):
    value = {
        "search_metadata": {"status": "Success"},
        "jobs_results": [{"job_id": str(number), "title": "Data Scientist", "company_name": "Test"}] if jobs else [],
    }
    if token:
        value["serpapi_pagination"] = {"next_page_token": token}
    return value


class CrawlTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.env = {}
        notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
        with contextlib.redirect_stdout(io.StringIO()):
            for cell in notebook["cells"]:
                if cell.get("id") in {"crawl-imports", "crawl-utilities"}:
                    exec(compile("".join(cell["source"]), str(NOTEBOOK), "exec"), self.env)
        base = Path(self.temp.name)
        self.now = 10000.0
        self.waits = []

        def sleep(seconds):
            self.waits.append(seconds)
            self.now += seconds

        self.env.update(
            DATA_DIR=base / "data",
            REPORT_DIR=base / "reports",
            LEDGER_FILE=base / "reports" / "search-attempts.json",
            time=SimpleNamespace(time=lambda: self.now, sleep=sleep),
        )

    def run_crawl(self, titles):
        with contextlib.redirect_stdout(io.StringIO()):
            self.env["crawl_titles"](titles)

    def test_three_pages_follow_tokens_and_rerun_uses_zero_requests(self):
        client = FakeClient([response(1, "page-2"), response(2, "page-3"), response(3)])
        self.env["client"] = client
        self.run_crawl(["Data Scientist"])
        self.assertEqual([call.get("next_page_token") for call in client.calls], [None, "page-2", "page-3"])
        self.assertTrue(all("start" not in call and "num" not in call for call in client.calls))
        self.assertEqual(self.waits, [75.0, 75.0])
        self.assertEqual(len(self.env["saved_pages"]()), 3)
        self.run_crawl(["Data Scientist"])
        self.assertEqual(len(client.calls), 3)
        self.assertEqual(self.env["crawl_summary"](["Data Scientist"])[0]["status"], "complete_3_pages")

    def test_partial_run_resumes_from_last_token(self):
        first = FakeClient([response(1, "page-2")])
        self.env["client"] = first
        with contextlib.redirect_stdout(io.StringIO()):
            self.env["search_google_jobs"]("Data Scientist")
        self.env["client"] = FakeClient([response(2, "page-3"), response(3)])
        self.run_crawl(["Data Scientist"])
        self.assertEqual(self.env["client"].calls[0]["next_page_token"], "page-2")
        self.assertEqual(len(self.env["saved_pages"]()), 3)

    def test_exhausted_title_is_skipped_on_rerun(self):
        client = FakeClient([response()])
        self.env["client"] = client
        self.run_crawl(["Data Scientist"])
        self.run_crawl(["Data Scientist"])
        self.assertEqual(len(client.calls), 1)
        self.assertEqual(self.env["crawl_summary"](["Data Scientist"])[0]["status"], "exhausted")

    def test_empty_page_does_not_follow_a_spurious_token(self):
        client = FakeClient([response(jobs=False, token="spurious")])
        self.env["client"] = client
        self.run_crawl(["Data Scientist"])
        self.run_crawl(["Data Scientist"])
        self.assertEqual(len(client.calls), 1)

    def test_api_error_moves_to_next_title_without_saving_error_as_data(self):
        self.env["client"] = FakeClient([{"error": "temporary provider failure"}, response()])
        self.run_crawl(["Data Scientist", "Data Analyst"])
        self.assertEqual(len(self.env["client"].calls), 2)
        self.assertEqual(len(self.env["saved_pages"]()), 1)
        self.assertEqual(self.env["crawl_summary"](["Data Scientist"])[0]["status"], "api_error")

    def test_malformed_response_is_not_recorded_as_an_exhausted_title(self):
        self.env["client"] = FakeClient([{}])
        self.run_crawl(["Data Scientist"])
        self.assertEqual(self.env["saved_pages"](), [])
        self.assertEqual(self.env["crawl_summary"](["Data Scientist"])[0]["status"], "invalid_response")

    def test_timeout_is_not_retried_and_message_does_not_leak_key(self):
        self.env["client"] = FakeClient([TimeoutError("URL contains SECRET_TEST_KEY")])
        stream = io.StringIO()
        with contextlib.redirect_stdout(stream):
            self.env["crawl_titles"](["Data Scientist"])
        self.assertNotIn("SECRET_TEST_KEY", stream.getvalue())
        self.assertEqual(len(self.env["client"].calls), 1)
        self.assertEqual(self.env["load_attempts"]()[0]["status"], "request_error")

    def test_quota_error_stops_all_titles(self):
        class QuotaError(Exception):
            status_code = 429

        self.env["client"] = FakeClient([QuotaError("Sensitive URL")])
        with self.assertRaisesRegex(RuntimeError, "HTTP 429"):
            self.run_crawl(["Data Scientist", "Data Analyst"])
        self.assertEqual(len(self.env["client"].calls), 1)

    def test_request_budget_is_persistent(self):
        self.env["MAX_SEARCH_ATTEMPTS"] = 1
        self.env["client"] = FakeClient([response(1, "page-2")])
        with self.assertRaisesRegex(RuntimeError, "budget"):
            self.run_crawl(["Data Scientist"])
        self.assertEqual(len(self.env["load_attempts"]()), 1)
        self.assertEqual(len(self.env["client"].calls), 1)

    def test_saved_credentials_are_redacted_but_job_data_is_preserved(self):
        self.env["API_KEY"] = "SECRET_TEST_KEY"
        value = response()
        value["api_key"] = "SECRET_TEST_KEY"
        value["search_parameters"] = {"api_key": "SECRET_TEST_KEY"}
        value["url"] = "https://example.test/search?api_key=SECRET_TEST_KEY&q=test"
        self.env["client"] = FakeClient([value])
        self.run_crawl(["Data Scientist"])
        path, saved = self.env["saved_pages"]()[0]
        self.assertNotIn("SECRET_TEST_KEY", path.read_text())
        self.assertNotIn("api_key", saved)
        self.assertEqual(saved["jobs_results"], value["jobs_results"])

    def test_repeated_token_stops_and_is_reported(self):
        self.env["client"] = FakeClient([response(1, "same"), response(2, "same")])
        self.run_crawl(["Data Scientist"])
        self.assertEqual(len(self.env["client"].calls), 2)
        self.assertEqual(self.env["crawl_summary"](["Data Scientist"])[0]["status"], "pagination_loop")

    def test_instructor_example_is_not_counted(self):
        self.env["save_json"](self.env["DATA_DIR"] / "example-out-test.json", response())
        self.assertEqual(self.env["saved_pages"](), [])

    def test_corrupt_cache_stops_before_any_network_request(self):
        folder = self.env["DATA_DIR"]
        folder.mkdir()
        (folder / "broken.json").write_text("{", encoding="utf-8")
        self.env["client"] = FakeClient([])
        with self.assertRaises(json.JSONDecodeError):
            self.run_crawl(["Data Scientist"])
        self.assertEqual(self.env["client"].calls, [])

    def test_safe_filenames(self):
        self.assertEqual(self.env["slugify"]("Research Scientist (AI/ML)"), "research-scientist-ai-ml")


if __name__ == "__main__":
    unittest.main()
