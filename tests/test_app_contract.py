import importlib.util
import shutil
import sqlite3
import unittest
import uuid
from pathlib import Path


FASTAPI_AVAILABLE = importlib.util.find_spec("fastapi") is not None


@unittest.skipUnless(FASTAPI_AVAILABLE, "fastapi is not installed in this environment")
class AppContractTests(unittest.TestCase):
    def setUp(self):
        repo_root = Path(__file__).resolve().parents[1]
        self.temp_dir = repo_root / "tests" / f"_tmp_app_contract_{uuid.uuid4().hex}"
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        self.addCleanup(lambda: shutil.rmtree(self.temp_dir, ignore_errors=True))
        self.db_path = self.temp_dir / "cbdb_test.sqlite"
        self._create_test_db(self.db_path)

        from app import AppSettings, create_app
        from evaluation_tasks import EvaluationTaskPool
        from fastapi.testclient import TestClient
        from feedback_store import MemoryFeedbackStore

        self.feedback_store = MemoryFeedbackStore()
        settings = AppSettings(
            preload_models=["markov"],
            required_models=["markov"],
            sample_data_path=repo_root / "data" / "sample_names.json",
            db_path=self.db_path,
            markov_artifact_path=repo_root / "missing-markov.pkl",
            lstm_weights_path=repo_root / "missing-lstm.pth",
            lstm_vocab_path=repo_root / "missing-vocab.pkl",
            transformer_weights_path=repo_root / "missing-transformer.pth",
            transformer_vocab_path=repo_root / "missing-transformer-vocab.pkl",
            eval_task_pool_path=repo_root / "data" / "sample_eval_tasks.json",
            cors_origins=["http://localhost:5173"],
            enable_dynasty_mode=True,
            enable_fim_mode=False,
            feedback_database_url=None,
        )
        self.client = TestClient(
            create_app(
                settings,
                feedback_store=self.feedback_store,
                task_pool=EvaluationTaskPool(repo_root / "data" / "sample_eval_tasks.json"),
            )
        )
        self.client.__enter__()

    def tearDown(self):
        self.client.__exit__(None, None, None)

    def _create_test_db(self, path: Path) -> None:
        connection = sqlite3.connect(path)
        cursor = connection.cursor()
        cursor.execute(
            """
            CREATE TABLE BIOG_MAIN (
                c_surname_chn TEXT,
                c_mingzi_chn TEXT,
                c_dy INTEGER
            )
            """
        )
        cursor.executemany(
            "INSERT INTO BIOG_MAIN (c_surname_chn, c_mingzi_chn, c_dy) VALUES (?, ?, ?)",
            [
                ("\u674e", "\u6e05\u8fdc", 6),
                ("\u738b", "\u6e05\u5b89", 6),
                ("\u5f20", "\u6e05\u97f5", 15),
                ("\u9648", "\u5b50\u660e", 19),
                ("\u6797", "\u660e\u8f69", 20),
            ],
        )
        connection.commit()
        connection.close()

    def test_models_endpoint_reports_markov(self):
        response = self.client.get("/models")
        payload = response.json()

        self.assertEqual(response.status_code, 200)
        self.assertTrue(any(item["model_type"] == "markov" for item in payload))

    def test_generate_accepts_markov_without_temperature(self):
        response = self.client.post(
            "/generate",
            json={"model_type": "markov", "count": 2, "seed": "\u674e"},
        )

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["model_used"], "markov")
        self.assertEqual(len(body["generations"]), 2)

    def test_generate_supports_lucky_mode_without_seed(self):
        response = self.client.post("/generate", json={"model_type": "markov", "count": 2})

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["model_used"], "markov")
        self.assertEqual(len(body["generations"]), 2)
        self.assertTrue(all(len(name) >= 2 for name in body["generations"]))

    def test_generate_rejects_markov_temperature(self):
        response = self.client.post(
            "/generate",
            json={"model_type": "markov", "count": 2, "seed": "\u674e", "temperature": 0.8},
        )

        self.assertEqual(response.status_code, 400)

    def test_generate_supports_middle_token_constraint(self):
        response = self.client.post(
            "/generate",
            json={
                "model_type": "markov",
                "count": 2,
                "seed": "\u674e",
                "constraint_text": "\u6e05",
                "constraint_position": "middle",
            },
        )

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(len(body["generations"]), 2)
        self.assertTrue(all(name[1] == "\u6e05" for name in body["generations"]))
        self.assertEqual(body["metadata"]["normalized_constraint_text"], "\u6e05")
        self.assertEqual(body["metadata"]["constraint_position"], "middle")
        self.assertGreaterEqual(body["metadata"]["constraint_support_count"], 1)

    def test_generate_returns_422_for_impossible_constraint(self):
        response = self.client.post(
            "/generate",
            json={
                "model_type": "markov",
                "count": 2,
                "seed": "\u674e",
                "constraint_text": "\u5983",
                "constraint_position": "middle",
            },
        )

        self.assertEqual(response.status_code, 422)
        self.assertIn("does not appear in the cleaned training corpus", response.json()["detail"])

    def test_historical_pattern_returns_support_metadata(self):
        response = self.client.post(
            "/generate/historical-pattern",
            json={
                "model_type": "markov",
                "count": 2,
                "seed": "\u674e",
                "fixed_token": "\u6e05",
                "position": "middle",
            },
        )

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["model_used"], "markov")
        self.assertEqual(body["support_level"], "weak")
        self.assertGreaterEqual(body["support_count"], 1)
        self.assertIn("evidence_message", body)
        self.assertTrue(all(name[1] == "\u6e05" for name in body["generations"]))

    def test_historical_pattern_none_is_controlled(self):
        response = self.client.post(
            "/generate/historical-pattern",
            json={
                "model_type": "markov",
                "count": 2,
                "fixed_token": "\u5983",
                "position": "middle",
            },
        )

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["generations"], [])
        self.assertEqual(body["support_level"], "none")
        self.assertTrue(body["search_exhausted"])

    def test_infill_returns_controlled_503_without_artifact(self):
        response = self.client.post(
            "/generate/infill",
            json={
                "count": 2,
                "fixed_token": "\u5983",
                "position": "middle",
                "temperature": 0.8,
            },
        )

        self.assertEqual(response.status_code, 503)
        self.assertIn("FIM model artifact is not available", response.json()["detail"])

    def test_dynasties_endpoint_reports_curated_options(self):
        response = self.client.get("/dynasties")
        payload = response.json()

        self.assertEqual(response.status_code, 200)
        self.assertEqual([item["code"] for item in payload], ["Tang", "Song", "Ming", "Qing"])
        self.assertTrue(payload[0]["available"])

    def test_generate_dynasty_works_when_db_is_available(self):
        response = self.client.post(
            "/generate/dynasty",
            json={"dynasty_id": 6, "count": 2, "seed": "\u674e"},
        )

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["model_used"], "markov")
        self.assertEqual(body["metadata"]["dynasty_code"], "Tang")
        self.assertEqual(len(body["generations"]), 2)

    def test_generate_dynasty_supports_token_constraint(self):
        response = self.client.post(
            "/generate/dynasty",
            json={
                "dynasty_id": 6,
                "count": 2,
                "seed": "\u674e",
                "constraint_text": "\u6e05",
                "constraint_position": "middle",
            },
        )

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertTrue(all(name[1] == "\u6e05" for name in body["generations"]))

    def test_compare_returns_available_models(self):
        response = self.client.post("/compare", json={"count": 2, "seed": "\u674e"})
        body = response.json()

        self.assertEqual(response.status_code, 200)
        self.assertEqual(body["run_id"], "portfolio_demo_v1")
        self.assertEqual(len(body["results"]), 1)
        self.assertEqual(body["results"][0]["model_type"], "markov")

    def test_eval_tasks_returns_anonymous_task(self):
        response = self.client.get("/eval/tasks", params={"task_type": "blind_pairwise"})
        body = response.json()

        self.assertEqual(response.status_code, 200)
        self.assertEqual(body["task_type"], "blind_pairwise")
        self.assertIn("left", body["presented_payload"])
        self.assertNotIn("hidden_answer", body)

    def test_feedback_records_memory_event(self):
        task_response = self.client.get("/eval/tasks", params={"task_type": "blind_pairwise"})
        task_id = task_response.json()["task_id"]

        response = self.client.post(
            "/feedback",
            json={
                "session_id": "session-test-001",
                "locale": "en",
                "surface": "evaluation_lab",
                "task_type": "blind_pairwise",
                "task_id": task_id,
                "response_label": "left",
                "response_payload": {"selected": "left"},
                "latency_ms": 1200,
            },
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "accepted")
        self.assertEqual(len(self.feedback_store.events), 1)
        self.assertEqual(self.feedback_store.events[0]["task_type"], "blind_pairwise")


if __name__ == "__main__":
    unittest.main()
