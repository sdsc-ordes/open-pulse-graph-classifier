from fastapi.testclient import TestClient

from openpulse_graph_classifier.inference.api import app

client = TestClient(app)


def test_test():
    response = client.get("/test", headers={"X-Token": "coneofsilence"})
    assert response.status_code == 200
