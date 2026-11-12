from unittest.mock import patch

import server


def test_evaluation_write_requires_admin_authentication():
    response = server.app.test_client().post(
        "/api/interactions/507f1f77bcf86cd799439011/evaluation",
        json={
            "correct_skill_selected": True,
            "activity_matches_emotion": True,
            "response_educationally_aligned": True,
        },
    )

    assert response.status_code == 401


def test_evaluation_summary_requires_admin_authentication():
    with patch.object(server.learning_store, "evaluation_summary") as summary:
        response = server.app.test_client().get("/api/evaluations/summary")

    assert response.status_code == 401
    summary.assert_not_called()
