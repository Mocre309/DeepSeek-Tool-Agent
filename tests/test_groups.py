"""Renaming conversations and organising the sidebar into groups."""

from __future__ import annotations

from fastapi.testclient import TestClient


class TestRename:
    def test_rename_updates_the_list(self, client: TestClient) -> None:
        client.post("/api/chat", json={"message": "原始标题", "session_id": "r1"})
        response = client.patch("/api/sessions/r1", json={"title": "支付幂等设计"})
        assert response.status_code == 200
        assert response.json()["title"] == "支付幂等设计"

        listed = next(
            item for item in client.get("/api/sessions").json() if item["session_id"] == "r1"
        )
        assert listed["title"] == "支付幂等设计"

    def test_rename_collapses_whitespace(self, client: TestClient) -> None:
        client.patch("/api/sessions/r2", json={"title": "  多余   空格  "})
        assert client.get("/api/sessions").json()[0]["title"] == "多余 空格"

    def test_empty_title_is_rejected(self, client: TestClient) -> None:
        response = client.patch("/api/sessions/r3", json={"title": "   "})
        assert response.status_code == 400
        assert "标题" in response.json()["detail"]

    def test_rename_can_target_a_fresh_session(self, client: TestClient) -> None:
        response = client.patch("/api/sessions/brand-new", json={"title": "还没聊过"})
        assert response.status_code == 200
        assert response.json()["title"] == "还没聊过"


class TestGroups:
    def test_create_and_list(self, client: TestClient) -> None:
        assert client.post("/api/groups", json={"name": "项目 A"}).status_code == 201
        groups = client.get("/api/groups").json()
        assert [group["name"] for group in groups] == ["项目 A"]
        assert groups[0]["count"] == 0

    def test_duplicate_group_is_rejected(self, client: TestClient) -> None:
        client.post("/api/groups", json={"name": "重复组"})
        response = client.post("/api/groups", json={"name": "重复组"})
        assert response.status_code == 409

    def test_blank_group_name_is_rejected(self, client: TestClient) -> None:
        assert client.post("/api/groups", json={"name": "   "}).status_code == 422

    def test_moving_a_conversation_into_a_group(self, client: TestClient) -> None:
        client.post("/api/chat", json={"message": "归到组里", "session_id": "g1"})
        client.post("/api/groups", json={"name": "工作"})

        response = client.patch("/api/sessions/g1", json={"group_name": "工作"})
        assert response.status_code == 200
        assert response.json()["group_name"] == "工作"

        groups = client.get("/api/groups").json()
        assert groups[0]["count"] == 1

    def test_unknown_group_is_created_on_assignment(self, client: TestClient) -> None:
        client.post("/api/chat", json={"message": "直接扔进去", "session_id": "g2"})
        client.patch("/api/sessions/g2", json={"group_name": "临时组"})
        assert [group["name"] for group in client.get("/api/groups").json()] == ["临时组"]

    def test_clearing_the_group(self, client: TestClient) -> None:
        client.post("/api/chat", json={"message": "先分组", "session_id": "g3"})
        client.patch("/api/sessions/g3", json={"group_name": "待整理"})
        response = client.patch("/api/sessions/g3", json={"group_name": ""})
        assert response.json()["group_name"] == ""

    def test_delete_group_frees_its_conversations(self, client: TestClient) -> None:
        client.post("/api/chat", json={"message": "会被释放", "session_id": "g4"})
        client.post("/api/groups", json={"name": "临时"})
        client.patch("/api/sessions/g4", json={"group_name": "临时"})

        assert client.delete("/api/groups/临时").status_code == 204
        assert client.get("/api/groups").json() == []

        listed = next(
            item for item in client.get("/api/sessions").json() if item["session_id"] == "g4"
        )
        assert listed["group_name"] == ""
        # the conversation itself survives
        assert listed["turns"] == 1

    def test_rename_and_group_in_one_patch(self, client: TestClient) -> None:
        client.post("/api/chat", json={"message": "一次性改", "session_id": "g5"})
        response = client.patch(
            "/api/sessions/g5", json={"title": "新名字", "group_name": "一起改"}
        )
        body = response.json()
        assert body["title"] == "新名字"
        assert body["group_name"] == "一起改"

    def test_workspace_and_title_together(self, client: TestClient, tmp_path) -> None:
        target = tmp_path / "ws-both"
        body = client.patch(
            "/api/sessions/g6",
            json={"title": "组合写", "group_name": "组合组", "workspace_dir": str(target)},
        ).json()
        assert body["title"] == "组合写"
        assert body["group_name"] == "组合组"
        assert body["workspace_dir"] == str(target.resolve())


class TestGroupRename:
    def test_rename_moves_the_conversations_with_it(self, client: TestClient) -> None:
        client.post("/api/groups", json={"name": "旧组"})
        client.post("/api/chat", json={"message": "组里的对话", "session_id": "gr1"})
        client.patch("/api/sessions/gr1", json={"group_name": "旧组"})

        response = client.patch("/api/groups/旧组", json={"name": "新组"})
        assert response.status_code == 200
        assert response.json() == {"name": "新组", "count": 1}

        assert [group["name"] for group in client.get("/api/groups").json()] == ["新组"]
        listed = next(
            item for item in client.get("/api/sessions").json() if item["session_id"] == "gr1"
        )
        assert listed["group_name"] == "新组"

    def test_rename_to_an_existing_name_is_refused(self, client: TestClient) -> None:
        client.post("/api/groups", json={"name": "甲"})
        client.post("/api/groups", json={"name": "乙"})
        response = client.patch("/api/groups/甲", json={"name": "乙"})
        assert response.status_code == 409
        assert sorted(group["name"] for group in client.get("/api/groups").json()) == ["乙", "甲"]

    def test_rename_unknown_group_is_404(self, client: TestClient) -> None:
        response = client.patch("/api/groups/不存在", json={"name": "别的"})
        assert response.status_code == 404

    def test_rename_to_blank_is_rejected(self, client: TestClient) -> None:
        client.post("/api/groups", json={"name": "留着"})
        assert client.patch("/api/groups/留着", json={"name": "   "}).status_code == 422

    def test_rename_to_the_same_name_is_a_noop(self, client: TestClient) -> None:
        client.post("/api/groups", json={"name": "同名"})
        response = client.patch("/api/groups/同名", json={"name": "同名"})
        assert response.status_code == 200
        assert [group["name"] for group in client.get("/api/groups").json()] == ["同名"]


class TestGroupTrash:
    """Deleting a group is reversible: the membership is archived first."""

    def test_delete_archives_and_restore_reattaches(self, client: TestClient) -> None:
        client.post("/api/chat", json={"message": "分组里的对话", "session_id": "t1"})
        client.post("/api/groups", json={"name": "会被删"})
        client.patch("/api/sessions/t1", json={"group_name": "会被删"})

        assert client.delete("/api/groups/会被删").status_code == 204
        trash = client.get("/api/groups/trash").json()
        assert [(item["name"], item["count"]) for item in trash] == [("会被删", 1)]

        restored = client.post("/api/groups/restore", json={"name": "会被删"}).json()
        assert restored == {"name": "会被删", "count": 1}

        listed = next(
            item for item in client.get("/api/sessions").json() if item["session_id"] == "t1"
        )
        assert listed["group_name"] == "会被删"
        assert client.get("/api/groups/trash").json() == []

    def test_restore_unknown_group_is_404(self, client: TestClient) -> None:
        response = client.post("/api/groups/restore", json={"name": "从没删过"})
        assert response.status_code == 404

    def test_restore_into_an_existing_name_is_refused(self, client: TestClient) -> None:
        client.post("/api/groups", json={"name": "重名组"})
        client.delete("/api/groups/重名组")
        client.post("/api/groups", json={"name": "重名组"})
        response = client.post("/api/groups/restore", json={"name": "重名组"})
        assert response.status_code == 409

    def test_deleting_twice_merges_the_membership(self, client: TestClient) -> None:
        client.post("/api/chat", json={"message": "第一个", "session_id": "t2"})
        client.post("/api/chat", json={"message": "第二个", "session_id": "t3"})
        client.post("/api/groups", json={"name": "两轮"})
        client.patch("/api/sessions/t2", json={"group_name": "两轮"})
        client.delete("/api/groups/两轮")

        client.post("/api/groups", json={"name": "两轮"})
        client.patch("/api/sessions/t3", json={"group_name": "两轮"})
        client.delete("/api/groups/两轮")

        trash = client.get("/api/groups/trash").json()
        assert trash[0]["count"] == 2

        client.post("/api/groups/restore", json={"name": "两轮"})
        members = {
            item["session_id"]
            for item in client.get("/api/sessions").json()
            if item["group_name"] == "两轮"
        }
        assert members == {"t2", "t3"}


class TestNewChatInGroup:
    def test_new_session_can_be_created_directly_in_a_group(self, client: TestClient) -> None:
        """The UI creates a fresh id, then pins it to a group before the first turn."""
        client.post("/api/groups", json={"name": "待开工"})
        created = client.patch("/api/sessions/new-in-group", json={"group_name": "待开工"})
        assert created.status_code == 200
        assert created.json()["group_name"] == "待开工"

        groups = client.get("/api/groups").json()
        assert groups[0]["count"] == 1

        # it still behaves like a normal empty conversation
        history = client.get("/api/sessions/new-in-group/messages").json()
        assert history["messages"] == []
