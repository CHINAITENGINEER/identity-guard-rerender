from __future__ import annotations

import unittest
from unittest.mock import patch

from tools.frontend_full_batch import prepare_group, require_accepted_layout


class _ClickTarget:
    def click(self) -> None:
        return None


class _FakePage:
    def locator(self, _selector: str) -> _ClickTarget:
        return _ClickTarget()


class FrontendFullBatchTests(unittest.TestCase):
    def test_accepted_layout_mode_can_enter_compose(self) -> None:
        state = {"mode": "智能构图完成", "status": "布局门禁通过"}
        self.assertIs(require_accepted_layout(state), state)

    def test_rejected_layout_draft_fails_closed(self) -> None:
        state = {
            "mode": "自动构图未通过",
            "status": "已生成可调整草案，但桌面承托门禁拒绝。",
        }
        with self.assertRaisesRegex(RuntimeError, "拒绝将可调整草案"):
            require_accepted_layout(state)

    def test_missing_layout_result_fails_closed(self) -> None:
        with self.assertRaisesRegex(RuntimeError, r"mode=\(empty\)"):
            require_accepted_layout({"status": "智能构图请求异常结束"})

    def test_prepare_group_does_not_return_rejected_layout(self) -> None:
        rejected = {
            "deep_scene_ready": True,
            "mode": "自动构图未通过",
            "status": "承托或前沿距离门禁拒绝",
        }
        with (
            patch(
                "tools.frontend_full_batch.wait_busy_cycle",
                side_effect=[rejected, rejected],
            ),
            patch(
                "tools.frontend_full_batch.ensure_cutouts",
                return_value=rejected,
            ),
            self.assertRaisesRegex(RuntimeError, "批处理已停止"),
        ):
            prepare_group(_FakePage())


if __name__ == "__main__":
    unittest.main()
