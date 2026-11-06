import json
import tempfile
import unittest
from pathlib import Path

from highway_cli.background_replay import BackgroundActionReplay


class BackgroundActionReplayTests(unittest.TestCase):
    def test_loads_step_indexed_actions(self):
        row = {
            "seed": 17,
            "episode": 0,
            "decision_step": 2,
            "max_nade_criticality": 1.25,
            "selected_actions": [
                {"vehicle_id": "BV-003", "action_index": 7, "action": "-3.0"}
            ],
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trajectory.jsonl"
            path.write_text(json.dumps(row) + "\n", encoding="utf-8")
            replay = BackgroundActionReplay.from_path(path)

        actions = replay.actions_for(17, 0, 2)
        self.assertEqual(len(actions), 1)
        self.assertEqual(actions[0].vehicle_id, "BV-003")
        self.assertEqual(actions[0].action_index, 7)
        self.assertEqual(replay.criticality_for(17, 0, 2), 1.25)
        self.assertTrue(replay.has_step(17, 0, 2))
        self.assertFalse(replay.has_step(17, 0, 3))


if __name__ == "__main__":
    unittest.main()
