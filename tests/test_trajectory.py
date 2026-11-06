import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from highway_cli.trajectory_audit import analyze_episode, summarize_trajectories
from highway_cli.trajectory_trace import capture_environment, ensure_vehicle_ids


def vehicle(identifier, x, lane, speed, *, crashed=False, selected=False):
    return {
        "vehicle_id": identifier,
        "role": "selected_background_vehicle" if selected else "background_vehicle",
        "x_m": float(x),
        "y_m": float(lane * 4),
        "relative_longitudinal_m": float(x),
        "relative_lateral_m": float(lane * 4),
        "lane_index": lane,
        "speed_mps": float(speed),
        "acceleration_mps2": 0.0,
        "heading_rad": 0.0,
        "crashed": crashed,
    }


def row(step, av, bv, *, crash=False, action="-2.0"):
    pre = {"simulation_time_s": float(step), "vehicles": [av, bv], "crashed_vehicle_ids": []}
    post_av = dict(av, crashed=crash)
    post_bv = dict(bv, crashed=crash)
    post = {
        "simulation_time_s": float(step + 1),
        "vehicles": [post_av, post_bv],
        "crashed_vehicle_ids": ["AV", "BV-001"] if crash else [],
    }
    return {
        "trajectory_version": "highway-trajectory-v1",
        "seed": 7,
        "episode": 0,
        "episode_id": "nade-7-0",
        "decision_step": step,
        "max_nade_criticality": 0.5,
        "selected_actions": [
            {"vehicle_id": "BV-001", "candidate_index": 0, "action": action}
        ],
        "cav_action": 1,
        "pre": pre,
        "post": post,
        "done": crash,
        "scene_type": "AV-Crash" if crash else None,
    }


class TrajectoryTests(unittest.TestCase):
    def test_capture_assigns_stable_ids_and_roles(self):
        av = SimpleNamespace(
            position=(10.0, 0.0), lane_index=(0, 0, 0), velocity=20.0,
            action={"acceleration": 0.2}, heading=0.0, crashed=False,
        )
        bv = SimpleNamespace(
            position=(15.0, 4.0), lane_index=(0, 0, 1), velocity=18.0,
            action={"acceleration": -1.0}, heading=0.0, crashed=False,
        )
        env = SimpleNamespace(
            vehicle=av, controlled_bvs=[bv], road=SimpleNamespace(vehicles=[av, bv]), time=3.0
        )
        identifiers = {}
        ensure_vehicle_ids(env, identifiers)
        first = capture_environment(env, identifiers, selected_vehicle_ids=["BV-001"])
        second = capture_environment(env, identifiers, selected_vehicle_ids=["BV-001"])
        self.assertEqual(first, second)
        self.assertEqual(["AV", "BV-001"], [item["vehicle_id"] for item in first["vehicles"]])
        self.assertEqual("selected_background_vehicle", first["vehicles"][1]["role"])

    def test_audit_classifies_rear_end_collision(self):
        rows = [
            row(0, vehicle("AV", 0, 0, 30), vehicle("BV-001", 10, 0, 20, selected=True)),
            row(
                1,
                vehicle("AV", 8, 0, 30),
                vehicle("BV-001", 11, 0, 20, selected=True),
                crash=True,
                action="-4.0",
            ),
        ]
        result = analyze_episode(rows)
        self.assertTrue(result["crash"])
        self.assertEqual("av_rear_ended_background_vehicle", result["collision_type"])
        self.assertTrue(result["collision_partner_was_selected"])
        self.assertAlmostEqual(0.3, result["min_partner_ttc_s"])

    def test_summary_reads_multiple_files(self):
        crash_rows = [
            row(
                0,
                vehicle("AV", 0, 0, 30),
                vehicle("BV-001", 3, 0, 20, selected=True),
                crash=True,
            )
        ]
        safe = row(0, vehicle("AV", 0, 0, 20), vehicle("BV-001", 30, 0, 20))
        safe["done"] = True
        safe["scene_type"] = "AV-Finish-Test"
        with tempfile.TemporaryDirectory() as temporary:
            paths = []
            for name, rows in (("crash.jsonl", crash_rows), ("safe.jsonl", [safe])):
                path = Path(temporary) / name
                path.write_text(
                    "".join(json.dumps(item) + "\n" for item in rows), encoding="utf-8"
                )
                paths.append(path)
            result = summarize_trajectories(paths)
        self.assertEqual(2, result["episodes"])
        self.assertEqual(1, result["collisions"])
        self.assertEqual(0.5, result["collision_rate"])


if __name__ == "__main__":
    unittest.main()
