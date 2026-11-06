from pathlib import Path
import unittest

from highway_cli.baseline_runner import (
    build_baseline_manifest,
    planned_commands,
    validate_manifest,
)


class BaselineRunnerTests(unittest.TestCase):
    def test_nde_manifest_uses_distinct_condition_and_cli_policy(self) -> None:
        manifest = build_baseline_manifest(
            nade_repo=Path("nade-public"),
            results_dir=Path("results"),
            seeds=[1000, 1001],
            method="nde",
            protocol_status="frozen_authorized",
        )

        validate_manifest(manifest)
        self.assertEqual("nde", manifest["method"])
        self.assertEqual("nde", manifest["tasks"][0]["condition_id"])
        self.assertTrue(manifest["tasks"][0]["task_id"].endswith("__nde"))
        command = planned_commands(manifest, limit=1)[0]
        policy_index = command.index("--background-policy")
        self.assertEqual("nde", command[policy_index + 1])

    def test_existing_nade_manifest_without_method_remains_valid(self) -> None:
        manifest = build_baseline_manifest(
            nade_repo=Path("nade-public"),
            results_dir=Path("results"),
            seeds=[1000],
        )
        manifest.pop("method")

        validate_manifest(manifest)
        command = planned_commands(manifest, limit=1)[0]
        policy_index = command.index("--background-policy")
        self.assertEqual("nade", command[policy_index + 1])

    def test_idm_mobil_manifest_passes_cav_policy_to_cli(self) -> None:
        manifest = build_baseline_manifest(
            nade_repo=Path("nade-public"),
            results_dir=Path("results"),
            seeds=[1000],
            cav_policy="idm-mobil",
        )
        command = planned_commands(manifest, limit=1)[0]
        self.assertEqual("idm-mobil", command[command.index("--cav-policy") + 1])


if __name__ == "__main__":
    unittest.main()
