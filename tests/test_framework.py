import json
import subprocess
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from highway_cli.actions import nade_action_space
from highway_cli.benchmark_eval import (
    read_jsonl,
    score_records,
    summarize_horizon_ablation,
    summarize_representation_sizes,
    summarize_records,
    summarize_strata,
)
from highway_cli.benchmark_predict import predict_benchmark
from highway_cli.benchmark_split import collection_seed, prepare_seed_splits
from highway_cli.branch_rollout import (
    aggregate_repeated_rollouts,
    compare_rollout_rankings,
    rank_rollout_outcomes,
)
from highway_cli.demo import adversarial_specification, demo_state
from highway_cli.coverage_audit import audit_coverage, coverage_features
from highway_cli.representations import (
    SYMMETRIC_MANEUVER_EVIDENCE_V1,
    SYMMETRIC_MANEUVER_EVIDENCE_V2,
)
from highway_cli.providers.mock import MockProvider
from highway_cli.providers.jev import JevProvider
from highway_cli.providers.microsoft_decision import MicrosoftDecisionProvider
from highway_cli.providers.hierarchical_openrouter import (
    HierarchicalOpenRouterDecisionsProvider,
)
from highway_cli.providers.intent_openrouter import (
    IntentHierarchicalOpenRouterDecisionsProvider,
)
from highway_cli.providers.openai_compatible import (
    ModelOutputError,
    OpenAICompatibleProvider,
)
from highway_cli.providers.http import _safe_http_detail
from highway_cli.providers.replay import ReplayProvider
from highway_cli.nade_adapter import audit_nade_native_slots, encode_nade_state
from highway_cli.proposal import importance_ratio, mixed_proposal
from highway_cli.pilot_selection import select_pilot
from highway_cli.label_stability import assess_label_stability
from highway_cli.hierarchical_actions import (
    CHANGE_LEFT,
    CHANGE_RIGHT,
    KEEP_LANE,
    analyze_hierarchical_records,
    hierarchical_action_structure,
    maneuver_for_action,
)
from highway_cli.soft_label_eval import score_soft_records
from highway_cli.input_quality_eval import compare_input_quality_records
from highway_cli.interaction_graph import audit_interaction_graph_intent
from highway_cli.maneuver_eval import compare_maneuver_predictions, score_maneuver_records
from highway_cli.representations import (
    ACTION_CONDITIONED_V1,
    KINEMATIC_V1,
    RELATIONAL_V1,
    REPRESENTATION_PROFILES,
    RISK_AWARE_V1,
    TEMPORAL_ACTION_CONDITIONED_V1,
    TEMPORAL_RELATIONAL_V1,
    HIERARCHICAL_RELATIONAL_V1,
    INTERACTION_GRAPH_INTENT_V1,
    NADE_NATIVE_V1,
    POST_MANEUVER_CONFLICT_V1,
    POST_MANEUVER_CONFLICT_V2,
    STAGE_ROUTED_CONFLICT_V1,
)
from highway_cli.representation_audit import audit_records, audit_state
from highway_cli.tracing import append_trace
from highway_cli.cli import _build_decision_provider, build_parser
from highway_cli.schema import DecisionSpecification, VehicleState
from highway_cli.model_config import load_model_config
from highway_cli.native_benchmark import enrich_benchmark_with_native_observations
from highway_cli.reliability import summarize_model_reliability
from highway_cli.formal_runner import (
    _run_one_task,
    build_formal_manifest,
    classify_trace,
    is_transport_error,
    planned_commands,
    summarize_formal_run,
    validate_manifest,
)


class OpenRouterDecisionProviderTests(unittest.TestCase):
    def _exercise_provider(self, provider, expected_model, expected_name):
        state = demo_state()
        specification = adversarial_specification(state)
        selected = specification.actions[0].value
        probabilities = {
            action.value: (1.0 if action.value == selected else 0.0)
            for action in specification.actions
        }
        response = {
            "model": expected_model + "-pinned",
            "answers": {
                specification.name: {
                    "type": "choice",
                    "choice": selected,
                    "probabilities": probabilities,
                    "confidence": 0.9,
                }
            },
            "usage": {"inputTokens": 10, "outputTokens": 1},
        }
        with patch(
            "highway_cli.providers.openrouter_decisions.post_json",
            return_value=response,
        ) as request:
            result = provider.decide(state, specification)

        endpoint, key, payload, timeout = request.call_args.args
        self.assertEqual("https://openrouter.ai/api/alpha/decisions", endpoint)
        self.assertEqual("test-key", key)
        self.assertEqual(expected_model, payload["model"])
        self.assertIsInstance(payload["state"], str)
        self.assertEqual("choice", payload["questions"][specification.name]["type"])
        self.assertEqual(set(probabilities), set(payload["questions"][specification.name]["criteria"]))
        self.assertGreater(timeout, 0)
        self.assertEqual(expected_name, result.provider)
        self.assertEqual(selected, result.action)
        self.assertAlmostEqual(1.0, sum(result.probabilities.values()))

    def test_jev_uses_openrouter_decisions_contract(self):
        self._exercise_provider(
            JevProvider(api_key="test-key"), "typesafe/jev-1.13", "jev"
        )

    def test_microsoft_decision_uses_openrouter_decisions_contract(self):
        self._exercise_provider(
            MicrosoftDecisionProvider(api_key="test-key"),
            "microsoft/microsoft-decision-1",
            "decision-1",
        )

    def test_flat_decisions_provider_normalizes_rounded_probabilities(self):
        state = demo_state()
        specification = adversarial_specification(state)
        response = {
            "model": "typesafe/jev-1.13-pinned",
            "answers": {
                specification.name: {
                    "type": "choice",
                    "choice": specification.actions[0].value,
                    "probabilities": {
                        specification.actions[0].value: 0.50,
                        specification.actions[1].value: 0.49,
                    },
                }
            },
        }
        provider = JevProvider(api_key="test-key")
        with patch(
            "highway_cli.providers.openrouter_decisions.post_json",
            return_value=response,
        ):
            result = provider.decide(state, specification)
        self.assertAlmostEqual(1.0, sum(result.probabilities.values()))
        self.assertEqual(specification.actions[0].value, result.action)

    def test_hierarchical_provider_composes_leaf_probabilities(self):
        state = demo_state()
        specification = adversarial_specification(state)
        response = {
            "model": "test-model-pinned",
            "answers": {
                "maneuver": {
                    "type": "choice",
                    "choice": KEEP_LANE,
                    "probabilities": {KEEP_LANE: 0.7, CHANGE_RIGHT: 0.3},
                },
                "longitudinal_action_given_keep_lane": {
                    "type": "choice",
                    "choice": "-4.0",
                    "probabilities": {
                        "-4.0": 0.4,
                        "-2.0": 0.3,
                        "0.0": 0.2,
                        "1.0": 0.1,
                    },
                },
            },
            "usage": {"inputTokens": 20, "outputTokens": 2},
        }
        provider = HierarchicalOpenRouterDecisionsProvider(
            "test-model", api_key="test-key"
        )
        with patch(
            "highway_cli.providers.hierarchical_openrouter.post_json",
            return_value=response,
        ) as request:
            result = provider.decide(state, specification)

        payload = request.call_args.args[2]
        self.assertEqual(
            {"maneuver", "longitudinal_action_given_keep_lane"},
            set(payload["questions"]),
        )
        self.assertEqual("-4.0", result.action)
        self.assertAlmostEqual(0.3, result.probabilities["LANE_RIGHT"])
        self.assertAlmostEqual(0.28, result.probabilities["-4.0"])
        self.assertAlmostEqual(1.0, sum(result.probabilities.values()))

    def test_hierarchical_provider_skips_trivial_maneuver_question(self):
        data = demo_state().to_dict()
        data["available_actions"] = [
            item for item in data["available_actions"] if item["kind"] == "longitudinal"
        ]
        state = type(demo_state()).from_dict(data)
        specification = adversarial_specification(state)
        probabilities = {
            action.value: (1.0 if action.value == "0.0" else 0.0)
            for action in specification.actions
        }
        response = {
            "model": "test-model-pinned",
            "answers": {
                "longitudinal_action_given_keep_lane": {
                    "type": "choice",
                    "choice": "0.0",
                    "probabilities": probabilities,
                }
            },
        }
        provider = HierarchicalOpenRouterDecisionsProvider(
            "test-model", api_key="test-key"
        )
        with patch(
            "highway_cli.providers.hierarchical_openrouter.post_json",
            return_value=response,
        ) as request:
            result = provider.decide(state, specification)
        self.assertNotIn("maneuver", request.call_args.args[2]["questions"])
        self.assertEqual("0.0", result.action)

    def test_hierarchical_provider_skips_single_longitudinal_choice(self):
        data = demo_state().to_dict()
        data["available_actions"] = [
            item
            for item in data["available_actions"]
            if item["value"] in {"LANE_RIGHT", "0.0"}
        ]
        state = type(demo_state()).from_dict(data)
        specification = adversarial_specification(state)
        response = {
            "model": "test-model-pinned",
            "answers": {
                "maneuver": {
                    "type": "choice",
                    "choice": KEEP_LANE,
                    "probabilities": {KEEP_LANE: 0.75, CHANGE_RIGHT: 0.25},
                }
            },
        }
        provider = HierarchicalOpenRouterDecisionsProvider(
            "test-model", api_key="test-key"
        )
        with patch(
            "highway_cli.providers.hierarchical_openrouter.post_json",
            return_value=response,
        ) as request:
            result = provider.decide(state, specification)

        payload = request.call_args.args[2]
        self.assertEqual({"maneuver"}, set(payload["questions"]))
        self.assertEqual("0.0", result.action)
        self.assertAlmostEqual(0.75, result.probabilities["0.0"])
        self.assertAlmostEqual(0.25, result.probabilities["LANE_RIGHT"])

    def test_intent_hierarchical_provider_conditions_leaf_action(self):
        state = demo_state()
        specification = adversarial_specification(state)
        intent_response = {
            "model": "test-model-pinned",
            "answers": {
                "interaction_intent": {
                    "type": "choice",
                    "choice": "DECELERATION_PRESSURE",
                    "probabilities": {"DECELERATION_PRESSURE": 1.0},
                }
            },
            "usage": {"inputTokens": 30, "outputTokens": 1},
        }
        action_response = {
            "model": "test-model-pinned",
            "answers": {
                "action_given_intent": {
                    "type": "choice",
                    "choice": "-2.0",
                    "probabilities": {"-4.0": 0.25, "-2.0": 0.75},
                }
            },
            "usage": {"inputTokens": 35, "outputTokens": 1},
        }
        provider = IntentHierarchicalOpenRouterDecisionsProvider(
            "test-model", api_key="test-key"
        )
        with patch(
            "highway_cli.providers.intent_openrouter.post_json",
            side_effect=[intent_response, action_response],
        ) as request:
            result = provider.decide(state, specification)

        self.assertEqual(2, request.call_count)
        first_payload = request.call_args_list[0].args[2]
        second_payload = request.call_args_list[1].args[2]
        self.assertIn("interaction_intent", first_payload["questions"])
        self.assertIn("action_given_intent", second_payload["questions"])
        conditioned = json.loads(second_payload["state"])
        self.assertEqual(
            "DECELERATION_PRESSURE",
            conditioned["selected_interaction_intent"]["intent_id"],
        )
        self.assertEqual("-2.0", result.action)
        self.assertAlmostEqual(1.0, sum(result.probabilities.values()))
        self.assertEqual(65, result.usage["input_tokens"])
        self.assertEqual(
            "DECELERATION_PRESSURE", result.raw_response["selected_intent"]
        )

    def test_intent_interface_hides_actions_outside_execution_support(self):
        state = demo_state()
        action = next(item for item in nade_action_space() if item.value == "-2.0")
        specification = DecisionSpecification(
            name="adversarial_background_action",
            instructions="Choose one executable action.",
            actions=[action],
        )
        response = {
            "model": "test-model-pinned",
            "answers": {
                "interaction_intent": {
                    "type": "choice",
                    "choice": "DECELERATION_PRESSURE",
                    "probabilities": {"DECELERATION_PRESSURE": 1.0},
                }
            },
            "usage": {"inputTokens": 10, "outputTokens": 1},
        }
        provider = IntentHierarchicalOpenRouterDecisionsProvider(
            "test-model", api_key="test-key"
        )
        with patch(
            "highway_cli.providers.intent_openrouter.post_json",
            return_value=response,
        ) as request:
            result = provider.decide(state, specification)

        representation = json.loads(request.call_args.args[2]["state"])
        intents = {
            item["intent_id"]: item
            for item in representation["intent_interface"]["intent_candidates"]
        }
        self.assertFalse(intents["RIGHT_LANE_INTERACTION"]["available"])
        self.assertEqual([], intents["RIGHT_LANE_INTERACTION"]["compatible_leaf_actions"])
        self.assertEqual("-2.0", result.action)

    def test_post_maneuver_cards_reach_intent_provider_payload(self):
        state = demo_state()
        specification = adversarial_specification(state)
        intent_response = {
            "model": "test-model-pinned",
            "answers": {
                "interaction_intent": {
                    "type": "choice",
                    "choice": "RIGHT_LANE_INTERACTION",
                    "probabilities": {"RIGHT_LANE_INTERACTION": 1.0},
                }
            },
            "usage": {},
        }
        provider = IntentHierarchicalOpenRouterDecisionsProvider(
            "test-model",
            api_key="test-key",
            representation_profile=POST_MANEUVER_CONFLICT_V1,
        )
        with patch(
            "highway_cli.providers.intent_openrouter.post_json",
            return_value=intent_response,
        ) as request:
            result = provider.decide(state, specification)
        payload = request.call_args.args[2]
        representation = json.loads(payload["state"])
        self.assertIn("post_maneuver_conflict_interface", representation)
        self.assertIn(
            "post-maneuver conflict cards",
            payload["questions"]["interaction_intent"]["instructions"],
        )
        self.assertEqual("LANE_RIGHT", result.action)

    def test_stage_routed_profile_uses_graph_for_leaf_choice(self):
        state = demo_state()
        specification = adversarial_specification(state)
        intent_response = {
            "model": "test-model-pinned",
            "answers": {
                "interaction_intent": {
                    "type": "choice",
                    "choice": "DECELERATION_PRESSURE",
                    "probabilities": {"DECELERATION_PRESSURE": 1.0},
                }
            },
            "usage": {},
        }
        action_response = {
            "model": "test-model-pinned",
            "answers": {
                "action_given_intent": {
                    "type": "choice",
                    "choice": "-2.0",
                    "probabilities": {"-2.0": 1.0},
                }
            },
            "usage": {},
        }
        provider = IntentHierarchicalOpenRouterDecisionsProvider(
            "test-model",
            api_key="test-key",
            representation_profile=STAGE_ROUTED_CONFLICT_V1,
        )
        with patch(
            "highway_cli.providers.intent_openrouter.post_json",
            side_effect=[intent_response, action_response],
        ) as request:
            result = provider.decide(state, specification)

        intent_state = json.loads(request.call_args_list[0].args[2]["state"])
        leaf_state = json.loads(request.call_args_list[1].args[2]["state"])
        self.assertIn("post_maneuver_conflict_interface", intent_state)
        self.assertIn("stage_evidence_routing", intent_state)
        self.assertNotIn("post_maneuver_conflict_interface", leaf_state)
        self.assertEqual(
            INTERACTION_GRAPH_INTENT_V1, leaf_state["representation_profile"]
        )
        self.assertEqual(
            INTERACTION_GRAPH_INTENT_V1,
            result.raw_response["leaf_stage_representation"],
        )


class ClosedLoopProviderConfigurationTests(unittest.TestCase):
    def test_experiment_parser_accepts_live_intent_provider(self):
        args = build_parser().parse_args(
            [
                "experiment",
                "--provider",
                "jev",
                "--decision-mode",
                "intent-hierarchical",
                "--representation",
                INTERACTION_GRAPH_INTENT_V1,
                "--report-output",
                "run.json",
            ]
        )
        self.assertEqual("jev", args.provider)
        self.assertEqual("intent-hierarchical", args.decision_mode)
        self.assertEqual("run.json", args.report_output)

    def test_live_provider_builder_uses_frozen_method(self):
        provider = _build_decision_provider(
            "decision-1",
            decision_mode="intent-hierarchical",
            representation=INTERACTION_GRAPH_INTENT_V1,
        )
        self.assertIsInstance(provider, IntentHierarchicalOpenRouterDecisionsProvider)
        self.assertEqual("decision-1-intent-hierarchical", provider.name)
        self.assertEqual("microsoft/microsoft-decision-1", provider.model)

    def test_live_provider_builder_accepts_post_maneuver_conflict(self):
        provider = _build_decision_provider(
            "jev",
            decision_mode="intent-hierarchical",
            representation=POST_MANEUVER_CONFLICT_V1,
        )
        self.assertIsInstance(provider, IntentHierarchicalOpenRouterDecisionsProvider)
        self.assertEqual(POST_MANEUVER_CONFLICT_V1, provider.representation_profile)
        v2_provider = _build_decision_provider(
            "decision-1",
            decision_mode="intent-hierarchical",
            representation=POST_MANEUVER_CONFLICT_V2,
        )
        self.assertEqual(POST_MANEUVER_CONFLICT_V2, v2_provider.representation_profile)
        routed_provider = _build_decision_provider(
            "jev",
            decision_mode="intent-hierarchical",
            representation=STAGE_ROUTED_CONFLICT_V1,
        )
        self.assertEqual(STAGE_ROUTED_CONFLICT_V1, routed_provider.representation_profile)

    def test_intent_mode_rejects_wrong_representation(self):
        with self.assertRaisesRegex(ValueError, "interaction-graph-intent-v1"):
            _build_decision_provider(
                "jev",
                decision_mode="intent-hierarchical",
                representation=HIERARCHICAL_RELATIONAL_V1,
            )

    def test_openai_compatible_builder_uses_secret_free_config(self):
        with tempfile.TemporaryDirectory() as directory:
            config = Path(directory) / "model.json"
            config.write_text(
                json.dumps(
                    {
                        "model": {
                            "adapter": "openai-compatible",
                            "model": "vendor/test-model",
                            "base_url": "https://example.test/v1",
                            "api_key_env": "TEST_MODEL_KEY",
                            "format_retries": 2,
                        }
                    }
                ),
                encoding="utf-8",
            )
            with patch.dict("os.environ", {"TEST_MODEL_KEY": "test-key"}):
                provider = _build_decision_provider(
                    "openai-compatible",
                    decision_mode="intent-hierarchical",
                    representation=INTERACTION_GRAPH_INTENT_V1,
                    config=str(config),
                )
        self.assertIsInstance(provider, OpenAICompatibleProvider)
        self.assertEqual("vendor/test-model", provider.model)
        self.assertEqual(2, provider.format_retries)

    def test_model_config_rejects_embedded_key(self):
        with tempfile.TemporaryDirectory() as directory:
            config = Path(directory) / "unsafe.json"
            config.write_text(
                json.dumps(
                    {
                        "model": "vendor/test-model",
                        "base_url": "https://example.test/v1",
                        "api_key": "must-not-be-here",
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "api_key_env"):
                load_model_config(config)

    def test_model_config_allows_explicit_keyless_local_endpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            config = Path(directory) / "local.json"
            config.write_text(
                json.dumps(
                    {
                        "model": "local-model",
                        "base_url": "http://127.0.0.1:11434/v1",
                        "api_key_required": False,
                        "api_key_env": "",
                    }
                ),
                encoding="utf-8",
            )
            loaded = load_model_config(config)
        self.assertFalse(loaded["api_key_required"])
        self.assertEqual("", loaded["api_key_env"])


class OpenAICompatibleProviderTests(unittest.TestCase):
    @staticmethod
    def _response(value, *, model="vendor/test-pinned"):
        return {
            "model": model,
            "choices": [{"message": {"content": value}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 3, "cost": 0.001},
        }

    def _provider(self, **overrides):
        options = {
            "model": "vendor/test",
            "base_url": "https://example.test/v1",
            "api_key": "test-key",
            "representation_profile": INTERACTION_GRAPH_INTENT_V1,
            "format_retries": 1,
        }
        options.update(overrides)
        return OpenAICompatibleProvider(**options)

    def test_flat_provider_returns_one_hot_action(self):
        state = demo_state()
        specification = adversarial_specification(state)
        provider = self._provider(decision_mode="flat")
        with patch(
            "highway_cli.providers.openai_compatible.post_json",
            return_value=self._response('{"selected_action":"-2.0"}'),
        ) as request:
            result = provider.decide(state, specification)
        self.assertEqual(
            "https://example.test/v1/chat/completions", request.call_args.args[0]
        )
        self.assertEqual("-2.0", result.action)
        self.assertEqual(1.0, result.probabilities["-2.0"])
        self.assertTrue(result.raw_response["reliability"]["first_pass_valid"])
        self.assertEqual(1, result.usage["api_calls"])

    def test_keyless_local_provider_passes_empty_key_to_transport(self):
        provider = self._provider(
            decision_mode="flat", api_key="", api_key_env="", api_key_required=False
        )
        with patch(
            "highway_cli.providers.openai_compatible.post_json",
            return_value=self._response('{"selected_action":"-2.0"}'),
        ) as request:
            result = provider.decide(demo_state(), adversarial_specification(demo_state()))
        self.assertEqual("-2.0", result.action)
        self.assertEqual("", request.call_args.args[1])

    def test_http_error_detail_drops_account_metadata(self):
        detail = _safe_http_detail(
            json.dumps(
                {
                    "error": {"message": "blocked", "code": 403},
                    "user_id": "must-not-leak",
                }
            )
        )
        self.assertIn("blocked", detail)
        self.assertNotIn("must-not-leak", detail)

    def test_invalid_first_response_is_retried_and_audited(self):
        state = demo_state()
        specification = adversarial_specification(state)
        provider = self._provider(decision_mode="flat")
        with patch(
            "highway_cli.providers.openai_compatible.post_json",
            side_effect=[
                self._response("I choose -2.0"),
                self._response('{"selected_action":"-2.0"}'),
            ],
        ) as request:
            result = provider.decide(state, specification)
        reliability = result.raw_response["reliability"]
        self.assertEqual(2, request.call_count)
        self.assertFalse(reliability["first_pass_valid"])
        self.assertTrue(reliability["eventual_valid"])
        self.assertEqual(2, result.usage["api_calls"])

    def test_exhausted_format_retries_preserve_audit(self):
        provider = self._provider(decision_mode="flat", format_retries=1)
        with patch(
            "highway_cli.providers.openai_compatible.post_json",
            return_value=self._response("not json"),
        ):
            with self.assertRaises(ModelOutputError) as raised:
                provider.decide(demo_state(), adversarial_specification(demo_state()))
        self.assertEqual(2, raised.exception.usage["api_calls"])
        self.assertFalse(raised.exception.audit["stages"][0]["eventual_valid"])

    def test_intent_provider_uses_two_structured_stages(self):
        provider = self._provider(decision_mode="intent-hierarchical")
        responses = [
            self._response('{"selected_intent":"DECELERATION_PRESSURE"}'),
            self._response('{"selected_action":"-2.0"}'),
        ]
        with patch(
            "highway_cli.providers.openai_compatible.post_json",
            side_effect=responses,
        ) as request:
            result = provider.decide(
                demo_state(), adversarial_specification(demo_state())
            )
        self.assertEqual(2, request.call_count)
        self.assertEqual("-2.0", result.action)
        self.assertEqual(
            "DECELERATION_PRESSURE", result.raw_response["selected_intent"]
        )
        self.assertEqual(2, len(result.raw_response["reliability"]["stages"]))

    def test_reliability_summary_separates_first_pass_retry_and_fallback(self):
        with tempfile.TemporaryDirectory() as directory:
            trace = Path(directory) / "trace.jsonl"
            rows = [
                {
                    "result": {
                        "fallback_used": False,
                        "usage": {"api_calls": 1},
                        "raw_response": {
                            "reliability": {
                                "first_pass_valid": True,
                                "eventual_valid": True,
                                "stages": [
                                    {
                                        "stage": "intent",
                                        "first_pass_valid": True,
                                        "attempt_count": 1,
                                    }
                                ],
                            }
                        },
                    }
                },
                {
                    "result": {
                        "fallback_used": False,
                        "usage": {"api_calls": 2},
                        "raw_response": {
                            "reliability": {
                                "first_pass_valid": False,
                                "eventual_valid": True,
                                "stages": [
                                    {
                                        "stage": "intent",
                                        "first_pass_valid": False,
                                        "attempt_count": 2,
                                    }
                                ],
                            }
                        },
                    }
                },
            ]
            trace.write_text(
                "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
            )
            report = summarize_model_reliability([trace])
        self.assertEqual(2, report["decisions"])
        self.assertEqual(0.5, report["first_pass_valid_rate"])
        self.assertEqual(1, report["format_retry_calls"])


class FormalRunnerTests(unittest.TestCase):
    def test_formal_command_passes_idm_mobil_cav_policy(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = build_formal_manifest(
                nade_repo=root / "nade",
                results_dir=root / "results",
                seeds=[1000],
                cav_policy="idm-mobil",
            )
            command = planned_commands(manifest, limit=1)[0]
        self.assertEqual("idm-mobil", command[command.index("--cav-policy") + 1])

    def test_custom_model_condition_is_embedded_in_manifest_and_command(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            condition = {
                "condition_id": "llm-candidate",
                "provider": "openai-compatible",
                "decision_mode": "intent-hierarchical",
                "representation": INTERACTION_GRAPH_INTENT_V1,
                "model": "vendor/model-version",
                "base_url": "https://example.test/v1",
                "api_key_env": "MODEL_KEY",
                "format_retries": 1,
                "api_key_required": False,
            }
            manifest = build_formal_manifest(
                nade_repo=root / "nade",
                results_dir=root / "results",
                seeds=[1000],
                conditions=[condition],
            )
            command = planned_commands(manifest, limit=1)[0]
        self.assertEqual(1, manifest["task_count"])
        self.assertIn("--base-url", command)
        self.assertIn("https://example.test/v1", command)
        self.assertIn("--format-retries", command)
        self.assertIn("--no-api-key", command)
        self.assertNotIn("test-key", json.dumps(manifest))

    def test_manifest_is_seed_major_and_paired(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = build_formal_manifest(
                nade_repo=root / "nade",
                results_dir=root / "results",
                seeds=[100, 101],
            )
        self.assertEqual(8, manifest["task_count"])
        self.assertEqual([100] * 4 + [101] * 4, [t["seed"] for t in manifest["tasks"]])
        self.assertEqual(
            [
                "jev-reference",
                "jev-candidate",
                "decision1-reference",
                "decision1-candidate",
            ],
            [t["condition_id"] for t in manifest["tasks"][:4]],
        )
        validate_manifest(manifest)

    def test_formal_command_includes_optional_pinned_model(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = build_formal_manifest(
                nade_repo=root / "nade",
                results_dir=root / "results",
                seeds=[100],
            )
            manifest["tasks"][0]["model"] = "typesafe/jev-1.13-pinned"
            command = planned_commands(manifest, limit=1)[0]
        self.assertIn("--model", command)
        self.assertEqual(
            "typesafe/jev-1.13-pinned", command[command.index("--model") + 1]
        )

    def test_trace_classification_separates_transport_and_model_failures(self):
        rows = [
            {"result": {"fallback_used": False, "error": None}},
            {
                "result": {
                    "fallback_used": True,
                    "error": "ProviderHTTPError: timed out",
                }
            },
            {
                "result": {
                    "fallback_used": True,
                    "error": "DecisionValidationError: invalid action",
                }
            },
        ]
        with tempfile.TemporaryDirectory() as temporary:
            trace = Path(temporary) / "trace.jsonl"
            trace.write_text(
                "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
            )
            result = classify_trace(trace)
        self.assertEqual(3, result["trace_rows"])
        self.assertEqual(1, len(result["transport_fallbacks"]))
        self.assertEqual(1, len(result["semantic_fallbacks"]))
        self.assertTrue(is_transport_error("[WinError 10013] socket access denied"))
        self.assertFalse(
            is_transport_error(
                "ProviderHTTPError: HTTP 400: unsupported_request_argument"
            )
        )

    def test_status_is_resumable_from_task_results(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = build_formal_manifest(
                nade_repo=root / "nade",
                results_dir=root / "results",
                seeds=[100],
            )
            task = manifest["tasks"][0]
            task_dir = root / "results" / "tasks" / task["task_id"]
            task_dir.mkdir(parents=True)
            (task_dir / "result.json").write_text(
                json.dumps(
                    {
                        "status": "completed",
                        "observed_cost_usd": 0.001,
                    }
                ),
                encoding="utf-8",
            )
            summary = summarize_formal_run(manifest)
        self.assertEqual(1, summary["completed_task_count"])
        self.assertEqual(3, summary["status_counts"]["pending"])
        self.assertAlmostEqual(0.001, summary["observed_successful_call_cost_usd"])

    def test_formal_cli_defaults_to_non_executing_preview(self):
        args = build_parser().parse_args(
            ["run-formal", "--manifest", "formal.json"]
        )
        self.assertFalse(args.execute)
        self.assertEqual(2, args.workers)

    def test_network_failure_is_retained_then_same_seed_is_retried(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = build_formal_manifest(
                nade_repo=root / "nade",
                results_dir=root / "results",
                seeds=[123],
            )
            task = manifest["tasks"][0]
            calls = []

            def fake_run(command, **_kwargs):
                calls.append(command)
                report_path = Path(command[command.index("--report-output") + 1])
                trace_path = Path(command[command.index("--decision-trace") + 1])
                report_path.parent.mkdir(parents=True, exist_ok=True)
                is_first = len(calls) == 1
                trace_row = {
                    "result": {
                        "fallback_used": is_first,
                        "error": "ProviderHTTPError: timed out" if is_first else None,
                    }
                }
                trace_path.write_text(json.dumps(trace_row) + "\n", encoding="utf-8")
                report_path.write_text(
                    json.dumps(
                        {
                            "provider_requests": 1,
                            "provider_cost_usd_successful": 0.001,
                        }
                    ),
                    encoding="utf-8",
                )
                return subprocess.CompletedProcess(command, 0, "", "")

            with patch("highway_cli.formal_runner.subprocess.run", side_effect=fake_run):
                result = _run_one_task(
                    manifest,
                    task,
                    project_dir=root,
                    network_retries=2,
                    task_timeout_s=30,
                )
            self.assertEqual("completed", result["status"])
            self.assertEqual(2, result["accepted_attempt"])
            self.assertEqual(2, len(calls))
            self.assertEqual(
                calls[0][calls[0].index("--seed") + 1],
                calls[1][calls[1].index("--seed") + 1],
            )
            task_dir = root / "results" / "tasks" / task["task_id"]
            first = json.loads(
                (task_dir / "attempt-01" / "attempt.json").read_text(encoding="utf-8")
            )
            self.assertEqual("excluded_transport_failure", first["disposition"])

            with patch("highway_cli.formal_runner.subprocess.run") as rerun:
                resumed = _run_one_task(
                    manifest,
                    task,
                    project_dir=root,
                    network_retries=2,
                    task_timeout_s=30,
                )
            self.assertEqual("completed", resumed["status"])
            rerun.assert_not_called()


class ActionSpaceTests(unittest.TestCase):
    def test_matches_public_nade_layout(self):
        actions = nade_action_space()
        self.assertEqual(33, len(actions))
        self.assertEqual("LANE_LEFT", actions[0].value)
        self.assertEqual("LANE_RIGHT", actions[1].value)
        self.assertEqual("-4.0", actions[2].value)
        self.assertEqual("2.0", actions[-1].value)
        self.assertEqual(list(range(33)), [action.index for action in actions])

    def test_hierarchical_action_mapping_is_lossless(self):
        actions = nade_action_space()
        hierarchy = hierarchical_action_structure(actions)
        leaves = [
            item["value"]
            for values in hierarchy["second_stage_actions"].values()
            for item in values
        ]
        self.assertEqual({action.value for action in actions}, set(leaves))
        self.assertEqual(KEEP_LANE, maneuver_for_action("-1.0"))
        self.assertEqual(CHANGE_LEFT, maneuver_for_action("LANE_LEFT"))
        self.assertEqual(CHANGE_RIGHT, maneuver_for_action("LANE_RIGHT"))


class HierarchicalActionAnalysisTests(unittest.TestCase):
    def test_maneuver_consensus_can_absorb_acceleration_near_ties(self):
        state = demo_state().to_dict()

        def outcomes(best, second):
            values = [best, second] + [
                action["value"]
                for action in state["available_actions"]
                if action["value"] not in {best, second}
            ]
            return [
                {
                    "action": action,
                    "criticality_rank": rank,
                    "kind": (
                        "lateral" if action.startswith("LANE_") else "longitudinal"
                    ),
                }
                for rank, action in enumerate(values, start=1)
            ]

        row = {
            "state": state,
            "horizon_action_outcomes": outcomes("-4.0", "0.0"),
            "horizon_repetition_outcomes": [
                outcomes("-4.0", "0.0"),
                outcomes("0.0", "-4.0"),
            ],
            "horizon_stability": {"aggregate_best_action": "-4.0"},
        }
        report = analyze_hierarchical_records([row])
        self.assertEqual(1.0, report["lossless_mapping_rate"])
        self.assertEqual(1, report["best_maneuver_counts"][KEEP_LANE])
        stability = report["repeated_rollout_stability"]
        self.assertEqual(0.5, stability["mean_exact_action_consensus_rate"])
        self.assertEqual(1.0, stability["mean_maneuver_consensus_rate"])


class CoverageAuditTests(unittest.TestCase):
    def test_state_only_coverage_features_and_gaps(self):
        state = demo_state().to_dict()
        features = coverage_features(state)
        self.assertEqual("right:ahead", features["relation_cell"])
        self.assertEqual("closing", features["gap_trend"])
        self.assertEqual("keep+right", features["maneuver_support"])
        report = audit_coverage([{"state": state}])
        self.assertEqual(1, report["examples"])
        self.assertEqual(1, report["interaction_clusters"])
        self.assertIn("same_lane:behind", report["coverage_gaps"]["unobserved_relation_cells"])
        self.assertEqual(5, report["action_coverage"]["unique_available_actions"])

    def test_coverage_progression_reports_only_new_cells(self):
        from highway_cli.coverage_audit import coverage_progression

        ahead = demo_state().to_dict()
        behind = demo_state().to_dict()
        ahead["controlled_background_vehicle"]["relative_longitudinal_m"] = 8.0
        behind["controlled_background_vehicle"]["relative_longitudinal_m"] = -8.0
        report = coverage_progression(
            [("a", [{"state": ahead}]), ("b", [{"state": behind}])]
        )
        self.assertEqual(report["stages"][0]["new_relation_cells"], ["right:ahead"])
        self.assertEqual(report["stages"][1]["new_relation_cells"], ["right:behind"])

    def test_candidate_selection_audit_locates_filter_stage(self):
        from highway_cli.coverage_audit import audit_candidate_selection

        ahead = demo_state().to_dict()
        behind = demo_state().to_dict()
        behind["controlled_background_vehicle"]["relative_longitudinal_m"] = -8.0
        records = [
            {
                "state": ahead,
                "nade_interaction_case": "bv_ahead_adjacent_lane",
                "positive_criticality": True,
                "nade_selected": True,
            },
            {
                "state": behind,
                "nade_interaction_case": "bv_behind_adjacent_lane",
                "positive_criticality": False,
                "nade_selected": False,
            },
        ]
        report = audit_candidate_selection(records)
        self.assertEqual(
            "matched_nade_case_but_zero_criticality",
            report["relation_cells"]["right:behind"]["disappearance_stage"],
        )


class DevelopmentSelectionTests(unittest.TestCase):
    def test_selection_is_label_blind_and_one_per_cluster(self):
        from highway_cli.development_selection import select_development_states

        first = {"census_version": "v", "collection_seed": 6, "state": demo_state().to_dict()}
        second = json.loads(json.dumps(first))
        second["state"]["time"]["decision_step"] = 2
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source.jsonl"
            source.write_text(
                "\n".join(json.dumps(row) for row in (first, second)) + "\n",
                encoding="utf-8",
            )
            output = Path(directory) / "selected.jsonl"
            manifest = Path(directory) / "manifest.json"
            report = select_development_states(
                [source], output_path=output, manifest_path=manifest, selection_seed=7
            )
            self.assertEqual(1, report["selected_states"])
            self.assertEqual(1, len(read_jsonl(output)))

    def test_freeze_development_rejects_duplicate_keys(self):
        from highway_cli.development_selection import freeze_development_benchmark

        row = {"collection_seed": 6, "state": demo_state().to_dict()}
        with tempfile.TemporaryDirectory() as directory:
            first = Path(directory) / "first.jsonl"
            second = Path(directory) / "second.jsonl"
            content = json.dumps(row) + "\n"
            first.write_text(content, encoding="utf-8")
            second.write_text(content, encoding="utf-8")
            with self.assertRaises(ValueError):
                freeze_development_benchmark(
                    [first, second],
                    output_path=Path(directory) / "output.jsonl",
                    manifest_path=Path(directory) / "manifest.json",
                )


class ManeuverEvidenceTests(unittest.TestCase):
    def test_cards_are_symmetric_and_include_unavailable_maneuvers(self):
        from highway_cli.maneuver_evidence import audit_maneuver_evidence

        state = demo_state()
        payload = state.to_representation(SYMMETRIC_MANEUVER_EVIDENCE_V1)
        cards = payload["maneuver_evidence_cards"]
        self.assertEqual(
            ["KEEP_LANE", "CHANGE_LEFT", "CHANGE_RIGHT"],
            [card["maneuver"] for card in cards],
        )
        self.assertEqual(set(cards[0]), set(cards[1]))
        self.assertEqual(set(cards[1]), set(cards[2]))
        report = audit_maneuver_evidence([{"state": state.to_dict()}])
        self.assertTrue(report["all_examples_have_symmetric_card_schema"])
        self.assertTrue(report["all_examples_match_original_action_availability"])

    def test_v2_separates_longitudinal_alignment_from_lane_conflict(self):
        from highway_cli.maneuver_evidence import audit_maneuver_evidence

        state = demo_state()
        payload = state.to_representation(SYMMETRIC_MANEUVER_EVIDENCE_V2)
        cards = {card["maneuver"]: card for card in payload["maneuver_evidence_cards"]}
        self.assertNotIn("nearby_vehicles", payload)
        for card in cards.values():
            self.assertIn("longitudinal_alignment_evidence", card)
            self.assertIn("av_lane_conflict_evidence", card)
            self.assertIn("non_av_target_lane_traffic_evidence", card)
        for card in cards.values():
            conflict = card["av_lane_conflict_evidence"]
            expected_same_lane = (
                card["target_lane_within_road"]
                and card["target_lane_index"] == state.autonomous_vehicle.lane_index
            )
            self.assertEqual(
                expected_same_lane, conflict["target_lane_is_av_current_lane"]
            )
            if not expected_same_lane:
                self.assertFalse(
                    conflict["direct_conflict_under_constant_lane_projection"]
                )
        self.assertEqual(
            cards[KEEP_LANE]["longitudinal_alignment_evidence"],
            cards[CHANGE_RIGHT]["longitudinal_alignment_evidence"],
        )
        report = audit_maneuver_evidence(
            [{"state": state.to_dict()}], profile=SYMMETRIC_MANEUVER_EVIDENCE_V2
        )
        self.assertTrue(report["all_examples_have_symmetric_card_schema"])

    def test_maneuver_ab_includes_by_action_space_not_label(self):
        from highway_cli.maneuver_ab import select_maneuver_ab

        row = {
            "collection_seed": 6,
            "state": demo_state().to_dict(),
            "horizon_action_outcomes": [
                {"action": "0.0", "criticality_rank": 1}
            ],
        }
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "benchmark.jsonl"
            source.write_text(json.dumps(row) + "\n", encoding="utf-8")
            output = Path(directory) / "selection.jsonl"
            manifest = Path(directory) / "manifest.json"
            report = select_maneuver_ab(
                source, output_path=output, manifest_path=manifest
            )
            self.assertEqual(1, report["selected_states"])
            self.assertFalse(report["label_used_for_inclusion"])


class SchemaTests(unittest.TestCase):
    def test_canonical_state_is_stable(self):
        state = demo_state()
        first = state.canonical_json()
        second = state.canonical_json()
        self.assertEqual(first, second)
        self.assertEqual("highway-state-v1", json.loads(first)["schema_version"])

    def test_nearby_vehicles_are_spatially_sorted(self):
        data = demo_state().to_dict()
        ids = [vehicle["vehicle_id"] for vehicle in data["nearby_vehicles"]]
        self.assertEqual(["BV-1", "BV-7"], ids)

    def test_state_dictionary_round_trip(self):
        state = demo_state()
        reconstructed = type(state).from_dict(state.to_dict())
        self.assertEqual(state.canonical_json(), reconstructed.canonical_json())


class ProviderTests(unittest.TestCase):
    def test_mock_and_replay_round_trip(self):
        state = demo_state()
        specification = adversarial_specification(state)
        result = MockProvider().decide(state, specification)
        result.validate(specification)

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trace.jsonl"
            append_trace(path, state, specification, result)
            replayed = ReplayProvider(path).decide(state, specification)
        self.assertEqual(result.action, replayed.action)
        self.assertEqual(result.probabilities, replayed.probabilities)
        self.assertEqual("replay", replayed.provider)
        self.assertEqual(0, replayed.usage["api_calls"])
        self.assertEqual(0.0, replayed.usage["cost"])
        self.assertEqual(result.provider, replayed.raw_response["replay_source"]["provider"])


class NadeAdapterTests(unittest.TestCase):
    class Vehicle:
        def __init__(self, identifier, x, y, lane, speed):
            self.id = identifier
            self.position = (x, y)
            self.lane_index = ("a", "b", lane)
            self.velocity = speed
            self.action = {"acceleration": 0.0}

    def test_live_environment_encoding(self):
        av = self.Vehicle("ignored", 100.0, 0.0, 0, 30.0)
        bv = self.Vehicle("bv-1", 115.0, 0.0, 0, 25.0)
        neighbor = self.Vehicle("bv-2", 90.0, 4.0, 1, 31.0)
        road = type("Road", (), {"vehicles": [av, bv, neighbor]})()
        env = type(
            "Env",
            (),
            {
                "vehicle": av,
                "road": road,
                "min_lane": 0,
                "max_lane": 2,
                "max_velocity": 40,
                "time": 7,
            },
        )()
        state = encode_nade_state(
            env,
            bv,
            episode_id="test",
            decision_step=7,
            criticality=0.25,
            vehicle_ids={id(bv): "BV-001", id(neighbor): "BV-002"},
        ).to_dict()
        self.assertEqual("BV-001", state["controlled_background_vehicle"]["vehicle_id"])
        self.assertEqual(3.0, state["risk"]["ttc_s"])
        self.assertEqual(32, len(state["available_actions"]))
        self.assertNotIn(
            "LANE_LEFT", [action["value"] for action in state["available_actions"]]
        )
        self.assertEqual(0.0, state["risk"]["predicted_min_distance_m"])
        self.assertEqual(
            ["BV-002"], [item["vehicle_id"] for item in state["nearby_vehicles"]]
        )

    def test_nade_native_slot_audit_matches_live_six_slot_order(self):
        av = self.Vehicle("ignored", 100.0, 0.0, 0, 30.0)
        bv = self.Vehicle("bv-1", 115.0, 4.0, 1, 25.0)
        front_same = self.Vehicle("bv-2", 130.0, 4.0, 1, 28.0)
        rear_right = self.Vehicle("bv-3", 105.0, 8.0, 2, 27.0)
        road = type("Road", (), {"vehicles": [av, bv, front_same, rear_right]})()
        env = type(
            "Env",
            (),
            {
                "vehicle": av,
                "road": road,
                "min_lane": 0,
                "max_lane": 2,
                "max_velocity": 40,
                "time": 1,
            },
        )()
        vehicle_ids = {
            id(av): "AV",
            id(bv): "BV-001",
            id(front_same): "BV-002",
            id(rear_right): "BV-003",
        }
        state = encode_nade_state(
            env,
            bv,
            episode_id="test",
            decision_step=1,
            vehicle_ids=vehicle_ids,
        )
        raw = [front_same, None, None, av, None, rear_right]
        audit = audit_nade_native_slots(state, raw, vehicle_ids=vehicle_ids)
        self.assertTrue(audit["exact_match"])
        self.assertEqual(6, audit["matched_slots"])


class NativeBenchmarkTests(unittest.TestCase):
    def test_enrichment_preserves_labels_and_requires_exact_state_match(self):
        state = demo_state().to_dict()
        native = [None, None, None, None, None, None]
        replayed = dict(state)
        replayed["nade_native_observation"] = native
        benchmark_row = {
            "collection_seed": 2,
            "state": state,
            "action_outcomes": {"0.0": {"av_crash": False}},
        }
        census_row = {"collection_seed": 2, "state": replayed}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            benchmark = root / "benchmark.jsonl"
            census = root / "census.jsonl"
            output = root / "enriched.jsonl"
            benchmark.write_text(json.dumps(benchmark_row) + "\n", encoding="utf-8")
            census.write_text(json.dumps(census_row) + "\n", encoding="utf-8")
            report = enrich_benchmark_with_native_observations(
                benchmark, [census], output_path=output
            )
            enriched = json.loads(output.read_text(encoding="utf-8"))
        self.assertEqual(1, report["states"])
        self.assertEqual(benchmark_row["action_outcomes"], enriched["action_outcomes"])
        self.assertEqual(native, enriched["state"]["nade_native_observation"])
        self.assertIn(NADE_NATIVE_V1, enriched["representations"])


class RepresentationTests(unittest.TestCase):
    def test_profiles_are_deterministic_and_distinct(self):
        state = demo_state()
        payloads = {
            profile: state.canonical_representation(profile)
            for profile in REPRESENTATION_PROFILES
        }
        self.assertEqual(len(REPRESENTATION_PROFILES), len(set(payloads.values())))
        for profile, payload in payloads.items():
            self.assertEqual(profile, json.loads(payload)["representation_profile"])
            self.assertEqual(payload, state.canonical_representation(profile))

    def test_relational_profile_adds_semantic_roles(self):
        state = demo_state()
        relational = state.to_representation(RELATIONAL_V1)
        controlled = relational["controlled_background_vehicle"]
        self.assertEqual("right_1_ahead", controlled["semantic_slot"])
        slots = [vehicle["semantic_slot"] for vehicle in relational["nearby_vehicles"]]
        self.assertIn("same_lane_ahead", slots)
        self.assertIn("right_1_behind", slots)
        self.assertNotIn("risk", relational)

    def test_nade_native_profile_reconstructs_six_subject_centered_slots(self):
        rendered = demo_state().to_representation(NADE_NATIVE_V1)
        slots = rendered["surrounding_vehicle_slots"]
        self.assertEqual(
            {
                "front_same_lane",
                "rear_same_lane",
                "front_left_lane",
                "rear_left_lane",
                "front_right_lane",
                "rear_right_lane",
            },
            set(slots),
        )
        self.assertEqual(-30.4, slots["rear_same_lane"]["relative_longitudinal_to_controlled_bv_m"])
        self.assertEqual(22.6, slots["front_left_lane"]["relative_longitudinal_to_controlled_bv_m"])
        self.assertTrue(slots["rear_left_lane"]["is_autonomous_vehicle"])
        self.assertIsNone(slots["front_same_lane"])
        self.assertIsNone(slots["front_right_lane"])
        self.assertNotIn("vehicle_id", json.dumps(rendered))
        self.assertNotIn("nade_criticality", json.dumps(rendered))
        self.assertNotIn("ttc", json.dumps(rendered).lower())
        self.assertNotIn("recent_history", rendered)

    def test_nade_native_profile_selects_nearest_actor_and_enforces_range(self):
        state = replace(
            demo_state(),
            nearby_vehicles=[
                VehicleState("near", "neighbor", 1, 20.0, 4.0, 29.0, -1.0),
                VehicleState("far", "neighbor", 1, 40.0, 4.0, 31.0, 1.0),
                VehicleState("outside", "neighbor", 1, 132.4, 4.0, 30.0, 0.0),
            ],
        )
        slots = state.to_representation(NADE_NATIVE_V1)["surrounding_vehicle_slots"]
        self.assertEqual(7.6, slots["front_same_lane"]["relative_longitudinal_to_controlled_bv_m"])
        self.assertIsNone(slots["rear_same_lane"])

    def test_risk_and_action_conditioned_profiles(self):
        state = demo_state()
        kinematic = state.to_representation(KINEMATIC_V1)
        risk_aware = state.to_representation(RISK_AWARE_V1)
        action_conditioned = state.to_representation(ACTION_CONDITIONED_V1)
        self.assertNotIn("risk", kinematic)
        self.assertEqual(2.1, risk_aware["risk"]["predicted_min_distance_m"])
        assessments = action_conditioned["candidate_action_assessments"]
        self.assertEqual(len(state.available_actions), len(assessments))
        right = next(item for item in assessments if item["action"] == "LANE_RIGHT")
        self.assertTrue(right["feasible"])
        self.assertEqual(2, right["target_lane_index"])

    def test_temporal_profiles_make_history_explicit(self):
        data = demo_state().to_dict()
        data["recent_history"] = {
            "history_available": True,
            "previous_relative_longitudinal_m": 20.0,
            "relative_longitudinal_change_m": -2.0,
            "previous_relative_speed_mps": -1.0,
            "relative_speed_change_mps": -0.5,
            "previous_lane_index": 1,
            "lane_changed": False,
        }
        state = type(demo_state()).from_dict(data)
        temporal = state.to_representation(TEMPORAL_RELATIONAL_V1)
        context = temporal["temporal_context"]
        self.assertEqual("closing", context["distance_trend"])
        self.assertEqual("1->1", context["lane_transition"])
        prospective = state.to_representation(TEMPORAL_ACTION_CONDITIONED_V1)
        self.assertEqual(
            3.0, prospective["candidate_action_assessments"][0]["horizon_s"]
        )
        self.assertNotIn("nade_criticality", prospective["risk"])
        self.assertNotIn("episode_id", prospective)
        self.assertNotIn("decision_step", prospective)

    def test_hierarchical_relational_replaces_flat_action_list(self):
        state = demo_state()
        rendered = state.to_representation(HIERARCHICAL_RELATIONAL_V1)
        self.assertNotIn("available_actions", rendered)
        hierarchy = rendered["action_hierarchy"]
        self.assertEqual(
            [KEEP_LANE, CHANGE_RIGHT],
            [item["maneuver"] for item in hierarchy["first_stage_maneuvers"]],
        )
        leaves = {
            item["value"]
            for values in hierarchy["second_stage_actions"].values()
            for item in values
        }
        self.assertEqual({action.value for action in state.available_actions}, leaves)
        self.assertEqual(1, len(rendered["lateral_maneuver_context"]))

    def test_audit_separates_label_leakage_from_privileged_fields(self):
        state = demo_state()
        reports = audit_state(state)
        self.assertTrue(
            reports[TEMPORAL_ACTION_CONDITIONED_V1]["portable_primary_candidate"]
        )
        self.assertFalse(reports[RISK_AWARE_V1]["portable_primary_candidate"])
        self.assertIn(
            "risk.nade_criticality",
            reports[RISK_AWARE_V1]["simulator_internal_paths"],
        )
        self.assertTrue(
            all(report["passes_no_label_leakage_check"] for report in reports.values())
        )

        aggregate = audit_records([{"state": state.to_dict()}])
        self.assertFalse(aggregate["direct_label_leakage_detected"])
        self.assertEqual(
            "primary_portable_representation",
            aggregate["profile_reports"][TEMPORAL_RELATIONAL_V1]["recommended_role"],
        )

    def test_interaction_graph_intent_is_lossless_and_identifier_free(self):
        state = demo_state()
        rendered = state.to_representation(INTERACTION_GRAPH_INTENT_V1)
        graph = rendered["scene_graph"]
        self.assertEqual(
            2 + len(state.nearby_vehicles), len(graph["actor_nodes"])
        )
        node_ids = {
            node["node_id"]
            for node in [*graph["actor_nodes"], *graph["lane_nodes"]]
        }
        for edge in [
            *graph["lane_edges"],
            *graph["controlled_bv_interaction_edges"],
        ]:
            self.assertIn(edge["source"], node_ids)
            self.assertIn(edge["target"], node_ids)
        intent_actions = {
            action
            for intent in rendered["intent_interface"]["intent_candidates"]
            for action in intent["compatible_leaf_actions"]
        }
        self.assertEqual(
            {action.value for action in state.available_actions}, intent_actions
        )
        serialized = json.dumps(rendered, sort_keys=True)
        self.assertNotIn("BV-1", serialized)
        self.assertNotIn("BV-7", serialized)
        self.assertTrue(
            audit_state(state)[INTERACTION_GRAPH_INTENT_V1][
                "portable_primary_candidate"
            ]
        )
        audit = audit_interaction_graph_intent([{"state": state.to_dict()}])
        self.assertTrue(audit["passes_structural_audit"])

    def test_post_maneuver_conflict_adds_symmetric_label_free_cards(self):
        state = replace(demo_state(), available_actions=nade_action_space())
        rendered = state.to_representation(POST_MANEUVER_CONFLICT_V1)
        cards = {
            item["maneuver"]: item
            for item in rendered["post_maneuver_conflict_interface"][
                "maneuver_cards"
            ]
        }
        self.assertEqual({KEEP_LANE, CHANGE_LEFT, CHANGE_RIGHT}, set(cards))
        left = cards[CHANGE_LEFT]
        self.assertTrue(left["executable"])
        self.assertEqual(0, left["target_lane_index"])
        self.assertTrue(
            left["post_maneuver_av_relation"][
                "same_lane_as_av_after_maneuver"
            ]
        )
        self.assertEqual(
            "AV",
            left["target_lane_local_traffic"]["nearest_rear_actor"][
                "actor_role"
            ],
        )
        self.assertEqual("high", left["conflict_level"])
        self.assertFalse(
            cards[CHANGE_RIGHT]["post_maneuver_av_relation"][
                "same_lane_as_av_after_maneuver"
            ]
        )
        self.assertTrue(
            audit_state(state)[POST_MANEUVER_CONFLICT_V1][
                "portable_primary_candidate"
            ]
        )
        serialized = json.dumps(rendered, sort_keys=True)
        self.assertNotIn("BV-1", serialized)
        self.assertNotIn("BV-7", serialized)

    def test_post_maneuver_pressure_envelope_uses_only_available_actions(self):
        state = demo_state()
        rendered = state.to_representation(POST_MANEUVER_CONFLICT_V1)
        cards = rendered["post_maneuver_conflict_interface"]["maneuver_cards"]
        for card in cards:
            envelope = card["follow_up_longitudinal_pressure_envelope"]
            if envelope is None:
                continue
            values = {
                item["controlled_bv_acceleration_mps2"]
                for item in envelope["representative_available_accelerations"]
            }
            available = {
                action.acceleration_mps2
                for action in state.available_actions
                if action.kind == "longitudinal"
            }
            self.assertTrue(values <= available)

    def test_post_maneuver_v2_separates_av_relevance_from_traffic_constraints(self):
        state = replace(demo_state(), available_actions=nade_action_space())
        rendered = state.to_representation(POST_MANEUVER_CONFLICT_V2)
        interface = rendered["post_maneuver_conflict_interface"]
        self.assertEqual("AV", interface["decision_scope"]["adversarial_safety_target"])
        cards = {item["maneuver"]: item for item in interface["maneuver_cards"]}
        self.assertEqual(
            "direct_high_av_interaction",
            cards[CHANGE_LEFT]["av_safety_relevance"],
        )
        self.assertIn("traffic_feasibility_constraints", cards[CHANGE_LEFT])
        self.assertNotIn("target_lane_local_traffic", cards[CHANGE_LEFT])
        self.assertTrue(
            audit_state(state)[POST_MANEUVER_CONFLICT_V2][
                "portable_primary_candidate"
            ]
        )


class ProposalTests(unittest.TestCase):
    def test_mixture_preserves_ndd_support_and_ratio(self):
        actions = nade_action_space()
        ndd = [0.0] * len(actions)
        ndd[2] = 0.25
        ndd[3] = 0.75
        proposal = mixed_proposal({actions[2].value: 1.0}, ndd, actions, ndd_mix=0.1)
        self.assertAlmostEqual(1.0, sum(proposal))
        self.assertGreater(proposal[3], 0.0)
        self.assertAlmostEqual(0.75 / proposal[3], importance_ratio(3, ndd, proposal))

    def test_provider_mass_on_impossible_action_is_removed(self):
        actions = nade_action_space()
        ndd = [0.0] * len(actions)
        ndd[2] = 1.0
        proposal = mixed_proposal({"LANE_LEFT": 1.0}, ndd, actions, ndd_mix=0.1)
        self.assertEqual(0.0, proposal[0])
        self.assertAlmostEqual(1.0, proposal[2])


class BranchRolloutTests(unittest.TestCase):
    def test_collision_then_distance_define_criticality_rank(self):
        outcomes = [
            {
                "action_index": 2,
                "av_crash": False,
                "controlled_bv_crash": False,
                "min_center_distance_m": 2.0,
                "min_ttc_s": 0.2,
            },
            {
                "action_index": 3,
                "av_crash": True,
                "controlled_bv_crash": True,
                "min_center_distance_m": 3.0,
                "min_ttc_s": 0.4,
            },
            {
                "action_index": 4,
                "av_crash": False,
                "controlled_bv_crash": False,
                "min_center_distance_m": 1.0,
                "min_ttc_s": None,
            },
        ]
        ranked = rank_rollout_outcomes(outcomes)
        by_index = {item["action_index"]: item for item in ranked}
        self.assertEqual(1, by_index[3]["criticality_rank"])
        self.assertEqual(2, by_index[4]["criticality_rank"])
        self.assertTrue(by_index[3]["is_best_action"])

    def test_repeated_rollouts_report_rank_instability(self):
        def outcome(action_index, rank, distance):
            return {
                "action_index": action_index,
                "action": str(action_index),
                "kind": "longitudinal",
                "acceleration_mps2": float(action_index),
                "criticality_rank": rank,
                "min_center_distance_m": distance,
                "min_abs_longitudinal_gap_m": distance,
                "min_ttc_s": None,
                "av_crash": False,
                "controlled_bv_crash": False,
            }

        result = aggregate_repeated_rollouts(
            [
                [outcome(2, 1, 1.0), outcome(3, 2, 2.0)],
                [outcome(2, 2, 3.0), outcome(3, 1, 1.0)],
            ]
        )
        self.assertEqual(0.5, result["stability"]["consensus_top_action_rate"])
        self.assertEqual(-1.0, result["stability"]["mean_pairwise_spearman"])
        self.assertEqual(2, result["action_outcomes"][0]["rollout_repetitions"])

    def test_horizon_comparison_reports_rank_shift(self):
        short = [
            {"action_index": 2, "action": "brake", "criticality_rank": 1},
            {"action_index": 3, "action": "hold", "criticality_rank": 2},
            {"action_index": 4, "action": "accelerate", "criticality_rank": 3},
            {"action_index": 5, "action": "fast", "criticality_rank": 4},
        ]
        long = [
            {"action_index": 2, "action": "brake", "criticality_rank": 2},
            {"action_index": 3, "action": "hold", "criticality_rank": 1},
            {"action_index": 4, "action": "accelerate", "criticality_rank": 4},
            {"action_index": 5, "action": "fast", "criticality_rank": 3},
        ]
        comparison = compare_rollout_rankings(short, long)
        self.assertFalse(comparison["top1_match"])
        self.assertEqual(2, comparison["short_best_long_horizon_rank"])
        self.assertEqual(2, comparison["top3_overlap_count"])
        self.assertAlmostEqual(0.5, comparison["top3_jaccard"])
        self.assertAlmostEqual(0.6, comparison["spearman_rank_correlation"])

    def test_horizon_ablation_matches_identical_states(self):
        state = {
            "episode_id": "e1",
            "time": {"decision_step": 2},
            "controlled_background_vehicle": {"vehicle_id": "BV-1"},
        }
        one_step = {
            "state": state,
            "action_outcomes": [
                {"action_index": 2, "action": "brake", "criticality_rank": 1},
                {"action_index": 3, "action": "hold", "criticality_rank": 2},
            ],
        }
        two_step = {
            "state": state,
            "validation_horizon_policy_steps": 2,
            "horizon_action_outcomes": [
                {"action_index": 2, "action": "brake", "criticality_rank": 2},
                {"action_index": 3, "action": "hold", "criticality_rank": 1},
            ],
        }
        report = summarize_horizon_ablation([[one_step], [two_step]])
        self.assertEqual([1, 2], report["horizons"])
        self.assertEqual(1, report["matched_examples"])
        self.assertTrue(report["all_files_fully_matched"])
        self.assertEqual(0, report["pairwise_against_one_step"]["1_vs_2"]["top1_match_rate"])
        self.assertEqual(-1.0, report["pairwise_against_one_step"]["1_vs_2"]["mean_spearman"])

    def test_benchmark_scoring_reports_rank_and_invalid_actions(self):
        benchmark = {
            "state": {"episode_id": "e1", "time": {"decision_step": 4}},
            "validation_horizon_policy_steps": 3,
            "action_outcomes": [
                {
                    "action": "brake",
                    "criticality_rank": 1,
                    "min_center_distance_m": 1.0,
                    "av_crash": True,
                    "controlled_bv_crash": True,
                },
                {
                    "action": "hold",
                    "criticality_rank": 2,
                    "min_center_distance_m": 3.0,
                    "av_crash": False,
                    "controlled_bv_crash": False,
                },
            ],
            "horizon_action_outcomes": [
                {
                    "action": "brake",
                    "criticality_rank": 2,
                    "min_center_distance_m": 3.0,
                    "av_crash": False,
                    "controlled_bv_crash": False,
                },
                {
                    "action": "hold",
                    "criticality_rank": 1,
                    "min_center_distance_m": 1.0,
                    "av_crash": True,
                    "controlled_bv_crash": True,
                },
            ],
        }
        report = score_records(
            [benchmark],
            [{"episode_id": "e1", "decision_step": 4, "action": "hold"}],
        )
        self.assertEqual(0.0, report["top1_accuracy"])
        self.assertEqual(1.0, report["top3_accuracy"])
        self.assertEqual(1.0, report["mean_normalized_rank"])
        self.assertEqual(2, report["mean_criticality_rank"])
        self.assertEqual(2.0, report["mean_distance_regret_m"])
        self.assertEqual(0.0, report["collision_choice_success_rate"])
        validation = score_records(
            [benchmark],
            [{"episode_id": "e1", "decision_step": 4, "action": "hold"}],
            target="validation",
        )
        self.assertEqual(1.0, validation["top1_accuracy"])
        self.assertEqual([3], validation["evaluation_horizon_policy_steps"])

    def test_offline_prediction_and_summary_pipeline(self):
        state = demo_state()
        actions = state.available_actions
        outcomes = [
            {
                "action": action.value,
                "kind": action.kind,
                "criticality_rank": index + 1,
                "min_center_distance_m": float(index + 1),
                "av_crash": False,
                "controlled_bv_crash": False,
            }
            for index, action in enumerate(actions)
        ]
        benchmark = {
            "benchmark_version": "highway-decision-benchmark-v1",
            "state": state.to_dict(),
            "representations": {
                profile: state.to_representation(profile)
                for profile in REPRESENTATION_PROFILES
            },
            "action_outcomes": outcomes,
        }
        with tempfile.TemporaryDirectory() as directory:
            benchmark_path = Path(directory) / "benchmark.jsonl"
            prediction_path = Path(directory) / "predictions.jsonl"
            benchmark_path.write_text(json.dumps(benchmark) + "\n", encoding="utf-8")
            report = predict_benchmark(
                benchmark_path,
                prediction_path,
                MockProvider(),
                representation_profile=RELATIONAL_V1,
            )
            prediction = read_jsonl(prediction_path)[0]
        self.assertEqual(1, report["successes"])
        self.assertEqual(RELATIONAL_V1, prediction["representation_profile"])
        self.assertEqual(
            RELATIONAL_V1, prediction["model_input"]["representation_profile"]
        )
        summary = summarize_records([benchmark])
        self.assertEqual(1, summary["examples"])
        self.assertEqual(len(actions), summary["candidate_actions_mean"])
        self.assertEqual(0.0, summary["one_step_collision_opportunity_rate"])
        self.assertEqual(0.0, summary["validation_collision_opportunity_rate"])
        sizes = summarize_representation_sizes([benchmark])
        self.assertEqual(1, sizes["examples"])
        self.assertEqual(
            set(REPRESENTATION_PROFILES), set(sizes["profiles"])
        )

    def test_benchmark_stratification_uses_validation_outcomes(self):
        state = demo_state().to_dict()
        state["risk"]["ttc_s"] = 4.0
        state["risk"]["lane_conflict"] = True
        record = {
            "state": state,
            "action_outcomes": [],
            "horizon_action_outcomes": [
                {
                    "action": "brake",
                    "criticality_rank": 1,
                    "min_center_distance_m": 1.0,
                    "av_crash": True,
                    "controlled_bv_crash": True,
                },
                {
                    "action": "hold",
                    "criticality_rank": 2,
                    "min_center_distance_m": 1.1,
                    "av_crash": False,
                    "controlled_bv_crash": False,
                },
            ],
            "horizon_validation": {"top1_match": False},
        }
        report = summarize_strata([record])
        self.assertEqual(1, report["counts"]["collision_opportunity"])
        self.assertEqual(1, report["counts"]["low_ttc"])
        self.assertEqual(1, report["counts"]["lane_conflict"])
        self.assertEqual(1, report["counts"]["one_step_three_step_reversal"])
        self.assertFalse(report["manifest"][0]["near_tie"])
        self.assertEqual(
            1,
            report["interaction_clusters"]["collision_opportunity_clusters"],
        )

    def test_seed_split_is_disjoint_and_frozen(self):
        def row(seed):
            state = demo_state().to_dict()
            state["episode_id"] = f"nade-{seed}-1"
            state["time"]["decision_step"] = seed
            return {"state": state, "action_outcomes": []}

        self.assertEqual(7, collection_seed(row(7)))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.jsonl"
            source.write_text(
                "\n".join(json.dumps(row(seed)) for seed in (2, 3)) + "\n",
                encoding="utf-8",
            )
            development = root / "development.jsonl"
            confirmation = root / "confirmation.jsonl"
            report = prepare_seed_splits(
                [source],
                development_seeds=[2],
                confirmation_seeds=[3],
                development_output=development,
                confirmation_output=confirmation,
            )
            self.assertTrue(report["seed_disjoint"])
            self.assertEqual(1, report["development"]["examples"])
            self.assertEqual(1, report["confirmation"]["examples"])
            self.assertEqual(64, len(report["development"]["sha256"]))
            with self.assertRaises(FileExistsError):
                prepare_seed_splits(
                    [source],
                    development_seeds=[2],
                    confirmation_seeds=[3],
                    development_output=development,
                    confirmation_output=confirmation,
                )

    def test_pilot_selection_deduplicates_interaction_clusters(self):
        def row(step, collision):
            state = demo_state().to_dict()
            state["episode_id"] = "nade-2-1"
            state["time"]["decision_step"] = step
            state["controlled_background_vehicle"]["vehicle_id"] = "BV-X"
            outcome = {
                "action": "brake",
                "kind": "longitudinal",
                "criticality_rank": 1,
                "min_center_distance_m": 1.0,
                "av_crash": collision,
                "controlled_bv_crash": collision,
            }
            return {
                "collection_seed": 2,
                "state": state,
                "action_outcomes": [outcome],
                "horizon_action_outcomes": [outcome],
                "horizon_validation": {"top1_match": True},
            }

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.jsonl"
            source.write_text(
                "\n".join(json.dumps(item) for item in (row(1, False), row(2, True)))
                + "\n",
                encoding="utf-8",
            )
            report = select_pilot(
                [source],
                output_path=root / "pilot.jsonl",
                manifest_path=root / "manifest.json",
                max_states=2,
                min_per_stratum=1,
            )
            self.assertEqual(1, report["source_interaction_clusters"])
            self.assertEqual(1, report["selected_examples"])
            self.assertEqual(1, report["stratum_counts"]["collision_opportunity"])

    def test_label_stability_matches_states_and_collision_repetitions(self):
        state = demo_state().to_dict()
        reference_outcomes = [
            {
                "action": "brake",
                "action_index": 0,
                "criticality_rank": 1,
                "av_crash": True,
                "controlled_bv_crash": True,
            },
            {
                "action": "hold",
                "action_index": 1,
                "criticality_rank": 2,
                "av_crash": False,
                "controlled_bv_crash": False,
            },
        ]
        repeated_outcomes = [
            {
                **reference_outcomes[0],
                "controlled_collision_rate": 1.0,
            },
            {
                **reference_outcomes[1],
                "controlled_collision_rate": 0.0,
            },
        ]
        reference = {
            "collection_seed": 1,
            "state": state,
            "horizon_action_outcomes": reference_outcomes,
        }
        repeated = {
            "collection_seed": 1,
            "state": state,
            "horizon_action_outcomes": repeated_outcomes,
            "horizon_repetition_outcomes": [reference_outcomes, reference_outcomes],
            "horizon_stability": {
                "aggregate_best_action": "brake",
                "aggregate_best_matches_consensus": True,
                "consensus_top_action": "brake",
                "consensus_top_action_rate": 1.0,
                "mean_pairwise_spearman": 1.0,
                "min_pairwise_spearman": 1.0,
                "rollout_repetitions": 2,
            },
        }
        report = assess_label_stability([reference], [repeated])
        self.assertEqual(1, report["summary"]["matched_examples"])
        self.assertEqual({"stable": 1}, report["summary"]["quality_counts"])
        self.assertEqual(
            1,
            report["summary"][
                "reference_collision_opportunity_preserved_all_repetitions"
            ],
        )
        self.assertTrue(report["details"][0]["state_identical"])
        self.assertTrue(report["details"][0]["top_action_consensus_stable"])
        self.assertTrue(report["details"][0]["full_ranking_stable"])
        self.assertTrue(
            report["details"][0]["collision_event_consistent_with_reference"]
        )

    def test_soft_label_scoring_uses_repetition_frequencies_and_probabilities(self):
        state = demo_state().to_dict()
        outcomes = [
            {
                "action": "brake",
                "criticality_rank": 1,
                "repetition_ranks": [1, 1, 1, 2, 2],
                "mean_repetition_rank": 1.4,
                "controlled_collision_rate": 0.8,
                "mean_min_center_distance_m": 1.0,
                "min_center_distance_m": 1.0,
            },
            {
                "action": "hold",
                "criticality_rank": 2,
                "repetition_ranks": [2, 2, 2, 1, 1],
                "mean_repetition_rank": 1.6,
                "controlled_collision_rate": 0.2,
                "mean_min_center_distance_m": 1.2,
                "min_center_distance_m": 1.2,
            },
        ]
        benchmark = {
            "state": state,
            "horizon_action_outcomes": outcomes,
            "horizon_stability": {
                "rollout_repetitions": 5,
                "consensus_top_action": "brake",
            },
        }
        prediction = {
            "state": state,
            "result": {
                "action": "hold",
                "probabilities": {"brake": 0.25, "hold": 0.75},
            },
        }
        report = score_soft_records([benchmark], [prediction])
        self.assertEqual(0.0, report["modal_top1_accuracy"])
        self.assertEqual(0.0, report["soft_equivalent_action_accuracy"])
        self.assertEqual(0.0, report["near_optimal_action_rate"])
        self.assertAlmostEqual(0.4, report["mean_selected_top_action_frequency"])
        self.assertAlmostEqual(0.6, report["mean_collision_probability_regret"])
        self.assertEqual(0.0, report["collision_opportunity_miss_rate"])
        self.assertEqual(0.0, report["bottom_tail_selection_rate"])
        self.assertEqual(0.0, report["severe_valid_decision_error_rate"])
        self.assertAlmostEqual(1.55, report["mean_expected_repetition_rank"])
        self.assertAlmostEqual(0.35, report["mean_expected_collision_rate"])
        self.assertAlmostEqual(0.245, report["mean_top_action_distribution_brier"])
        self.assertAlmostEqual(
            0.35, report["mean_top_action_distribution_total_variation"]
        )
        conditional_prediction = {
            "state": state,
            "result": {
                "action": "hold",
                "probabilities": {"brake": 0.25, "hold": 0.75},
                "raw_response": {
                    "leaf_probabilities_are_conditional_on_selected_intent": True
                },
            },
        }
        conditional_report = score_soft_records(
            [benchmark], [conditional_prediction]
        )
        self.assertEqual(0, conditional_report["distribution_predictions"])
        self.assertEqual(
            {"missing": 1}, conditional_report["probability_status_counts"]
        )

    def test_soft_label_scoring_reports_distance_regret_without_collision(self):
        state = demo_state().to_dict()
        outcomes = [
            {
                "action": "brake",
                "criticality_rank": 1,
                "repetition_ranks": [1, 1],
                "mean_repetition_rank": 1.0,
                "controlled_collision_rate": 0.0,
                "mean_min_center_distance_m": 1.0,
                "min_center_distance_m": 1.0,
            },
            {
                "action": "hold",
                "criticality_rank": 2,
                "repetition_ranks": [2, 2],
                "mean_repetition_rank": 2.0,
                "controlled_collision_rate": 0.0,
                "mean_min_center_distance_m": 1.4,
                "min_center_distance_m": 1.4,
            },
        ]
        benchmark = {
            "state": state,
            "horizon_action_outcomes": outcomes,
            "horizon_stability": {
                "rollout_repetitions": 2,
                "consensus_top_action": "brake",
            },
        }
        prediction = {"state": state, "result": {"action": "hold"}}
        report = score_soft_records([benchmark], [prediction])
        self.assertAlmostEqual(
            0.4, report["mean_distance_regret_m_without_collision_opportunity"]
        )
        self.assertIsNone(report["collision_opportunity_miss_rate"])
        self.assertEqual(1.0, report["bottom_tail_selection_rate"])
        self.assertEqual(1.0, report["severe_valid_decision_error_rate"])

        comparison = compare_input_quality_records(
            [benchmark],
            [{"state": state, "result": {"action": "brake"}}],
            [prediction],
            bootstrap_samples=100,
            bootstrap_seed=7,
            comparison_status="unseen_confirmation",
        )
        self.assertEqual("unseen_confirmation", comparison["comparison_status"])
        near_optimal = comparison["paired_metrics"]["near_optimal_action_rate"]
        self.assertEqual(-1.0, near_optimal["candidate_minus_baseline"])
        self.assertEqual(-1.0, near_optimal["improvement"])
        distance = comparison["paired_metrics"][
            "mean_distance_regret_m_without_collision_opportunity"
        ]
        self.assertAlmostEqual(0.4, distance["candidate_minus_baseline"])
        self.assertAlmostEqual(-0.4, distance["improvement"])

    def test_maneuver_scoring_aggregates_leaf_probabilities(self):
        state = demo_state().to_dict()
        state["available_actions"] = [
            {"index": 0, "value": "LANE_LEFT", "kind": "lateral", "description": "left"},
            {"index": 1, "value": "-2.0", "kind": "longitudinal", "description": "brake"},
            {"index": 2, "value": "0.0", "kind": "longitudinal", "description": "hold"},
        ]
        repetition_left = [
            {"action": "LANE_LEFT", "criticality_rank": 1},
            {"action": "-2.0", "criticality_rank": 2},
        ]
        repetition_keep = [
            {"action": "-2.0", "criticality_rank": 1},
            {"action": "LANE_LEFT", "criticality_rank": 2},
        ]
        benchmark = {
            "state": state,
            "horizon_action_outcomes": repetition_left,
            "horizon_repetition_outcomes": [repetition_left, repetition_left, repetition_keep],
            "horizon_stability": {"aggregate_best_action": "LANE_LEFT"},
        }
        prediction = {
            "state": state,
            "result": {
                "action": "LANE_LEFT",
                "probabilities": {"LANE_LEFT": 0.6, "-2.0": 0.3, "0.0": 0.1},
            },
        }
        report = score_maneuver_records([benchmark], [prediction])
        self.assertEqual(1.0, report["accuracy"])
        self.assertAlmostEqual(0.6, report["details"][0]["predicted_maneuver_probabilities"]["CHANGE_LEFT"])
        self.assertAlmostEqual(0.4, report["details"][0]["predicted_maneuver_probabilities"]["KEEP_LANE"])
        self.assertAlmostEqual(2 / 3, report["details"][0]["target_maneuver_distribution"]["CHANGE_LEFT"])

    def test_maneuver_comparison_counts_paired_changes(self):
        state = demo_state().to_dict()
        benchmark = {
            "state": state,
            "horizon_action_outcomes": [{"action": "LANE_LEFT", "criticality_rank": 1}],
            "horizon_stability": {"aggregate_best_action": "LANE_LEFT"},
        }
        old = {"state": state, "result": {"action": "0.0"}}
        new = {"state": state, "result": {"action": "LANE_LEFT"}}
        report = compare_maneuver_predictions([benchmark], [old], [new])
        self.assertEqual(1, report["paired"]["candidate_only_correct"])
        self.assertEqual(1, report["paired"]["net_correct_gain"])
        self.assertEqual(1.0, report["paired"]["exact_mcnemar_p_two_sided"])


if __name__ == "__main__":
    unittest.main()

