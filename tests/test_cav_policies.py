import unittest

from highway_cli.cav_policies import (
    IDMMobilCAVAgent,
    nearest_longitudinal_action,
    validate_cav_policy,
)


ACTIONS = {
    0: "LANE_LEFT",
    1: "LANE_RIGHT",
    2: "-4.0",
    3: "-1.2",
    4: "0.0",
    5: "1.0",
    6: "2.0",
}


class FakeNetwork:
    @staticmethod
    def side_lanes(_lane_index):
        return []


class FakeRoad:
    def __init__(self):
        self.network = FakeNetwork()

    @staticmethod
    def neighbour_vehicles(_vehicle, _lane_index=None):
        return None, None


class FakeVehicle:
    LENGTH = 5.0

    def __init__(self, velocity=30.0):
        self.velocity = velocity
        self.lane_index = ("a", "b", 1)
        self.target_lane_index = self.lane_index
        self.road = FakeRoad()


class FakeEnv:
    def __init__(self):
        self.vehicle = FakeVehicle()


class CAVPolicyTests(unittest.TestCase):
    def test_policy_validation(self):
        self.assertEqual("av2", validate_cav_policy("av2"))
        self.assertEqual("idm-mobil", validate_cav_policy("idm-mobil"))
        with self.assertRaises(ValueError):
            validate_cav_policy("unknown")

    def test_nearest_longitudinal_action_respects_mask(self):
        self.assertEqual(3, nearest_longitudinal_action(-1.1, ACTIONS))
        indicator = [True, True, True, False, True, True, True]
        self.assertEqual(4, nearest_longitudinal_action(-1.1, ACTIONS, indicator))

    def test_idm_free_road_selects_acceleration(self):
        agent = IDMMobilCAVAgent(FakeEnv(), actions=ACTIONS)
        action = agent.decision(None, [True] * len(ACTIONS))
        self.assertEqual(5, action)


if __name__ == "__main__":
    unittest.main()
