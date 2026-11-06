# Scene representations

The manuscript's primary representation is `interaction-graph-intent-v1`. It
converts simulator state into an agent-centric graph centered on the automated
vehicle and selected background vehicle.

The graph exposes:

- semantic actor roles instead of simulator-specific identifiers;
- lane topology and reachability;
- front, rear, alongside, and closing relations;
- time-to-collision, headway, predicted clearance, and conflict cues;
- short motion history;
- feasible interaction intents mapped losslessly to the original 33 actions.

The model first selects an interaction intent and then a compatible low-level
action. Deterministic code owns syntax, feasibility checks, action indices, and
execution. This is a representation and interface choice, not a change to the
simulator's action support.

`nade-native-v1` reconstructs the subject-centered six-slot NADE observation and
serves as the controlled comparison. Other profiles remain available for
development audits and ablations; they should not be described as the primary
method unless a new protocol explicitly freezes them.

Run the offline inventory with:

```bash
highway show-representations
highway demo --representation interaction-graph-intent-v1
```
