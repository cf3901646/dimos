# Select the evaluation runtime per benchmark

Each `Evaluation` declares one runtime profile, and each run specification pins
that profile's model and thinking level. The runner constructs the declared
runtime and records its identity in `run.json`.

LIBERO-PRO uses `code-policy-v1`: Pi explores with debug trials, freezes a
callable, and leaves the measured runtime loop. VLN-CE uses `live-agent-v1`: Pi
and its persistent Python workspace remain active while the measured episode
runs. Both profiles use the same evaluation runner and evidence contract.

Navigation requires repeated choices from new observations and often uses a
long wall-clock horizon. Freezing a callable before the episode would remove
the decision-maker this condition intends to measure. Conversely, changing
LIBERO-PRO to a live agent would change its established CodePolicy condition.
The evaluation therefore owns this choice instead of exposing it as an
operator toggle.
