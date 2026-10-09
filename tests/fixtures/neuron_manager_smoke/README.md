# TEST ONLY: Neuron Manager Import Smoke Test

These files are synthetic fixtures used only to verify the import/evaluation plumbing. They are NOT trained AIMENG neurons and must never be treated as evidence of model capability.

From the repository root, test import with:

    python scripts/neuron_manager.py import --id TEST-ONLY-N-001 --artifact tests/fixtures/neuron_manager_smoke/TEST_ONLY_neuron.json

Then evaluate with:

    python scripts/neuron_manager.py evaluate --ids TEST-ONLY-N-001 --dataset tests/fixtures/neuron_manager_smoke/TEST_ONLY_heldout.jsonl

The expected MSE is approximately zero because this fixture encodes the same simple affine rule as the test targets. That result only proves the file/manager/evaluator pipeline works. It does not prove learning, language ability, or usefulness of any real neuron.

Remove the TEST-ONLY registry entry and generated runs after smoke testing. Never assign this fixture the production IDs N-001 or N-003.
