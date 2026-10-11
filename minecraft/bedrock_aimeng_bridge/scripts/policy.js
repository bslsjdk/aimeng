/*
 * AIMENG policy adapter contract.
 *
 * IMPORTANT: this first policy is a deterministic heuristic for checking the
 * live Bedrock observation/action loop. It is NOT the trained PyTorch model.
 * Replace selectAction with the exported AIMENG policy after we implement and
 * validate weight export + JavaScript inference parity.
 */
export const ACTIONS = Object.freeze({
  WALK_FORWARD: "walk_forward",
  TURN_LEFT: "turn_left",
  TURN_RIGHT: "turn_right",
  JUMP_FORWARD: "jump_forward",
  STOP: "stop"
});

export function selectAction(observation) {
  if (!observation || observation.goalDistance <= 1.35) return ACTIONS.STOP;
  if (observation.onGround && observation.supportAhead === false) return ACTIONS.JUMP_FORWARD;
  if (observation.headingErrorDegrees > 12) return ACTIONS.TURN_RIGHT;
  if (observation.headingErrorDegrees < -12) return ACTIONS.TURN_LEFT;
  return ACTIONS.WALK_FORWARD;
}
