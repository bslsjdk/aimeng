import { world, system } from "@minecraft/server";
import { selectAction, ACTIONS } from "./policy.js";

const activeAgents = new Map();
const REPORT_INTERVAL = 40;
const MAX_SPEED = 0.16;
const STEERING_GAIN = 0.22;
const JUMP_IMPULSE = 0.42;

function normalizeDegrees(degrees) {
  let result = degrees;
  while (result > 180) result -= 360;
  while (result < -180) result += 360;
  return result;
}

function horizontalDistance(a, b) {
  const dx = b.x - a.x;
  const dz = b.z - a.z;
  return Math.sqrt(dx * dx + dz * dz);
}

function blockAt(dimension, x, y, z) {
  try {
    return dimension.getBlock({ x: Math.floor(x), y: Math.floor(y), z: Math.floor(z) });
  } catch {
    return undefined;
  }
}

function makeObservation(player, state) {
  const location = player.location;
  const rotation = player.getRotation();
  const dx = state.goal.x - location.x;
  const dz = state.goal.z - location.z;
  const desiredYaw = Math.atan2(-dx, dz) * 180 / Math.PI;
  const headingErrorDegrees = normalizeDegrees(desiredYaw - rotation.y);
  const length = Math.max(0.001, Math.sqrt(dx * dx + dz * dz));
  const aheadX = location.x + (dx / length) * 1.25;
  const aheadZ = location.z + (dz / length) * 1.25;
  const supportHere = blockAt(player.dimension, location.x, location.y - 1, location.z);
  const supportAhead = blockAt(player.dimension, aheadX, location.y - 1, aheadZ);

  return {
    position: { x: location.x, y: location.y, z: location.z },
    velocity: player.getVelocity(),
    rotation,
    goal: { x: state.goal.x, y: state.goal.y, z: state.goal.z },
    goalDistance: horizontalDistance(location, state.goal),
    headingErrorDegrees,
    onGround: Boolean(player.isOnGround),
    supportHere: supportHere ? supportHere.typeId : "unknown",
    supportAhead: supportAhead ? supportAhead.typeId !== "minecraft:air" : null,
    dimensionId: player.dimension.id,
    tick: system.currentTick
  };
}

function stopHorizontalMotion(player) {
  const velocity = player.getVelocity();
  player.applyImpulse({ x: -velocity.x * 0.7, y: 0, z: -velocity.z * 0.7 });
}

function executeAction(player, state, observation, action) {
  if (action === ACTIONS.STOP) {
    stopHorizontalMotion(player);
    state.status = "goal_reached";
    state.completedAtTick = system.currentTick;
    player.sendMessage("§a[AIMENG] Waypoint reached. Run /scriptevent aimeng:start to try again.");
    return;
  }

  // Minecraft gravity and collision resolve movement; this bridge does not teleport.
  const rotation = player.getRotation();
  const targetYaw = Math.atan2(
    -(state.goal.x - observation.position.x),
    state.goal.z - observation.position.z
  ) * 180 / Math.PI;
  const yawError = normalizeDegrees(targetYaw - rotation.y);
  const nextYaw = rotation.y + Math.max(-8, Math.min(8, yawError));
  player.setRotation({ x: rotation.x, y: nextYaw });

  const yawRadians = nextYaw * Math.PI / 180;
  const desiredX = -Math.sin(yawRadians) * MAX_SPEED;
  const desiredZ = Math.cos(yawRadians) * MAX_SPEED;
  const velocity = player.getVelocity();
  let impulseY = 0;
  if (action === ACTIONS.JUMP_FORWARD && observation.onGround) impulseY = JUMP_IMPULSE;

  player.applyImpulse({
    x: (desiredX - velocity.x) * STEERING_GAIN,
    y: impulseY,
    z: (desiredZ - velocity.z) * STEERING_GAIN
  });
}

world.afterEvents.scriptEventReceive.subscribe((event) => {
  if (!event.id.startsWith("aimeng:")) return;
  const player = event.sourceEntity;
  if (!player || player.typeId !== "minecraft:player") return;

  if (event.id === "aimeng:start") {
    const location = player.location;
    const rotation = player.getRotation();
    const yaw = rotation.y * Math.PI / 180;
    const goal = {
      x: location.x - Math.sin(yaw) * 8,
      y: location.y,
      z: location.z + Math.cos(yaw) * 8
    };
    activeAgents.set(player.id, {
      playerId: player.id,
      goal,
      status: "running",
      startedAtTick: system.currentTick,
      lastReportTick: system.currentTick,
      actions: 0,
      lastAction: "none"
    });
    player.sendMessage("§e[AIMENG] Live Bedrock bridge started. Current policy is heuristic, not the trained network.");
    player.sendMessage("§7Waypoint placed about 8 blocks ahead. Use /scriptevent aimeng:stop to stop.");
    return;
  }

  if (event.id === "aimeng:stop") {
    activeAgents.delete(player.id);
    stopHorizontalMotion(player);
    player.sendMessage("§c[AIMENG] Controller stopped.");
    return;
  }

  if (event.id === "aimeng:status") {
    const state = activeAgents.get(player.id);
    player.sendMessage(state
      ? "[AIMENG] status=" + state.status + ", action=" + state.lastAction +
        ", actions=" + state.actions + ", elapsedTicks=" + (system.currentTick - state.startedAtTick)
      : "[AIMENG] No active controller. Run /scriptevent aimeng:start.");
  }
});

system.runInterval(() => {
  for (const [playerId, state] of activeAgents) {
    const player = world.getEntity(playerId);
    if (!player || player.typeId !== "minecraft:player" || state.status !== "running") {
      activeAgents.delete(playerId);
      continue;
    }

    const observation = makeObservation(player, state);
    const action = selectAction(observation);
    state.lastAction = action;
    state.actions += 1;
    executeAction(player, state, observation, action);

    if (system.currentTick - state.lastReportTick >= REPORT_INTERVAL && state.status === "running") {
      state.lastReportTick = system.currentTick;
      player.sendMessage(
        "§b[AIMENG] d=" + observation.goalDistance.toFixed(2) + "m, yawError=" +
        observation.headingErrorDegrees.toFixed(1) + "°, action=" + action +
        ", supportAhead=" + String(observation.supportAhead)
      );
    }

    if (system.currentTick - state.startedAtTick > 20 * 45) {
      activeAgents.delete(playerId);
      stopHorizontalMotion(player);
      player.sendMessage("§c[AIMENG] 45-second smoke-test timeout; no completion recorded.");
    }
  }
}, 1);
