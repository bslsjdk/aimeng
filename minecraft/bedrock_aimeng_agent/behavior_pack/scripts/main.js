import { world, system } from "@minecraft/server";

const agents = new Map();
const SPEED = 0.14;
const GAIN = 0.20;
const JUMP = 0.42;
const REPORT_EVERY = 40;
const MAX_TICKS = 20 * 45;

function wrap(deg) {
  while (deg > 180) deg -= 360;
  while (deg < -180) deg += 360;
  return deg;
}
function dist2(a, b) {
  const dx = a.x - b.x, dz = a.z - b.z;
  return Math.sqrt(dx * dx + dz * dz);
}
function getBlock(dimension, x, y, z) {
  try { return dimension.getBlock({x: Math.floor(x), y: Math.floor(y), z: Math.floor(z)}); }
  catch { return undefined; }
}
function nearestAgent(player) {
  const list = player.dimension.getEntities({
    type: "aimeng:parkour_agent",
    location: player.location,
    maxDistance: 32
  });
  let best, bestD = Infinity;
  for (const entity of list) {
    const d = dist2(entity.location, player.location);
    if (d < bestD) { best = entity; bestD = d; }
  }
  return best;
}
function observe(entity, state) {
  const p = entity.location;
  const rot = entity.getRotation();
  const dx = state.goal.x - p.x, dz = state.goal.z - p.z;
  const d = Math.max(0.001, Math.sqrt(dx * dx + dz * dz));
  const desiredYaw = Math.atan2(-dx, dz) * 180 / Math.PI;
  const headingError = wrap(desiredYaw - rot.y);
  const ahead = getBlock(entity.dimension, p.x + dx / d * 1.2, p.y - 1, p.z + dz / d * 1.2);
  const here = getBlock(entity.dimension, p.x, p.y - 1, p.z);
  return {
    position: {x:p.x, y:p.y, z:p.z},
    velocity: entity.getVelocity(),
    rotation: rot,
    goalDistance: dist2(p, state.goal),
    headingErrorDegrees: headingError,
    onGround: Boolean(entity.isOnGround),
    supportHere: here ? here.typeId : "unknown",
    supportAhead: ahead ? ahead.typeId !== "minecraft:air" : null
  };
}
function stop(entity) {
  const v = entity.getVelocity();
  entity.applyImpulse({x:-v.x * 0.7, y:0, z:-v.z * 0.7});
}
function tell(player, message) {
  try { player.sendMessage(message); } catch {}
}
function begin(player) {
  const entity = nearestAgent(player);
  if (!entity) {
    tell(player, "§c[AIMENG] No agent within 32 blocks. First run /summon aimeng:parkour_agent ~ ~ ~");
    return;
  }
  const p = entity.location, rot = entity.getRotation(), yaw = rot.y * Math.PI / 180;
  const goal = {x:p.x - Math.sin(yaw) * 8, y:p.y, z:p.z + Math.cos(yaw) * 8};
  agents.set(entity.id, {
    entityId: entity.id, ownerId: player.id, goal,
    status: "running", startedAt: system.currentTick,
    lastReport: system.currentTick, actions: 0, lastAction: "start"
  });
  tell(player, "§a[AIMENG] Agent controller started. The visible entity moves independently; current policy is heuristic, not the trained network.");
}
world.afterEvents.scriptEventReceive.subscribe((event) => {
  if (!event.id.startsWith("aimeng:agent_")) return;
  const player = event.sourceEntity;
  if (!player || player.typeId !== "minecraft:player") return;
  if (event.id === "aimeng:agent_start") { begin(player); return; }
  const entity = nearestAgent(player);
  if (!entity) { tell(player, "§c[AIMENG] No agent within 32 blocks."); return; }
  const state = agents.get(entity.id);
  if (event.id === "aimeng:agent_stop") {
    agents.delete(entity.id);
    stop(entity);
    tell(player, "§e[AIMENG] Agent stopped.");
  } else if (event.id === "aimeng:agent_status") {
    tell(player, state
      ? "[AIMENG] status=" + state.status + ", action=" + state.lastAction +
        ", actions=" + state.actions + ", ticks=" + (system.currentTick - state.startedAt)
      : "[AIMENG] Agent exists but is not running. Use /scriptevent aimeng:agent_start.");
  }
});
system.runInterval(() => {
  for (const [id, state] of agents) {
    const entity = world.getEntity(id);
    if (!entity || entity.typeId !== "aimeng:parkour_agent") { agents.delete(id); continue; }
    if (state.status !== "running") continue;
    const obs = observe(entity, state);
    if (obs.goalDistance <= 1.25) {
      stop(entity); state.status = "goal_reached";
      const owner = world.getEntity(state.ownerId);
      if (owner) tell(owner, "§a[AIMENG] Agent reached waypoint. Use agent_start to run another test.");
      continue;
    }
    if (system.currentTick - state.startedAt > MAX_TICKS) {
      stop(entity); state.status = "timeout";
      const owner = world.getEntity(state.ownerId);
      if (owner) tell(owner, "§c[AIMENG] Agent test timed out after 45 seconds.");
      continue;
    }

    const targetYaw = Math.atan2(-(state.goal.x - obs.position.x), state.goal.z - obs.position.z) * 180 / Math.PI;
    const yaw = obs.rotation.y + Math.max(-8, Math.min(8, wrap(targetYaw - obs.rotation.y)));
    entity.setRotation({x:0, y:yaw});
    const rad = yaw * Math.PI / 180;
    const vx = -Math.sin(rad) * SPEED, vz = Math.cos(rad) * SPEED;
    const v = entity.getVelocity();
    const jump = obs.onGround && obs.supportAhead === false ? JUMP : 0;
    entity.applyImpulse({x:(vx-v.x)*GAIN, y:jump, z:(vz-v.z)*GAIN});
    state.actions++;
    state.lastAction = jump ? "jump_forward" : (Math.abs(obs.headingErrorDegrees) > 12 ? "turn" : "walk_forward");

    if (system.currentTick - state.lastReport >= REPORT_EVERY) {
      state.lastReport = system.currentTick;
      const owner = world.getEntity(state.ownerId);
      if (owner) tell(owner, "§b[AIMENG] agent distance=" + obs.goalDistance.toFixed(2) +
        ", yawError=" + obs.headingErrorDegrees.toFixed(1) + "°, action=" + state.lastAction);
    }
  }
}, 1);
