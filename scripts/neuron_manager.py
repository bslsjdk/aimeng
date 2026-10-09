#!/usr/bin/env python3
"""AIMENG real-artifact neuron manager.

The registry is intentionally empty unless the user imports/registers real artifacts.
No built-in demo neurons are created. Every mutation and failure is appended to JSONL.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import sys
import time
import uuid
import zipfile
from pathlib import Path
from typing import Any

REGISTRY_SCHEMA = "aimeng.neuron_registry.v1"
ARTIFACT_FORMAT = "aimeng.linear_neuron.v1"


class ManagerError(RuntimeError):
    pass


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def append_log(log_path: Path, *, level: str, event: str, message: str,
               details: dict[str, Any] | None = None, run_id: str | None = None) -> dict:
    row = {
        "schema_version": "aimeng.neuron_manager_log.v1",
        "event_id": str(uuid.uuid4()),
        "run_id": run_id or str(uuid.uuid4()),
        "timestamp_unix": time.time(),
        "level": level,
        "event": event,
        "message": message,
        "details": details or {},
    }
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n")
    return row


def read_registry(path: Path) -> dict:
    if not path.is_file():
        raise ManagerError(f"真实神经元注册表不存在：{path}。请先导入/登记真实工件；系统不会自动生成示例神经元。")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ManagerError(f"注册表读取失败：{exc}") from exc
    if not isinstance(data, dict) or data.get("schema_version") != REGISTRY_SCHEMA:
        raise ManagerError(f"注册表 schema 不正确，必须为 {REGISTRY_SCHEMA}")
    neurons = data.get("neurons")
    if not isinstance(neurons, list):
        raise ManagerError("注册表 neurons 必须是数组")
    seen = set()
    for item in neurons:
        if not isinstance(item, dict) or not isinstance(item.get("id"), str) or not item["id"].strip():
            raise ManagerError("注册表包含缺少 id 的神经元记录")
        if item["id"] in seen:
            raise ManagerError(f"注册表存在重复神经元 ID：{item['id']}")
        seen.add(item["id"])
    return data


def write_registry(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
                    encoding="utf-8")
    os.replace(temp, path)


def resolve_artifact(root: Path, neuron: dict) -> Path:
    rel = neuron.get("artifact")
    if not isinstance(rel, str) or not rel.strip():
        raise ManagerError(f"{neuron.get('id')}: 未配置真实 artifact 路径")
    path = Path(rel)
    if not path.is_absolute():
        path = root / path
    path = path.resolve()
    if not path.is_file():
        raise ManagerError(f"{neuron.get('id')}: 工件文件不存在：{path}")
    expected = neuron.get("sha256")
    actual = sha256_file(path)
    if not isinstance(expected, str) or expected != actual:
        raise ManagerError(f"{neuron.get('id')}: 工件 SHA-256 不匹配，文件可能已变化或注册信息过期")
    return path


def selected_neurons(data: dict, ids: list[str]) -> list[dict]:
    if not ids:
        raise ManagerError("必须显式指定 --ids，拒绝对未知范围执行批量操作")
    index = {item["id"]: item for item in data["neurons"]}
    missing = [unit_id for unit_id in ids if unit_id not in index]
    if missing:
        raise ManagerError("以下 ID 不存在于真实注册表，未执行任何操作：" + ", ".join(missing))
    return [index[unit_id] for unit_id in dict.fromkeys(ids)]


def validate_linear_artifact(path: Path, expected_dims: int | None = None) -> tuple[list[float], float]:
    try:
        artifact = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ManagerError(f"无法读取线性神经元工件 {path}: {exc}") from exc
    if not isinstance(artifact, dict) or artifact.get("format") != ARTIFACT_FORMAT:
        raise ManagerError(f"暂不支持该工件格式：{path}；批量评测当前只支持 {ARTIFACT_FORMAT}")
    weights, bias = artifact.get("weights"), artifact.get("bias")
    if not isinstance(weights, list) or not weights or not all(
        isinstance(v, (int, float)) and math.isfinite(v) for v in weights
    ):
        raise ManagerError(f"{path}: weights 必须是非空有限数值数组")
    if not isinstance(bias, (int, float)) or not math.isfinite(bias):
        raise ManagerError(f"{path}: bias 必须是有限数值")
    if expected_dims is not None and len(weights) != expected_dims:
        raise ManagerError(f"{path}: 输入维度 {len(weights)} 与测试数据维度 {expected_dims} 不匹配")
    return [float(v) for v in weights], float(bias)


def load_dataset(path: Path) -> list[tuple[list[float], float]]:
    if not path.is_file():
        raise ManagerError(f"评测数据集不存在：{path}")
    rows = []
    try:
        for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            row = json.loads(line)
            inputs, target = row.get("input"), row.get("target")
            if not isinstance(inputs, list) or not inputs or not all(
                isinstance(v, (int, float)) and math.isfinite(v) for v in inputs
            ):
                raise ManagerError(f"评测数据第 {line_no} 行 input 格式错误")
            if not isinstance(target, (int, float)) or not math.isfinite(target):
                raise ManagerError(f"评测数据第 {line_no} 行 target 格式错误")
            rows.append(([float(v) for v in inputs], float(target)))
    except json.JSONDecodeError as exc:
        raise ManagerError(f"评测数据不是合法 JSONL：{exc}") from exc
    if not rows:
        raise ManagerError("评测数据集为空")
    return rows


def evaluate_one(neuron: dict, artifact: Path, dataset: list[tuple[list[float], float]]) -> dict:
    weights, bias = validate_linear_artifact(artifact, len(dataset[0][0]))
    losses = []
    for inputs, target in dataset:
        if len(inputs) != len(weights):
            raise ManagerError(f"{neuron['id']}: 评测样本输入维度不一致")
        prediction = sum(w * x for w, x in zip(weights, inputs)) + bias
        losses.append((prediction - target) ** 2)
    mse = sum(losses) / len(losses)
    if not math.isfinite(mse):
        raise ManagerError(f"{neuron['id']}: MSE 非有限值")
    return {"id": neuron["id"], "samples": len(losses), "mse": mse,
            "artifact_sha256": sha256_file(artifact), "status": "evaluated"}


def run(args: argparse.Namespace) -> dict:
    registry_path = Path(args.registry).resolve()
    root = Path(args.root).resolve()
    log_path = Path(args.log).resolve()
    run_id = str(uuid.uuid4())
    command = args.command
    try:
        data = read_registry(registry_path)
        ids = getattr(args, "ids", None) or []
        neurons = selected_neurons(data, ids) if command != "list" else data["neurons"]
        results: list[dict] = []

        if command == "list":
            results = [{"id": n["id"], "enabled": bool(n.get("enabled", True)),
                        "quality": n.get("quality", "unassessed"),
                        "artifact": n.get("artifact", "")} for n in neurons]
        elif command in ("enable", "disable"):
            enabled = command == "enable"
            for neuron in neurons:
                neuron["enabled"] = enabled
                neuron["updated_at_unix"] = time.time()
                results.append({"id": neuron["id"], "enabled": enabled, "status": "updated"})
            write_registry(registry_path, data)
        elif command == "save":
            output = Path(args.output).resolve()
            output.mkdir(parents=True, exist_ok=True)
            for neuron in neurons:
                source = resolve_artifact(root, neuron)
                target_dir = output / neuron["id"]
                target_dir.mkdir(parents=True, exist_ok=True)
                target = target_dir / f"revision-{int(neuron.get('revision', 1))}{source.suffix}"
                shutil.copy2(source, target)
                digest = sha256_file(target)
                manifest = {"schema_version": "aimeng.neuron_snapshot.v1", "id": neuron["id"],
                            "revision": int(neuron.get("revision", 1)), "artifact": target.name,
                            "sha256": digest, "saved_at_unix": time.time(),
                            "source_sha256": neuron["sha256"]}
                (target_dir / "manifest.json").write_text(
                    json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
                neuron["last_saved_snapshot"] = str(target)
                results.append({"id": neuron["id"], "status": "saved",
                                "snapshot": str(target), "sha256": digest})
            write_registry(registry_path, data)
        elif command == "export":
            output = Path(args.output).resolve()
            output.parent.mkdir(parents=True, exist_ok=True)
            manifest_items = []
            with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
                for neuron in neurons:
                    source = resolve_artifact(root, neuron)
                    archive_name = f"neurons/{neuron['id']}/{source.name}"
                    archive.write(source, archive_name)
                    manifest_items.append({"id": neuron["id"], "enabled": bool(neuron.get("enabled", True)),
                                           "revision": int(neuron.get("revision", 1)),
                                           "artifact": archive_name, "sha256": sha256_file(source),
                                           "quality": neuron.get("quality", "unassessed")})
                archive.writestr("manifest.json", json.dumps({
                    "schema_version": "aimeng.neuron_export.v1",
                    "exported_at_unix": time.time(), "neurons": manifest_items
                }, ensure_ascii=False, indent=2) + "\n")
            results = [{"status": "exported", "output": str(output), "count": len(manifest_items),
                        "ids": [n["id"] for n in neurons]}]
        elif command == "evaluate":
            dataset = load_dataset(Path(args.dataset).resolve())
            for neuron in neurons:
                artifact = resolve_artifact(root, neuron)
                score = evaluate_one(neuron, artifact, dataset)
                threshold = args.good_mse_threshold
                score["quality"] = "excellent" if score["mse"] <= threshold else "candidate"
                neuron["last_evaluation"] = score
                neuron["quality"] = score["quality"]
                neuron["updated_at_unix"] = time.time()
                results.append(score)
            write_registry(registry_path, data)
        else:
            raise ManagerError(f"未知操作：{command}")

        result = {"ok": True, "command": command, "run_id": run_id,
                  "selected_ids": [n["id"] for n in neurons], "results": results, "errors": []}
        append_log(log_path, level="INFO", event=f"batch_{command}_completed",
                   message=f"{command} 操作完成", details=result, run_id=run_id)
        return result
    except Exception as exc:
        result = {"ok": False, "command": command, "run_id": run_id,
                  "selected_ids": getattr(args, "ids", None) or [], "results": [],
                  "errors": [{"type": type(exc).__name__, "message": str(exc)}]}
        try:
            append_log(log_path, level="ERROR", event=f"batch_{command}_failed",
                       message=str(exc), details=result, run_id=run_id)
        except OSError:
            pass
        return result


def parser_for() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", default="data/neuron_registry.json")
    parser.add_argument("--root", default=".")
    parser.add_argument("--log", default="runs/neuron-manager.jsonl")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("list", help="列出真实注册表中的神经元；不生成示例项")
    for name in ("save", "export", "enable", "disable", "evaluate"):
        child = sub.add_parser(name)
        child.add_argument("--ids", nargs="+", required=True, help="明确指定要操作的真实神经元 ID")
        if name in ("save", "export"):
            child.add_argument("--output", required=True)
        if name == "evaluate":
            child.add_argument("--dataset", required=True, help="独立 JSONL，每行包含 input 数组和 target 数值")
            child.add_argument("--good-mse-threshold", type=float, default=0.05)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = parser_for().parse_args(argv)
    if getattr(args, "good_mse_threshold", 0.0) < 0:
        print(json.dumps({"ok": False, "errors": [{"message": "good MSE threshold must be >= 0"}]}))
        return 2
    result = run(args)
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
