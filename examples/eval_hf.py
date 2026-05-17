# -*- coding: utf-8 -*-
import os
import json
import time
import subprocess
from tensorboard.backend.event_processing import event_accumulator
from tensorboard.summary.writer.event_file_writer import EventFileWriter
from tensorboard.compat.proto.event_pb2 import Event

PROJECT_ROOT = "xxx"
TARGET_MODEL_PATH = "xxx"
TB_ROOT_BASE = os.path.join(PROJECT_ROOT, "tensorboard_log/eval_log/")

def merge_tb(tb_dir):
    merged_dir = os.path.join(tb_dir, "merged")
    os.makedirs(merged_dir, exist_ok=True)

    event_files = [
        os.path.join(tb_dir, f)
        for f in os.listdir(tb_dir)
        if f.startswith("events")
    ]
    if not event_files:
        return None

    writer = EventFileWriter(merged_dir)

    for ef in event_files:
        try:
            ea = event_accumulator.EventAccumulator(ef)
            ea.Reload()
            tags = ea.Tags().get("scalars", [])
            for tag in tags:
                for e in ea.Scalars(tag):
                    ev = Event(wall_time=e.wall_time, step=e.step)
                    ev.summary.value.add(tag=tag, simple_value=e.value)
                    writer.add_event(ev)
        except Exception as e:
            print(f"{e}")

    writer.close()
    return merged_dir


def read_scores(tb_merged_dir):
    if not tb_merged_dir:
        return {}

    event_files = [
        os.path.join(tb_merged_dir, f)
        for f in os.listdir(tb_merged_dir)
        if f.startswith("events")
    ]
    if not event_files:
        return {}

    ea = event_accumulator.EventAccumulator(event_files[0])
    ea.Reload()

    results = {}
    for tag in ea.Tags().get("scalars", []):
        if tag.startswith("val/test_score/") and "_pass@" not in tag:
            events = ea.Scalars(tag)
            if events:
                results[tag] = events[-1].value
    return results

def run_eval(model_abs_path, dataset, n_val, tb_output_dir):
    model_parent_dir = os.path.dirname(model_abs_path)
    model_dir_name = os.path.basename(model_abs_path)

    os.environ["MODEL_PATH"] = model_parent_dir
    os.environ["TENSORBOARD_DIR"] = tb_output_dir
    os.makedirs(os.environ["TENSORBOARD_DIR"], exist_ok=True)

    env = os.environ.copy()
    env.update({
        "SANDBOX_ENDPOINT": "http://127.0.0.1:12345/faas/sandbox/",
        "DATA_PATH": os.path.join(PROJECT_ROOT, "datasets/"),
        "CHECKPOINT_PATH": model_abs_path,
        "LOG_PATH": "./logs",
        "NNODES": "1",
        "GPUS_PER_NODE": "8",
        "RESUME": "False",
        "CONFIG_NAME": "simpletir_trainer"
    })

    cmd = [
        "bash", "train.sh",
        "--max_response_length", "8000",
        "--max_prompt_length", "16000",
        "--model_name", model_dir_name,
        "--max_turns", "5",
        "--valid_dataset", dataset,
        "--val_only", "True",
        "--n_val", str(n_val),
        "--output_acc_to_file", "False",
        "--val_sample_size", "500",
        "--sp_size", "2",
        "--val_before_train", "True",
    ]
    subprocess.check_call(cmd, env=env)


if __name__ == "__main__":

    if not os.path.exists(TARGET_MODEL_PATH):
        exit(1)

    model_name = os.path.basename(TARGET_MODEL_PATH)
    current_tb_dir = os.path.join(TB_ROOT_BASE, model_name)
    run_eval(TARGET_MODEL_PATH, "deepscaler/olympiad deepscaler/math500", 8, current_tb_dir)
    run_eval(TARGET_MODEL_PATH, "deepscaler/aime24 deepscaler/aime25 deepscaler/amc23 deepscaler/hmmt25", 32,
             current_tb_dir)
    time.sleep(5)
    merged_dir = merge_tb(current_tb_dir)
    final_results = {}
    if merged_dir:
        scores = read_scores(merged_dir)
        if scores:
            avg = sum(scores.values()) / len(scores)
            scores['average_accuracy'] = avg

        final_results[model_name] = scores

        json_path = os.path.join(TB_ROOT_BASE, f"{model_name}_results.json")
        with open(json_path, "w") as f:
            json.dump(final_results, f, indent=2, ensure_ascii=False)