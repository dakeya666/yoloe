from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from ultralytics import YOLOE
from ultralytics.models.yolo.yoloe import YOLOEVPDetectPredictor, YOLOEVPSegPredictor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate YOLOE hybrid inference: old-class regression, reference-image hybrid, prompt-only hybrid, and NMS comparison."
    )
    parser.add_argument("--weights", type=str, default="./runs/detect/VP/weights/best.pt", help="Model checkpoint path.")
    parser.add_argument("--source", type=str, default="./testImg", help="Inference source, image or directory.")
    parser.add_argument(
        "--refer-image",
        type=str,
        # default="./testImg/1.jpg",
        default="./testImg/3_3746972-3514_1_0.jpg",
        help="Reference image used to extract visual prompt embeddings.",
    )
    parser.add_argument(
        "--prompt-box",
        type=str,
        # default="383,620,469,744",
        default="40,40,2048,544",
        help="Visual prompt box in x1,y1,x2,y2 format.",
    )
    parser.add_argument(
        "--prompt-name",
        type=str,
        default="novel_defect_a",
        help="Readable name for the prompted novel class. Prefer a string label over an integer id.",
    )
    parser.add_argument(
        "--base-names",
        type=str,
        default="",
        help="Optional comma-separated old-class names. If omitted, the script uses the first --base-nc model names.",
    )
    parser.add_argument("--base-nc", type=int, default=20, help="Number of base classes to keep when base names are implicit.")
    parser.add_argument("--conf", type=float, default=0.2, help="Confidence threshold.")
    parser.add_argument("--iou", type=float, default=0.7, help="IoU threshold used by NMS.")
    parser.add_argument("--imgsz", type=int, default=640, help="Inference image size.")
    parser.add_argument("--device", type=str, default=None, help="Inference device, for example 0 or cpu.")
    parser.add_argument("--project", type=str, default="runs/testPar", help="Project directory for saved visualizations.")
    parser.add_argument(
        "--report-limit",
        type=int,
        default=12,
        help="Maximum number of class names or counts printed in each summary section.",
    )
    parser.add_argument("--show", action="store_true", help="Show prediction windows during inference.")
    parser.add_argument("--save", dest="save", action="store_true", help="Save rendered outputs.")
    parser.add_argument("--no-save", dest="save", action="store_false", help="Do not save rendered outputs.")
    parser.set_defaults(save=True)
    return parser.parse_args()


def parse_box(box_text: str) -> np.ndarray:
    values = [float(x.strip()) for x in box_text.split(",") if x.strip()]
    if len(values) != 4:
        raise ValueError(f"Expected 4 comma-separated values for --prompt-box, but got: {box_text}")
    x1, y1, x2, y2 = values
    if x2 <= x1 or y2 <= y1:
        raise ValueError(f"Invalid prompt box {box_text}: require x2>x1 and y2>y1.")
    return np.asarray([values], dtype=np.float32)


def parse_base_names(text: str) -> list[str] | None:
    names = [x.strip() for x in text.split(",") if x.strip()]
    return names or None


def ensure_local_path(path_str: str, label: str) -> None:
    if not path_str:
        raise ValueError(f"{label} is empty.")
    path = Path(path_str)
    if not path.exists():
        raise FileNotFoundError(f"{label} does not exist: {path}")


def load_model(weights: str) -> YOLOE:
    ensure_local_path(weights, "Weights")
    return YOLOE(weights)


def choose_vp_predictor(model: YOLOE):
    return YOLOEVPSegPredictor if model.task == "segment" else YOLOEVPDetectPredictor


def build_visual_prompts(prompt_box: np.ndarray, prompt_name: str) -> dict[str, Any]:
    return {"bboxes": prompt_box.copy(), "cls": [prompt_name]}


def base_prompt_kwargs(args: argparse.Namespace) -> dict[str, Any]:
    kwargs = {
        "conf": args.conf,
        "iou": args.iou,
        "imgsz": args.imgsz,
        "project": args.project,
        "save": args.save,
        "show": args.show,
    }
    if args.device is not None:
        kwargs["device"] = args.device
    return kwargs


def hybrid_prompt_kwargs(args: argparse.Namespace) -> dict[str, Any]:
    kwargs = {"base_nc": args.base_nc}
    base_names = parse_base_names(args.base_names)
    if base_names:
        kwargs["base_names"] = base_names
    return kwargs


def results_name_map(results) -> dict[int, str]:
    if not results:
        return {}
    names = getattr(results[0], "names", {})
    return names if isinstance(names, dict) else dict(enumerate(names))


def summarize_results(stage: str, results, report_limit: int) -> dict[str, Any]:
    names = results_name_map(results)
    counts: Counter[int] = Counter()
    image_detection_counts: dict[str, int] = {}

    for result in results:
        boxes = getattr(result, "boxes", None)
        if boxes is None:
            image_detection_counts[result.path] = 0
            continue

        cls_values = boxes.cls
        cls_ids = cls_values.tolist() if hasattr(cls_values, "tolist") else list(cls_values)
        cls_ids = [int(x) for x in cls_ids]
        counts.update(cls_ids)
        image_detection_counts[result.path] = len(cls_ids)

    total_images = len(results)
    total_detections = sum(image_detection_counts.values())

    print(f"\n=== {stage} ===")
    print(f"images={total_images}, detections={total_detections}, prototype_classes={len(names)}")
    if names:
        sample_names = ", ".join(f"{idx}:{name}" for idx, name in list(names.items())[:report_limit])
        suffix = " ..." if len(names) > report_limit else ""
        print(f"names={sample_names}{suffix}")
    if counts:
        print("top class counts:")
        for cls_id, count in counts.most_common(report_limit):
            print(f"  {cls_id}: {names.get(cls_id, str(cls_id))} x{count}")
    else:
        print("top class counts: none")

    return {
        "names": names,
        "counts": counts,
        "image_detection_counts": image_detection_counts,
        "total_images": total_images,
        "total_detections": total_detections,
    }


def check_hybrid_expectation(stage: str, summary: dict[str, Any], args: argparse.Namespace) -> None:
    names = summary["names"]
    expected_min_classes = args.base_nc + 1
    print("checks:")
    print(f"  expected prototype_classes >= {expected_min_classes}, got {len(names)}")
    print(f"  prompt name present: {args.prompt_name in names.values()}")


def run_old_class_regression(args: argparse.Namespace) -> dict[str, Any]:
    model = load_model(args.weights)
    results = model.predict(
        source=args.source,
        name="01_old_class_regression",
        agnostic_nms=False,
        **base_prompt_kwargs(args),
    )
    return summarize_results("01 Old-Class Regression", results, args.report_limit)


def run_reference_hybrid(args: argparse.Namespace, visual_prompts: dict[str, Any]) -> dict[str, Any]:
    model = load_model(args.weights)
    predictor = choose_vp_predictor(model)
    results = model.predict(
        source=args.source,
        refer_image=args.refer_image,
        visual_prompts=visual_prompts,
        predictor=predictor,
        name="02_reference_hybrid",
        agnostic_nms=False,
        **base_prompt_kwargs(args),
        **hybrid_prompt_kwargs(args),
    )
    summary = summarize_results("02 Reference-Image Hybrid", results, args.report_limit)
    check_hybrid_expectation("02 Reference-Image Hybrid", summary, args)
    return summary


def run_prompt_only_hybrid(args: argparse.Namespace, visual_prompts: dict[str, Any]) -> dict[str, Any]:
    model = load_model(args.weights)
    predictor = choose_vp_predictor(model)
    results = model.predict(
        source=args.source,
        visual_prompts=visual_prompts,
        predictor=predictor,
        name="03_prompt_only_hybrid",
        agnostic_nms=False,
        **base_prompt_kwargs(args),
        **hybrid_prompt_kwargs(args),
    )
    summary = summarize_results("03 Prompt-Only Hybrid", results, args.report_limit)
    check_hybrid_expectation("03 Prompt-Only Hybrid", summary, args)
    return summary


def run_single_nms_case(args: argparse.Namespace, visual_prompts: dict[str, Any], agnostic_nms: bool, run_name: str):
    model = load_model(args.weights)
    predictor = choose_vp_predictor(model)
    return model.predict(
        source=args.source,
        refer_image=args.refer_image,
        visual_prompts=visual_prompts,
        predictor=predictor,
        name=run_name,
        agnostic_nms=agnostic_nms,
        **base_prompt_kwargs(args),
        **hybrid_prompt_kwargs(args),
    )


def run_nms_comparison(args: argparse.Namespace, visual_prompts: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    aware_results = run_single_nms_case(
        args,
        visual_prompts,
        agnostic_nms=False,
        run_name="04_nms_class_aware",
    )
    agnostic_results = run_single_nms_case(
        args,
        visual_prompts,
        agnostic_nms=True,
        run_name="05_nms_agnostic",
    )

    aware_summary = summarize_results("04 NMS Compare - Class-Aware", aware_results, args.report_limit)
    agnostic_summary = summarize_results("05 NMS Compare - Agnostic", agnostic_results, args.report_limit)

    print("\n=== 06 NMS Difference Summary ===")
    print(f"class-aware detections={aware_summary['total_detections']}")
    print(f"agnostic detections={agnostic_summary['total_detections']}")

    all_paths = sorted(
        set(aware_summary["image_detection_counts"]).union(agnostic_summary["image_detection_counts"])
    )
    changed = False
    for path in all_paths:
        aware_count = aware_summary["image_detection_counts"].get(path, 0)
        agnostic_count = agnostic_summary["image_detection_counts"].get(path, 0)
        if aware_count != agnostic_count:
            changed = True
            print(f"  {Path(path).name}: class-aware={aware_count}, agnostic={agnostic_count}")
    if not changed:
        print("  no per-image detection-count difference was observed in this run.")

    return aware_summary, agnostic_summary


def main() -> None:
    args = parse_args()
    ensure_local_path(args.source, "Source")
    ensure_local_path(args.refer_image, "Reference image")

    prompt_box = parse_box(args.prompt_box)
    visual_prompts = build_visual_prompts(prompt_box, args.prompt_name)

    print("Validation plan:")
    print("  1. Old-class regression")
    print("  2. Reference-image hybrid inference")
    print("  3. Prompt-only hybrid inference")
    print("  4. NMS comparison")

    run_old_class_regression(args)
    run_reference_hybrid(args, visual_prompts)
    run_prompt_only_hybrid(args, visual_prompts)
    run_nms_comparison(args, visual_prompts)

    print("\nAll validation stages finished.")


if __name__ == "__main__":
    main()
