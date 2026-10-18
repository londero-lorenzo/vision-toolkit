import argparse
import json
from pathlib import Path

import cv2

from image_segmentation.board import detect_board, load_quad, rectify_homography, save_quad
from image_segmentation.person import PersonSegmenter
from image_segmentation.reconstruction import BoardReconstructor
from image_segmentation.video import format_time, iter_frames, median_frame


def parse_crop(value):
    return tuple(int(v) for v in value.split(",")) if value else None


def main():
    parser = argparse.ArgumentParser(description="Rimuove il docente dalla lavagna in un video di lezione.")
    parser.add_argument("video", type=Path)
    parser.add_argument("--out", type=Path, default=Path("output"))
    parser.add_argument("--step", type=float, default=1.0, help="secondi tra due frame analizzati")
    parser.add_argument("--start", type=float, default=0.0)
    parser.add_argument("--end", type=float, default=None)
    parser.add_argument("--crop", type=str, default=None, help="x,y,w,h dell'area del video")
    parser.add_argument("--quad", type=Path, default=None, help="JSON con i 4 angoli della lavagna")
    parser.add_argument("--window", type=int, default=7, help="campioni nella mediana temporale")
    parser.add_argument("--min-change", type=float, default=0.004,
                        help="frazione di pixel cambiati per salvare un nuovo snapshot")
    parser.add_argument("--refine", action="store_true", help="unisce DeepLab sul ritaglio alla maschera della posa")
    args = parser.parse_args()

    out = args.out / args.video.stem
    out.mkdir(parents=True, exist_ok=True)
    crop = parse_crop(args.crop)

    if args.quad:
        quad = load_quad(args.quad)
    else:
        background = median_frame(args.video, crop=crop)
        cv2.imwrite(str(out / "median.png"), background)
        quad = detect_board(background)
        if quad is None:
            raise SystemExit("Lavagna non trovata: passa --quad con gli angoli a mano.")
        save_quad(out / "quad.json", quad)

    H, size = rectify_homography(quad)
    rec = BoardReconstructor(H, size, window=args.window)
    prev_change = None

    with PersonSegmenter(refine=args.refine) as segment, open(out / "pointing.jsonl", "w", encoding="utf-8") as log:
        snapshot = None
        for t, frame in iter_frames(args.video, args.step, args.start, args.end, crop):
            person = segment(frame)
            occluded = rec.update(frame, person.mask)

            # Snapshot quando la lavagna è cambiata abbastanza e si è stabilizzata
            change = rec.change_since_snapshot()
            if change >= args.min_change and prev_change is not None and abs(change - prev_change) < args.min_change / 4:
                snapshot = f"board_{format_time(t)}.png"
                cv2.imwrite(str(out / snapshot), rec.take_snapshot())
                change = 0.0
            prev_change = change

            board_xy = rec.to_board(person.pointer)
            if person.landmarks is not None:
                log.write(json.dumps({
                    "t": round(t, 2),
                    "frame_xy": person.pointer,
                    "board_xy": board_xy,
                    "board_xy_norm": [round(board_xy[0] / size[0], 4), round(board_xy[1] / size[1], 4)] if board_xy else None,
                    "arm_angle": round(person.arm_angle, 1),
                    "occluded_fraction": round(float(occluded.mean()), 4),
                    "snapshot": snapshot,
                }) + "\n")

    cv2.imwrite(str(out / "board_final.png"), rec.image())
    print(f"Output in {out}")


if __name__ == "__main__":
    main()
