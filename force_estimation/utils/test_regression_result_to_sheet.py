"""
test_result.txt -> 구글 스프레드시트 '[ROTATION AUG][REGRESSION][BODYPARTS + FINGERPAD]' 탭에
그대로 복붙할 수 있는 텍스트 블록 생성.

시트 구조 (직접 확인함):
  - B열: 라벨(R2 SCORE/RMSE/MAE/X/Y/Z), C~H열: FOLD1~FOLD6, I열(All)/J열(STD): 수식이므로 건드리지 않음
  - 7~20행: 상단 요약 블록 (RMSE, MAE, R2 각각 Overall+X+Y+Z, 중간에 공백행)
  - 22~190행: bodypart별 블록 13개. 각 블록 = 이름행(공백 1줄) + R2/RMSE/MAE 각 Overall+X+Y+Z(12줄)
    bodypart 순서는 test_result.txt의 "REPORT FOR EACH BODYPART" 순서와 동일함.

사용법:
  python test_regression_result_to_sheet.py <test_result.txt 경로> [--fold N] [--out-dir DIR]

출력:
  <out-dir>/paste_top_summary.txt   -> 시트의 C7(또는 해당 fold 열)7행에 붙여넣기
  <out-dir>/paste_bodyparts_table.txt -> 시트의 C22(또는 해당 fold 열)22행에 붙여넣기
"""

import argparse
import re
import sys
from pathlib import Path

BODYPART_ORDER = [
    "EAR-HOLE",
    "EAR-PINNA",
    "FINGERPAD-INDEX",
    "FINGERPAD-MIDDLE",
    "FINGERPAD-THUMB",
    "MOUTH",
    "PALM",
    "PALMARSIDEFINGERS-ONE",
    "PALMARSIDEFINGERS-THREE",
    "PALMARSIDEFINGERS-TWO",
    "ROCKDOWN-FINGERSIDE",
    "ROCKDOWN-WRISTSIDE",
    "ROCKUP",
]

FOLD_COLUMN = {1: "C", 2: "D", 3: "E", 4: "F", 5: "G", 6: "H"}

NUM = r"[-+]?\d+\.\d+"

TOP_PATTERNS = {
    "r2": re.compile(rf"^\[R2 SCORE\] Overall: ({NUM}) Fx: ({NUM}) Fy: ({NUM}) Fz: ({NUM})"),
    "rmse": re.compile(rf"^\[RMSE\] Overall: ({NUM}) Fx: ({NUM}) Fy: ({NUM}) Fz: ({NUM})"),
    "mae": re.compile(rf"^\[MAE\] Overall: ({NUM}) Fx: ({NUM}) Fy: ({NUM}) Fz: ({NUM})"),
}

BODYPART_HEADER_RE = re.compile(
    rf"^\[([A-Z0-9\-]+)\] R2 SCORE: ({NUM}) RMSE: ({NUM}) MAE: ({NUM})"
)
R2_AXIS_RE = re.compile(rf"^\(R2 SCORE\) X: ({NUM}), Y: ({NUM}), Z: ({NUM})")
RMSE_AXIS_RE = re.compile(rf"^\(RMSE\) X: ({NUM}), Y: ({NUM}), Z: ({NUM})")
MAE_AXIS_RE = re.compile(rf"^\(MAE\) X: ({NUM}), Y: ({NUM}), Z: ({NUM})")


def parse_top_summary(lines):
    values = {}
    for line in lines[:3]:
        for key, pattern in TOP_PATTERNS.items():
            m = pattern.match(line.strip())
            if m:
                values[key] = m.groups()
    missing = [k for k in ("r2", "rmse", "mae") if k not in values]
    if missing:
        print(f"[경고] 상단 요약에서 못 찾은 항목: {missing}", file=sys.stderr)
    return values


def parse_bodyparts(text):
    marker = "REPORT FOR EACH BODYPART"
    idx = text.find(marker)
    if idx == -1:
        print("[경고] 'REPORT FOR EACH BODYPART' 섹션을 찾지 못했습니다.", file=sys.stderr)
        return {}

    section_lines = text[idx:].splitlines()
    result = {}
    i = 0
    while i < len(section_lines):
        header_m = BODYPART_HEADER_RE.match(section_lines[i].strip())
        if header_m:
            name, r2, rmse, mae = header_m.groups()
            r2x = r2y = r2z = rmsex = rmsey = rmsez = maex = maey = maez = ""
            for j in range(i + 1, min(i + 4, len(section_lines))):
                s = section_lines[j].strip()
                m = R2_AXIS_RE.match(s)
                if m:
                    r2x, r2y, r2z = m.groups()
                    continue
                m = RMSE_AXIS_RE.match(s)
                if m:
                    rmsex, rmsey, rmsez = m.groups()
                    continue
                m = MAE_AXIS_RE.match(s)
                if m:
                    maex, maey, maez = m.groups()
            result[name] = {
                "r2": r2, "r2x": r2x, "r2y": r2y, "r2z": r2z,
                "rmse": rmse, "rmsex": rmsex, "rmsey": rmsey, "rmsez": rmsez,
                "mae": mae, "maex": maex, "maey": maey, "maez": maez,
            }
        i += 1
    return result


def build_top_block(top_values):
    def row(key):
        return top_values.get(key, ("", "", "", ""))

    rmse_o, rmse_x, rmse_y, rmse_z = row("rmse")
    mae_o, mae_x, mae_y, mae_z = row("mae")
    r2_o, r2_x, r2_y, r2_z = row("r2")

    return [
        rmse_o, rmse_x, rmse_y, rmse_z, "",
        mae_o, mae_x, mae_y, mae_z, "",
        r2_o, r2_x, r2_y, r2_z,
    ]


def build_bodypart_block(bodypart_values):
    lines = []
    for name in BODYPART_ORDER:
        values = bodypart_values.get(name)
        if values is None:
            print(f"[경고] test_result.txt에 '{name}' bodypart가 없어서 빈 칸으로 채웁니다.", file=sys.stderr)
            values = {k: "" for k in (
                "r2", "r2x", "r2y", "r2z", "rmse", "rmsex", "rmsey", "rmsez", "mae", "maex", "maey", "maez"
            )}
        lines.append("")  # bodypart 이름행 (건드리지 않음)
        lines.extend([
            values["r2"], values["r2x"], values["r2y"], values["r2z"],
            values["rmse"], values["rmsex"], values["rmsey"], values["rmsez"],
            values["mae"], values["maex"], values["maey"], values["maez"],
        ])
    return lines


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("test_result_path", type=Path, help="test_result.txt 경로")
    parser.add_argument("--fold", type=int, choices=range(1, 7), default=None,
                         help="FOLD 번호(1~6). 지정하면 붙여넣을 셀 위치를 안내에 표시함")
    parser.add_argument("--out-dir", type=Path, default=None,
                         help="출력 파일 저장 위치 (기본값: test_result.txt와 같은 폴더)")
    args = parser.parse_args()

    text = args.test_result_path.read_text(encoding="utf-8")
    lines = text.splitlines()

    top_values = parse_top_summary(lines)
    bodypart_values = parse_bodyparts(text)

    top_block = build_top_block(top_values)
    bodypart_block = build_bodypart_block(bodypart_values)

    out_dir = args.out_dir or args.test_result_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    top_path = out_dir / "paste_top_summary.txt"
    bodypart_path = out_dir / "paste_bodyparts_table.txt"

    top_path.write_text("\n".join(top_block), encoding="utf-8")
    bodypart_path.write_text("\n".join(bodypart_block), encoding="utf-8")

    col = FOLD_COLUMN.get(args.fold) if args.fold else "?"
    print(f"작성 완료:")
    print(f"  {top_path}  -> 시트 '{col}7' 셀에 붙여넣기 (7~20행)")
    print(f"  {bodypart_path}  -> 시트 '{col}22' 셀에 붙여넣기 (22~190행)")
    print("  (I열 'All', J열 'STD'는 수식이라 건드리지 마세요)")
    if args.fold is None:
        print("  참고: --fold 1~6 을 지정하면 정확한 열(C~H)을 안내해 줍니다.")


if __name__ == "__main__":
    main()
