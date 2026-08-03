"""
test_result.txt (classification) -> 구글 스프레드시트 '[ROTATION AUG][CLASSIFICATION][BODYPARTS]' 탭에
그대로 복붙할 수 있는 컨퓨전 매트릭스 블록 생성.

시트 구조 (직접 확인함):
  - 폴드 블록은 58행 단위로 반복됨: FOLD1&2 -> 1~58행, FOLD3&4 -> 59~116행, FOLD5&6 -> 117~174행
  - 각 블록 안에서 홀수 fold(1,3,5)는 좌측(B열~), 짝수 fold(2,4,6)는 우측(Q열~)
  - "레이블 합치기 전 결과" 14x14 컨퓨전 매트릭스: 블록 시작 기준 5행째, B(or Q)열부터 14칸
    (행=실제 클래스, 열=예측 클래스, 순서는 test_result.txt의 confusion matrix와 동일: IDLE, FINGERPAD-THUMB,
     FINGERPAD-INDEX, FINGERPAD-MIDDLE, PALMARSIDEFINGERS-ONE/TWO/THREE, PALM, ROCKDOWN-WRISTSIDE/FINGERSIDE,
     ROCKUP, MOUTH, EAR-PINNA, EAR-HOLE)
  - "레이블 합친 결과" 8x8 컨퓨전 매트릭스: 블록 시작 기준 38행째, B(or Q)열부터 8칸
    (14개 클래스를 IDLE/FINGERPAD/PALMARSIDEFINGERS/PALM/ROCKDOWN/ROCKUP/MOUTH/EAR 8개로 병합)
  - N(샘플 수)/TP(대각선)/Recall/Precision/F1/Overall Accuracy/Macro,Weighted 통계는 전부 수식(#DIV/0!)
    이므로 이 스크립트는 건드리지 않음. 컨퓨전 매트릭스만 채우면 자동으로 계산됨.

사용법:
  python test_classification_result_to_sheet.py <test_result.txt 경로> --fold N [--out-dir DIR]
"""

import argparse
import re
import sys
from pathlib import Path

CLASS_ORDER_14 = [
    "IDLE",
    "FINGERPAD-THUMB",
    "FINGERPAD-INDEX",
    "FINGERPAD-MIDDLE",
    "PALMARSIDEFINGERS-ONE",
    "PALMARSIDEFINGERS-TWO",
    "PALMARSIDEFINGERS-THREE",
    "PALM",
    "ROCKDOWN-WRISTSIDE",
    "ROCKDOWN-FINGERSIDE",
    "ROCKUP",
    "MOUTH",
    "EAR-PINNA",
    "EAR-HOLE",
]

GROUP_OF_CLASS = {
    "IDLE": "IDLE",
    "FINGERPAD-THUMB": "FINGERPAD",
    "FINGERPAD-INDEX": "FINGERPAD",
    "FINGERPAD-MIDDLE": "FINGERPAD",
    "PALMARSIDEFINGERS-ONE": "PALMARSIDEFINGERS",
    "PALMARSIDEFINGERS-TWO": "PALMARSIDEFINGERS",
    "PALMARSIDEFINGERS-THREE": "PALMARSIDEFINGERS",
    "PALM": "PALM",
    "ROCKDOWN-WRISTSIDE": "ROCKDOWN",
    "ROCKDOWN-FINGERSIDE": "ROCKDOWN",
    "ROCKUP": "ROCKUP",
    "MOUTH": "MOUTH",
    "EAR-PINNA": "EAR",
    "EAR-HOLE": "EAR",
}

GROUP_ORDER_8 = ["IDLE", "FINGERPAD", "PALMARSIDEFINGERS", "PALM", "ROCKDOWN", "ROCKUP", "MOUTH", "EAR"]

MATRIX_ROW_BASE = 5
MERGED_ROW_BASE = 38
PAIR_BLOCK_SIZE = 58


def parse_confusion_matrix(text):
    marker = "[Confusion Matrix]"
    idx = text.find(marker)
    if idx == -1:
        raise ValueError("'[Confusion Matrix]' 섹션을 찾지 못했습니다.")

    lines = text[idx:].splitlines()[1:]  # marker 줄 자체는 건너뜀
    matrix = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            if matrix:
                break
            continue
        nums = [int(x) for x in re.findall(r"-?\d+", stripped)]
        if len(nums) != len(CLASS_ORDER_14):
            raise ValueError(
                f"컨퓨전 매트릭스 한 줄에서 {len(CLASS_ORDER_14)}개 숫자를 기대했는데 {len(nums)}개 찾음: {line!r}"
            )
        matrix.append(nums)
        if len(matrix) == len(CLASS_ORDER_14):
            break

    if len(matrix) != len(CLASS_ORDER_14):
        raise ValueError(f"컨퓨전 매트릭스 행을 {len(CLASS_ORDER_14)}개 찾지 못했습니다 (찾은 개수: {len(matrix)}).")

    return matrix


def build_merged_matrix(matrix_14):
    n = len(GROUP_ORDER_8)
    merged = [[0] * n for _ in range(n)]
    group_index = {g: i for i, g in enumerate(GROUP_ORDER_8)}

    for i, class_i in enumerate(CLASS_ORDER_14):
        gi = group_index[GROUP_OF_CLASS[class_i]]
        for j, class_j in enumerate(CLASS_ORDER_14):
            gj = group_index[GROUP_OF_CLASS[class_j]]
            merged[gi][gj] += matrix_14[i][j]

    return merged


def matrix_to_tsv(matrix):
    return "\n".join("\t".join(str(v) for v in row) for row in matrix)


def col_letter_for_fold(fold):
    return "B" if fold % 2 == 1 else "Q"


def row_offset_for_fold(fold):
    pair_index = (fold - 1) // 2
    return pair_index * PAIR_BLOCK_SIZE


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("test_result_path", type=Path, help="test_result.txt (classification) 경로")
    parser.add_argument("--fold", type=int, choices=range(1, 7), required=True,
                         help="FOLD 번호(1~6). 붙여넣을 정확한 셀 위치를 계산하는 데 필요함")
    parser.add_argument("--out-dir", type=Path, default=None,
                         help="출력 파일 저장 위치 (기본값: test_result.txt와 같은 폴더)")
    args = parser.parse_args()

    text = args.test_result_path.read_text(encoding="utf-8")

    matrix_14 = parse_confusion_matrix(text)
    merged_8 = build_merged_matrix(matrix_14)

    out_dir = args.out_dir or args.test_result_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    matrix_path = out_dir / f"paste_confusion_matrix_fold{args.fold}.txt"
    merged_path = out_dir / f"paste_merged_confusion_matrix_fold{args.fold}.txt"

    matrix_path.write_text(matrix_to_tsv(matrix_14), encoding="utf-8")
    merged_path.write_text(matrix_to_tsv(merged_8), encoding="utf-8")

    col = col_letter_for_fold(args.fold)
    offset = row_offset_for_fold(args.fold)
    matrix_cell = f"{col}{MATRIX_ROW_BASE + offset}"
    merged_cell = f"{col}{MERGED_ROW_BASE + offset}"

    print("작성 완료:")
    print(f"  {matrix_path}  -> 시트 '{matrix_cell}' 셀에 붙여넣기 (레이블 합치기 전, 14x14)")
    print(f"  {merged_path}  -> 시트 '{merged_cell}' 셀에 붙여넣기 (레이블 합친 결과, 8x8)")
    print("  두 파일 다 탭으로 구분된 표라서, 한 셀 선택하고 그대로 Ctrl+V 하면 전체 행렬이 채워집니다.")
    print("  N/TP/Recall/Precision/F1/Overall Accuracy/Macro,Weighted 값은 전부 수식이라 건드리지 않았습니다.")


if __name__ == "__main__":
    main()
