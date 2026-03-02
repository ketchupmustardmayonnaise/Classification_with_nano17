TAG = {
    0: "FINGERPAD-THUMB",
    1: "FINGERPAD-INDEX",
    2: "FINGERPAD-MIDDLE",
    3: "PALMARSIDEFINGERS-ONE",
    4: "PALMARSIDEFINGERS-TWO",
    5: "PALMARSIDEFINGERS-THREE",
    6: "PALM",
    7: "ROCKDOWN-WRISTSIDE",
    8: "ROCKDOWN-FINGERSIDE",
    9: "ROCKUP",
    10: "MOUTH",
    11: "EAR-PINNA",
    12: "EAR-HOLE"
}

MERGED_MAP = {
    0: 1, # FINGERPAD-THUMB  -> FINGERPAD
    1: 1, # FINGERPAD-INDEX  -> FINGERPAD
    2: 1, # FINGERPAD-MIDDLE -> FINGERPAD
    3: 2,
    4: 2,
    5: 2,
    6: 3,
    7: 4,
    8: 4,
    9: 5,
    10: 6,
    11: 7,
    12: 7
}

MERGED_TAG = {
    0: "IDLE",
    1: "FINGERPAD",
    2: "PALMARMARSIDEFINGERS",
    3: "PALM",
    4: "ROCKDOWN",
    5: "ROCKUP",
    6: "MOUTH",
    7: "EAR",
}

# reverse mapping: 문자열 -> 숫자
TAG_INV = {v: k for k, v in TAG.items()}