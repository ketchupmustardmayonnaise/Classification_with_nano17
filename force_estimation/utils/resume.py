import os

WANDB_RUN_ID_FILENAME = "wandb_run_id.txt"


def find_latest_run_dir(save_directory):
    """save_directory 바로 아래에서 last.ckpt를 가진 폴더 중, 가장 최근에 수정된 걸 찾음."""
    candidates = []
    if os.path.isdir(save_directory):
        for name in os.listdir(save_directory):
            run_dir = os.path.join(save_directory, name)
            ckpt_path = os.path.join(run_dir, "last.ckpt")
            if os.path.isfile(ckpt_path):
                candidates.append((os.path.getmtime(ckpt_path), run_dir))

    if not candidates:
        return None
    return max(candidates, key=lambda item: item[0])[1]


def read_wandb_run_id(save_dir):
    run_id_path = os.path.join(save_dir, WANDB_RUN_ID_FILENAME)
    if os.path.isfile(run_id_path):
        with open(run_id_path) as f:
            return f.read().strip() or None
    return None


def write_wandb_run_id(save_dir, run_id):
    run_id_path = os.path.join(save_dir, WANDB_RUN_ID_FILENAME)
    if not os.path.isfile(run_id_path):
        with open(run_id_path, "w") as f:
            f.write(run_id)
