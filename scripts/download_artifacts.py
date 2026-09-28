import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

DRIVE_URL = "https://drive.google.com/drive/folders/1IF5ARj6NP964dkD0tFAUGxsGRn33hjMd"
ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
STAGING = DATA / ".gdrive_download"

EXPECTED = [
    ("bm25s_index", "bm25s_index"),
    ("bm25s_index_bench", "bm25s_index_bench"),
    ("bm25s_index_title", "bm25s_index_title"),
    ("bm25s_index_desc", "bm25s_index_desc"),
    ("bm25s_index_params", "bm25s_index_params"),
    ("w311", "bm25s_index_weighted/w311"),
    ("tfidf_char_index", "tfidf_char_index"),
    ("e5_base_enriched", "embeddings/e5_base_enriched"),
    ("e5_base_bench", "embeddings/e5_base_bench"),
    ("query_embeddings_cache", "query_embeddings_cache"),
]


def download():
    STAGING.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["gdown", "--folder", DRIVE_URL, "-O", str(STAGING)],
        check=True,
    )


def extract_zips():
    for zip_path in list(STAGING.rglob("*.zip")):
        with zipfile.ZipFile(zip_path) as archive:
            archive.extractall(zip_path.parent)
        zip_path.unlink()


def place(drive_name, target_rel):
    target = DATA / target_rel
    if target.exists():
        print(f"уже есть: data/{target_rel}")
        return True
    matches = [p for p in STAGING.rglob(drive_name) if p.is_dir()]
    if not matches:
        print(f"НЕ НАЙДЕНО: {drive_name} (ожидался в data/{target_rel})")
        return False
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(matches[0]), str(target))
    print(f"готово: data/{target_rel}")
    return True


def main():
    download()
    extract_zips()
    ok = [place(drive_name, target_rel) for drive_name, target_rel in EXPECTED]
    shutil.rmtree(STAGING, ignore_errors=True)
    if not all(ok):
        sys.exit("Не все артефакты найдены - проверьте содержимое папки на Google Drive вручную.")
    print("Все артефакты на месте.")


if __name__ == "__main__":
    main()
