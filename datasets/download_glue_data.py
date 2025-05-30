import argparse
import io
import os
import shutil
import sys
import urllib.request
import urllib.request as URLLIB
import zipfile

# GLUE task names (lowercase) and URLs
TASKS = ["cola", "sst2", "mrpc", "qqp", "sts", "mnli", "qnli", "rte", "wnli", "diagnostic"]
ZIP_URL = {
    "cola": 'https://dl.fbaipublicfiles.com/glue/data/CoLA.zip',
    "sst2": 'https://dl.fbaipublicfiles.com/glue/data/SST-2.zip',
    "qqp":  'https://dl.fbaipublicfiles.com/glue/data/QQP-clean.zip',
    "sts":  'https://dl.fbaipublicfiles.com/glue/data/STS-B.zip',
    "mnli": 'https://dl.fbaipublicfiles.com/glue/data/MNLI.zip',
    "qnli": 'https://dl.fbaipublicfiles.com/glue/data/QNLIv2.zip',
    "rte":  'https://dl.fbaipublicfiles.com/glue/data/RTE.zip',
    "wnli": 'https://dl.fbaipublicfiles.com/glue/data/WNLI.zip',
}
DIAG_URL       = 'https://dl.fbaipublicfiles.com/glue/data/AX.tsv'
MRPC_TRAIN     = 'https://dl.fbaipublicfiles.com/senteval/senteval_data/msr_paraphrase_train.txt'
MRPC_TEST      = 'https://dl.fbaipublicfiles.com/senteval/senteval_data/msr_paraphrase_test.txt'
DEV_IDS_SOURCES = [
    'https://raw.githubusercontent.com/MegEngine/Models/master/official/nlp/bert/glue_data/MRPC/dev_ids.tsv',
    'https://raw.githubusercontent.com/nyu-mll/GLUE-baselines/master/glue_data/MRPC/dev_ids.tsv'
]

def fetch_and_unzip(task: str, data_dir: str):
    """Download ZIP for `task`, extract into <data_dir>/<task>/."""
    url    = ZIP_URL[task]
    zip_fp = os.path.join(data_dir, f"{task}.zip")
    outdir = os.path.join(data_dir, task)

    print(f"→ {task}: downloading ZIP…")
    urllib.request.urlretrieve(url, zip_fp)

    tmp = os.path.join(data_dir, f"_{task}_tmp")
    if os.path.isdir(tmp):
        shutil.rmtree(tmp)
    os.makedirs(tmp, exist_ok=True)

    with zipfile.ZipFile(zip_fp, 'r') as zf:
        zf.extractall(tmp)
    os.remove(zip_fp)

    entries_top = os.listdir(tmp)
    if len(entries_top)==1 and os.path.isdir(os.path.join(tmp, entries_top[0])):
        base = os.path.join(tmp, entries_top[0])
        names = os.listdir(base)
    else:
        base = tmp
        names = entries_top

    if os.path.isdir(outdir):
        shutil.rmtree(outdir)
    os.makedirs(outdir, exist_ok=True)

    for name in names:
        shutil.move(os.path.join(base, name), os.path.join(outdir, name))
    shutil.rmtree(tmp)

    print(f"→ {task}: extracted into `{outdir}/`")


def format_mrpc(data_dir: str, mrpc_raw: str):
    out = os.path.join(data_dir, "mrpc")
    os.makedirs(out, exist_ok=True)

    if mrpc_raw:
        train_src = os.path.join(mrpc_raw, "msr_paraphrase_train.txt")
        test_src  = os.path.join(mrpc_raw, "msr_paraphrase_test.txt")
    else:
        print("→ mrpc: downloading raw…")
        train_src = os.path.join(out, "msr_paraphrase_train.txt")
        test_src  = os.path.join(out, "msr_paraphrase_test.txt")
        URLLIB.urlretrieve(MRPC_TRAIN, train_src)
        URLLIB.urlretrieve(MRPC_TEST,  test_src)

    print("→ mrpc: formatting test.tsv…")
    with io.open(test_src, encoding='utf-8') as fin, \
            io.open(os.path.join(out, "test.tsv"), 'w', encoding='utf-8') as fout:
        fin.readline()
        fout.write("index\t#1 ID\t#2 ID\t#1 String\t#2 String\n")
        for idx, line in enumerate(fin):
            lbl,i1,i2,s1,s2 = line.strip().split('\t')
            fout.write(f"{idx}\t{i1}\t{i2}\t{s1}\t{s2}\n")

    dev_ids = os.path.join(out, "dev_ids.tsv")
    if not os.path.isfile(dev_ids):
        print("→ mrpc: fetching dev_ids.tsv…")
        for url in DEV_IDS_SOURCES:
            try:
                URLLIB.urlretrieve(url, dev_ids)
                break
            except:
                continue
        if not os.path.isfile(dev_ids):
            print(f"!! mrpc: failed to fetch dev_ids.tsv; place it in {out}")
            return

    pairs = set()
    with io.open(dev_ids, encoding='utf-8') as f:
        for line in f:
            a,b = line.strip().split('\t')
            pairs.add((a,b))

    print("→ mrpc: splitting train/dev…")
    with io.open(train_src, encoding='utf-8') as fin, \
            io.open(os.path.join(out, "train.tsv"), 'w', encoding='utf-8') as tr, \
            io.open(os.path.join(out, "dev.tsv"),   'w', encoding='utf-8') as dv:
        header = fin.readline()
        tr.write(header)
        dv.write(header)
        for line in fin:
            lbl,i1,i2,s1,s2 = line.strip().split('\t')
            (dv if (i1,i2) in pairs else tr).write(line)

    print("→ mrpc: done.")


def fetch_diagnostic(data_dir: str):
    out = os.path.join(data_dir, "diagnostic")
    os.makedirs(out, exist_ok=True)
    dest = os.path.join(out, "diagnostic.tsv")
    print("→ diagnostic: downloading…")
    urllib.request.urlretrieve(DIAG_URL, dest)
    print("→ diagnostic: saved.")


def main(argv):
    # script_dir = folder this .py lives in
    script_dir = os.path.dirname(os.path.realpath(__file__))

    p = argparse.ArgumentParser()
    p.add_argument(
        "--data_dir",
        default=script_dir,
        help="where to create <task>/ under (default: ./datasets/ next to this script)"
    )
    p.add_argument(
        "--tasks",
        default="all",
        help="comma-separated subset of: " + ",".join(TASKS)
    )
    p.add_argument(
        "--path_to_mrpc",
        default="",
        help="existing MRPC txts dir (optional)"
    )
    args = p.parse_args(argv)

    data_root = args.data_dir
    os.makedirs(data_root, exist_ok=True)

    req = args.tasks.lower().split(",")
    to_do = TASKS if "all" in req else [t for t in req if t in TASKS]

    for task in to_do:
        if task == "mrpc":
            format_mrpc(data_root, args.path_to_mrpc)
        elif task == "diagnostic":
            fetch_diagnostic(data_root)
        else:
            fetch_and_unzip(task, data_root)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
