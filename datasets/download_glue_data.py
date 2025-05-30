import argparse
import io
import os
import shutil
import sys
import urllib.request
import urllib.request as URLLIB
import zipfile

# URL map for GLUE tasks
tasks = ["CoLA", "SST", "MRPC", "QQP", "STS", "MNLI", "QNLI", "RTE", "WNLI", "diagnostic"]
TASK2PATH = {
    "CoLA": 'https://dl.fbaipublicfiles.com/glue/data/CoLA.zip',
    "SST": 'https://dl.fbaipublicfiles.com/glue/data/SST-2.zip',
    "QQP": 'https://dl.fbaipublicfiles.com/glue/data/QQP-clean.zip',
    "STS": 'https://dl.fbaipublicfiles.com/glue/data/STS-B.zip',
    "MNLI": 'https://dl.fbaipublicfiles.com/glue/data/MNLI.zip',
    "QNLI": 'https://dl.fbaipublicfiles.com/glue/data/QNLIv2.zip',
    "RTE": 'https://dl.fbaipublicfiles.com/glue/data/RTE.zip',
    "WNLI": 'https://dl.fbaipublicfiles.com/glue/data/WNLI.zip',
    "diagnostic": 'https://dl.fbaipublicfiles.com/glue/data/AX.tsv'
}

# URLs for MRPC data
default_mrpc_train = 'https://dl.fbaipublicfiles.com/senteval/senteval_data/msr_paraphrase_train.txt'
default_mrpc_test = 'https://dl.fbaipublicfiles.com/senteval/senteval_data/msr_paraphrase_test.txt'
# Fallback dev_ids.tsv sources
DEV_IDS_SOURCES = [
    'https://raw.githubusercontent.com/MegEngine/Models/master/official/nlp/bert/glue_data/MRPC/dev_ids.tsv',
    'https://raw.githubusercontent.com/nyu-mll/GLUE-baselines/master/glue_data/MRPC/dev_ids.tsv'
]


def download_and_extract(task, data_dir):
    print(f"Downloading and extracting {task}...")
    if task == "MNLI":
        print("\tNote: SNLI is no longer included; format manually if needed.")
    zip_path = os.path.join(data_dir, f"{task}.zip")
    urllib.request.urlretrieve(TASK2PATH[task], zip_path)

    # extract
    with zipfile.ZipFile(zip_path, 'r') as z:
        # For SST, rename SST-2 -> sst2
        if task == "SST":
            temp_dir = os.path.join(data_dir, "SST_tmp")
            z.extractall(temp_dir)
            # extracted folder is SST-2
            src = os.path.join(temp_dir, "SST-2")
            dst = os.path.join(data_dir, "sst2")
            if os.path.isdir(dst):
                shutil.rmtree(dst)
            shutil.move(src, dst)
            shutil.rmtree(temp_dir)
        else:
            z.extractall(data_dir)
    os.remove(zip_path)
    print("\tDone.")


def format_mrpc(data_dir, path_to_data):
    print("Processing MRPC...")
    mrpc_dir = os.path.join(data_dir, "MRPC")
    os.makedirs(mrpc_dir, exist_ok=True)

    # determine train/test sources
    if path_to_data:
        train_src = os.path.join(path_to_data, "msr_paraphrase_train.txt")
        test_src = os.path.join(path_to_data, "msr_paraphrase_test.txt")
    else:
        train_src = os.path.join(mrpc_dir, "msr_paraphrase_train.txt")
        test_src = os.path.join(mrpc_dir, "msr_paraphrase_test.txt")
        try:
            URLLIB.urlretrieve(default_mrpc_train, train_src)
            URLLIB.urlretrieve(default_mrpc_test, test_src)
        except Exception as e:
            print(f"Error downloading MRPC data: {e}")
            return

    if not os.path.isfile(train_src) or not os.path.isfile(test_src):
        print(f"Missing train/test files under {mrpc_dir}")
        return

    # write test.tsv
    with io.open(test_src, encoding='utf-8') as fin, \
            io.open(os.path.join(mrpc_dir, 'test.tsv'), 'w', encoding='utf-8') as fout:
        fin.readline()
        fout.write("index\t#1 ID\t#2 ID\t#1 String\t#2 String\n")
        for i, line in enumerate(fin):
            lab, id1, id2, s1, s2 = line.strip().split('\t')
            fout.write(f"{i}\t{id1}\t{id2}\t{s1}\t{s2}\n")

    # ensure dev_ids.tsv present; try multiple sources
    dev_ids_file = os.path.join(mrpc_dir, "dev_ids.tsv")
    if not os.path.isfile(dev_ids_file):
        print("\tFetching dev_ids.tsv from mirrors...")
        for url in DEV_IDS_SOURCES:
            try:
                URLLIB.urlretrieve(url, dev_ids_file)
                print(f"\tDownloaded dev_ids.tsv from {url}")
                break
            except Exception:
                print(f"\tFailed from {url}")
        if not os.path.isfile(dev_ids_file):
            print(f"Error: could not download dev_ids.tsv.\nPlease manually place dev_ids.tsv into {mrpc_dir}")
            return

    # load dev IDs
    dev_pairs = set()
    with io.open(dev_ids_file, encoding='utf-8') as f:
        for row in f:
            a, b = row.strip().split('\t')
            dev_pairs.add((a, b))

    # split train/dev
    with io.open(train_src, encoding='utf-8') as fin, \
            io.open(os.path.join(mrpc_dir, 'train.tsv'), 'w', encoding='utf-8') as fout_tr, \
            io.open(os.path.join(mrpc_dir, 'dev.tsv'), 'w', encoding='utf-8') as fout_dev:
        header = fin.readline()
        fout_tr.write(header)
        fout_dev.write(header)
        for line in fin:
            lab, id1, id2, s1, s2 = line.strip().split('\t')
            if (id1, id2) in dev_pairs:
                fout_dev.write(line)
            else:
                fout_tr.write(line)

    print("\tMRPC formatting complete.")


def download_diagnostic(data_dir):
    print("Downloading diagnostic...")
    diag = os.path.join(data_dir, 'diagnostic')
    os.makedirs(diag, exist_ok=True)
    urllib.request.urlretrieve(TASK2PATH['diagnostic'], os.path.join(diag, 'diagnostic.tsv'))
    print("\tDone.")


def get_tasks(names):
    parts = names.split(',')
    return tasks.copy() if 'all' in parts else [t for t in parts if t in tasks]


def main(argv):
    p = argparse.ArgumentParser()
    p.add_argument('--data_dir', default='glue_data', help='where to save')
    p.add_argument('--tasks', default='all', help='comma-list or all')
    p.add_argument('--path_to_mrpc', default='', help='existing MRPC txts')
    args = p.parse_args(argv)

    os.makedirs(args.data_dir, exist_ok=True)
    for t in get_tasks(args.tasks):
        if t == 'MRPC':
            format_mrpc(args.data_dir, args.path_to_mrpc)
        elif t == 'diagnostic':
            download_diagnostic(args.data_dir)
        else:
            download_and_extract(t, args.data_dir)


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
