import io
import json
import multiprocessing
import os
import shutil
import sys
import tarfile
import time
import zipfile
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
from webdataset import tenbin

KEYS = [
    "binaryInputNCHWPacked",
    "globalInputNC",
    "policyTargetsNCMove",
    "globalTargetsNC",
    "scoreDistrN",
    "valueTargetsNCHW",
]


def get_summary_data(summary_file):
    # Try a bunch of times, just to be robust to if the file is being swapped out in nfs
    for i in range(10):
        success = False
        try:
            with open(summary_file) as fp:
                summary_data = json.load(fp)
                success = True
        except OSError:
            success = False
        except ValueError:
            success = False
        if success:
            break
    return summary_data


def get_file_info(dirs, summary_data):
    all_files = []
    files_with_unknown_num_rows = []
    for d in dirs:
        for path in Path(d).iterdir():
            if path.is_dir():
                file_info = summary_data.get(str(path.resolve()))
                if file_info is not None:
                    for (filename, mtime, num_rows) in file_info:
                        filename = str(path / filename)
                        all_files.append((filename, mtime, num_rows))
            if path.is_file():
                if path.suffix == ".npz":
                    files_with_unknown_num_rows.append(str(path))
                    all_files.append((str(path), path.stat().st_mtime))
    print(f"Total number of files: {len(all_files)}", flush=True)
    print(
        f"Total number of files with unknown row count: {len(files_with_unknown_num_rows)}",
        flush=True,
    )
    return all_files, files_with_unknown_num_rows


def get_numpy_npz_headers(filename):
    with zipfile.ZipFile(filename) as z:
        wasbad = False
        npzheaders = {}
        for subfilename in z.namelist():
            npyfile = z.open(subfilename)
            try:
                version = np.lib.format.read_magic(npyfile)
            except ValueError:
                wasbad = True
                print(
                    f"WARNING: bad file, skipping it: {filename} (bad array {subfilename})"
                )
            else:
                (shape, is_fortran, dtype) = np.lib.format._read_array_header(
                    npyfile, version
                )
                npzheaders[subfilename] = (shape, is_fortran, dtype)
        if wasbad:
            return None
        return npzheaders


def compute_num_rows(filename):
    npheaders = get_numpy_npz_headers(filename)
    if npheaders is None or len(npheaders) <= 0:
        return (filename, 0)
    (shape, is_fortran, dtype) = npheaders["binaryInputNCHWPacked"]
    num_rows = shape[0]
    return (filename, num_rows)


def compute_all_num_rows(all_files, files_with_unknown_num_rows, num_processes):
    with multiprocessing.Pool(num_processes) as pool:
        results = pool.map(compute_num_rows, files_with_unknown_num_rows)
        results = dict(results)
        for i, info in enumerate(all_files):
            if len(info) < 3:
                # filename, modification time, num_rows
                all_files[i] = (info[0], info[1], results[info[0]])
    return all_files


def compute_desired_files(files_with_row_range, desired_num_rows):
    num_rows_total = 0
    # (filename, num_rows)
    desired_input_files = []
    # (filename, (start_row, end_row))
    desired_input_files_with_row_range = []
    print_stride = 1 + len(files_with_row_range) // 40
    for i in range(len(files_with_row_range)):
        (filename, (start_row, end_row)) = files_with_row_range[i]

        # This could happen if the .summary.json file is inaccurate after file deletions
        if not Path(filename).exists():
            continue

        desired_input_files.append((filename, end_row - start_row))
        desired_input_files_with_row_range.append((filename, (start_row, end_row)))

        num_rows_total += end_row - start_row
        if (
            i % print_stride == 0
            or num_rows_total >= desired_num_rows
            or i == len(files_with_row_range) - 1
        ):
            print(
                f"Using: {filename} ({start_row}-{end_row}) "
                f"({num_rows_total}/{desired_num_rows} desired rows)"
            )
        if num_rows_total >= desired_num_rows:
            break
    return num_rows_total, desired_input_files, desired_input_files_with_row_range


def group_input_files(input_files, worker_group_size):
    groups = []
    group_size_so_far = 0
    group_so_far = []
    for (input_file, num_rows) in input_files:
        if num_rows <= 0:
            continue
        group_so_far.append(input_file)
        group_size_so_far += num_rows
        if group_size_so_far >= worker_group_size:
            groups.append(group_so_far)
            group_so_far = []
            group_size_so_far = 0
    if group_size_so_far > 0:
        groups.append(group_so_far)
    return groups


def joint_shuffle(arrs):
    rand_state = np.random.get_state()
    for arr in arrs:
        assert len(arr) == len(arrs[0])
    for arr in arrs:
        np.random.set_state(rand_state)
        np.random.shuffle(arr)


def distribute_shard_data(
    group_idx, input_file_group, num_out_groups, out_tmp_dirs, keep_prob
):
    np.random.seed(
        [int.from_bytes(os.urandom(4), byteorder="little") for i in range(4)]
    )

    assert len(input_file_group) > 0

    list_binchwp = []
    list_ginc = []
    list_ptncm = []
    list_gtnc = []
    list_sdn = []
    list_vtnchw = []

    for input_file in input_file_group:
        with np.load(input_file) as npz:
            assert set(npz.keys()) == set(KEYS)
            list_binchwp.append(npz["binaryInputNCHWPacked"])
            list_ginc.append(npz["globalInputNC"])
            list_ptncm.append(npz["policyTargetsNCMove"])
            list_gtnc.append(npz["globalTargetsNC"])
            list_sdn.append(npz["scoreDistrN"])
            list_vtnchw.append(npz["valueTargetsNCHW"])

    binchwp = np.vstack(tuple(list_binchwp))
    ginc = np.vstack(tuple(list_ginc))
    ptncm = np.vstack(tuple(list_ptncm))
    gtnc = np.vstack(tuple(list_gtnc))
    sdn = np.vstack(tuple(list_sdn))
    vtnchw = np.vstack(tuple(list_vtnchw))

    joint_shuffle((binchwp, ginc, ptncm, gtnc, sdn, vtnchw))

    num_rows_to_keep = binchwp.shape[0]
    assert ginc.shape[0] == num_rows_to_keep
    assert ptncm.shape[0] == num_rows_to_keep
    assert gtnc.shape[0] == num_rows_to_keep
    assert sdn.shape[0] == num_rows_to_keep
    assert vtnchw.shape[0] == num_rows_to_keep

    if keep_prob < 1.0:
        num_rows_to_keep = min(
            num_rows_to_keep, int(round(num_rows_to_keep * keep_prob))
        )

    rand_assts = np.random.randint(num_out_groups, size=[num_rows_to_keep])
    counts = np.bincount(rand_assts, minlength=num_out_groups)
    countsums = np.cumsum(counts)
    assert countsums[len(countsums) - 1] == num_rows_to_keep

    for out_idx in range(num_out_groups):
        start = countsums[out_idx] - counts[out_idx]
        stop = countsums[out_idx]
        np.savez_compressed(
            out_tmp_dirs[out_idx] / f"{group_idx}.npz",
            binaryInputNCHWPacked=binchwp[start:stop],
            globalInputNC=ginc[start:stop],
            policyTargetsNCMove=ptncm[start:stop],
            globalTargetsNC=gtnc[start:stop],
            scoreDistrN=sdn[start:stop],
            valueTargetsNCHW=vtnchw[start:stop],
        )

    return counts


def add_file_to_tar(dst, name, stream):
    assert isinstance(name, str), type(name)

    info = tarfile.TarInfo(name=name)
    info.size = stream.getbuffer().nbytes
    info.uname, info.gname = "KataGo", "KataGo"
    info.mtime = time.time()

    stream.seek(0)
    dst.addfile(info, fileobj=stream)


def write_shard(filename, arrays, compression):
    binchwp = arrays["binchwp"]
    ginc = arrays["ginc"]
    ptncm = arrays["ptncm"]
    gtnc = arrays["gtnc"]
    sdn = arrays["sdn"]
    vtnchw = arrays["vtnchw"]

    mode = "w|"
    if compression != "None":
        mode += compression

    with tarfile.open(str(filename), mode) as dst:
        for i in range(binchwp.shape[0]):
            sample = [
                binchwp[i],
                ginc[i],
                ptncm[i],
                gtnc[i],
                sdn[i],
                vtnchw[i],
            ]
            stream = io.BytesIO()
            tenbin.write(stream, sample)

            tenname = f"{filename.stem}-{i:06d}.ten"
            add_file_to_tar(dst, tenname, stream)


def shardify_group(
    out_dir, shard_start_index, num_shards, out_tmp_dir, shard_size, compression,
):
    list_binchwp = []
    list_ginc = []
    list_ptncm = []
    list_gtnc = []
    list_sdn = []
    list_vtnchw = []

    for shard_filename in out_tmp_dir.glob("*.npz"):
        with np.load(shard_filename) as npz:
            assert set(npz.keys()) == set(KEYS)
            list_binchwp.append(npz["binaryInputNCHWPacked"])
            list_ginc.append(npz["globalInputNC"])
            list_ptncm.append(npz["policyTargetsNCMove"].astype(np.float32))
            list_gtnc.append(npz["globalTargetsNC"])
            list_sdn.append(npz["scoreDistrN"].astype(np.float32))
            list_vtnchw.append(npz["valueTargetsNCHW"].astype(np.float32))

    binchwp = np.vstack(tuple(list_binchwp))
    ginc = np.vstack(tuple(list_ginc))
    ptncm = np.vstack(tuple(list_ptncm))
    gtnc = np.vstack(tuple(list_gtnc))
    sdn = np.vstack(tuple(list_sdn))
    vtnchw = np.vstack(tuple(list_vtnchw))

    num_rows = binchwp.shape[0]
    assert num_rows >= num_shards * shard_size

    joint_shuffle((binchwp, ginc, ptncm, gtnc, sdn, vtnchw))

    for i in range(num_shards):
        filename = Path(out_dir) / f"data{shard_start_index + i}.tar"
        start_idx = shard_size * i
        end_idx = shard_size * (i + 1)
        write_shard(
            filename,
            {
                "binchwp": binchwp[start_idx:end_idx],
                "ginc": ginc[start_idx:end_idx],
                "ptncm": ptncm[start_idx:end_idx],
                "gtnc": gtnc[start_idx:end_idx],
                "sdn": sdn[start_idx:end_idx],
                "vtnchw": vtnchw[start_idx:end_idx],
            },
            compression,
        )

    return num_shards * shard_size


class TimeStuff(object):
    def __init__(self, taskstr):
        self.taskstr = taskstr

    def __enter__(self):
        print(f"Beginning: {self.taskstr}", flush=True)
        self.t0 = time.time()

    def __exit__(self, exception_type, exception_val, trace):
        self.t1 = time.time()
        elapsed_time = self.t1 - self.t0
        print(f"Finished: {self.taskstr} in {str(elapsed_time)} seconds", flush=True)
        return True


class TrainingWindow(object):
    def __init__(self, args):
        # parameters
        self.min_rows = args.min_rows
        self.max_rows = args.max_rows
        self.add_to_window = args.add_to_window
        self.exponent = args.taper_window_exponent
        self.expand_window_per_row = args.expand_window_per_row
        # How far offset do we start on the power-law window tail?
        # E.g. what number of postrandom rows do we need before the window size grows
        # by a factor of 2^(taper_window_exponent)?
        # For now, we set it equal to the min rows0
        if args.taper_window_scale is not None:
            self.offset = args.taper_window_scale
        else:
            self.offset = self.min_rows

        # internal variables
        # Number of data rows
        self.num_total_rows = 0
        # Number of random data rows, capped at min_rows
        # We never keep more than min_rows many data rows if they're from random.
        self.num_random_rows = 0
        # Number of NON-random rows
        self.num_postrandom_rows = 0
        self.desired_num_rows = 0

    def num_usable_rows(self):
        return self.num_random_rows + self.num_postrandom_rows

    def num_desired_rows(self):
        # Every post-random row moves one row beyond window_taper_offset
        base = self.num_usable_rows() - self.min_rows + self.offset + self.add_to_window
        # Apply power law
        power_term = base ** self.exponent
        # Correct for window_taper_offset so we're still anchored at 0
        power_term -= self.offset ** self.exponent
        # Scale so that we have an initial derivative of 1
        power_term /= self.exponent * (self.offset ** (self.exponent - 1))
        # Scale so that we have the desired initial slope, and add back the minimum random rows
        return int(power_term * self.expand_window_per_row + self.min_rows)

    def get_files_to_use(self, all_files):
        files_with_row_range = []
        for (filename, mtime, num_rows) in all_files:
            if num_rows <= 0:
                continue
            row_range = (self.num_total_rows, self.num_total_rows + num_rows)
            files_with_row_range.append((filename, row_range))

            self.num_total_rows += num_rows
            if "random" not in filename:
                self.num_postrandom_rows += num_rows
            else:
                self.num_random_rows += num_rows
                self.num_random_rows = min(self.num_random_rows, self.min_rows)

            # If we already have a window size bigger than max, then just stop
            if self.max_rows is not None and self.num_desired_rows() >= self.max_rows:
                break
        # Reverse so that recent files are first
        files_with_row_range.reverse()
        return files_with_row_range

    def compute_desired_num_rows(self):
        self.desired_num_rows = max(self.num_desired_rows(), self.min_rows)
        if self.max_rows is not None:
            self.desired_num_rows = min(self.desired_num_rows, self.max_rows)


def main(args):
    if args.min_rows is None:
        print(
            "NOTE: --min-rows was not specified, "
            "defaulting to requiring 250K rows before shuffling."
        )
        args.min_rows = 250000
    if args.keep_target_rows is None:
        print(
            "NOTE: --keep-target-rows was not specified, "
            "defaulting to keeping the first 1.2M rows."
        )
        print(
            "(slightly larger than default training epoch size of 1M, "
            "to give 1 epoch of data regardless of discreteness rows or batches "
            "per output file)"
        )
        print(
            "If you intended to shuffle the whole dataset instead, "
            "pass in --keep-target-rows <very large number>"
        )
        args.keep_target_rows = 1200000

    summary_data = {}
    if args.summary_file is not None:
        with TimeStuff(f"Loading {args.summary_file}"):
            summary_data = get_summary_data(args.summary_file)

    # Lists of npz files
    # all_files: all npz files, with known and unknown num_rows
    # files_with_unknown_num_rows: self explanatory
    with TimeStuff("Finding files"):
        all_npz_files, files_with_unknown_num_rows = get_file_info(
            args.dirs, summary_data
        )

    # sort by modification time
    with TimeStuff("Sorting .npz file list"):
        all_npz_files.sort(key=(lambda x: x[1]), reverse=False)

    # compute number of rows in unsummarized files
    with TimeStuff("Computing rows for unsummarized files"):
        all_npz_files = compute_all_num_rows(
            all_npz_files, files_with_unknown_num_rows, args.num_processes
        )

    window = TrainingWindow(args)
    # (filename, (start_row, end_row))
    files_with_row_range = window.get_files_to_use(all_npz_files)

    if window.num_total_rows <= 0:
        sys.exit("No rows found")
    elif window.num_total_rows < window.min_rows:
        sys.exit(
            f"Not enough rows, only {window.num_total_rows} "
            f"(fewer than {window.min_rows})"
        )
    else:
        print(
            f"Total rows found: {window.num_total_rows} "
            f"({window.num_usable_rows()} usable)"
        )

    window.compute_desired_num_rows()
    print(f"Desired num rows: {window.desired_num_rows} / {window.num_total_rows}")

    # Now assemble only the files we need to hit our desired window size
    with TimeStuff("Computing desired files from the desired window size"):
        (
            num_rows_total,
            desired_input_files,
            desired_input_files_with_row_range,
        ) = compute_desired_files(files_with_row_range, window.desired_num_rows)

    # shuffle input files
    np.random.seed()
    np.random.shuffle(desired_input_files)

    approx_rows_to_keep = min(num_rows_total, args.keep_target_rows)
    keep_prob = approx_rows_to_keep / num_rows_total

    shard_size = int(round(args.approx_shard_size / args.batch_size)) * args.batch_size
    num_shards = max(int(round(approx_rows_to_keep / shard_size)), 1)
    print(f"Shard size: {shard_size}", flush=True)

    # Group input files, where each group contains approximately worker_group_size rows
    desired_input_file_groups = group_input_files(
        desired_input_files, args.worker_group_size
    )
    print(
        f"Grouping {len(desired_input_files)} .npz data files into "
        f"{len(desired_input_file_groups)} sharding groups",
        flush=True,
    )

    num_worker_groups = max(int(round(approx_rows_to_keep / args.worker_group_size)), 1)
    out_tmp_dirs = [
        Path(args.out_tmp_dir) / f"tmp.shuf{i}" for i in range(num_worker_groups)
    ]

    def clean_tmp_dirs():
        for tmp_dir in out_tmp_dirs:
            if tmp_dir.exists():
                print(f"Cleaning up tmp dir: {tmp_dir}")
                shutil.rmtree(tmp_dir)

    clean_tmp_dirs()
    for tmp_dir in out_tmp_dirs:
        tmp_dir.mkdir(parents=True)

    # Merge .npz files in a group, shuffle, and split
    with multiprocessing.Pool(args.num_processes) as pool:
        with TimeStuff("Exporting data from groups randomly"):
            distribution_results = pool.starmap(
                distribute_shard_data,
                [
                    (
                        group_idx,
                        desired_input_file_groups[group_idx],
                        num_worker_groups,
                        out_tmp_dirs,
                        keep_prob,
                    )
                    for group_idx in range(len(desired_input_file_groups))
                ],
            )

    num_rows_in_groups = np.sum(distribution_results, axis=0)
    num_shards_in_groups = num_rows_in_groups // shard_size
    shard_start_index_in_groups = np.cumsum(num_shards_in_groups)
    shard_start_index_in_groups = np.insert(shard_start_index_in_groups, 0, 0)

    Path(args.out_dir).mkdir(exist_ok=True)
    print(f"Writing {num_shards_in_groups.sum()} shards")

    # Merge .npz files in each input file group into .tar shards
    with multiprocessing.Pool(args.num_processes) as pool:
        with TimeStuff("Shardifying distributed data"):
            merge_results = pool.starmap(
                shardify_group,
                [
                    (
                        args.out_dir,
                        shard_start_index_in_groups[group_idx],
                        num_shards_in_groups[group_idx],
                        out_tmp_dirs[group_idx],
                        shard_size,
                        args.compression,
                    )
                    for group_idx in range(num_worker_groups)
                ],
            )
        sys.stdout.flush()

    clean_tmp_dirs()

    dump_value = {
        "range": (
            min(desired_input_files_with_row_range, key=(lambda x: x[1][0]))[1][0],
            max(desired_input_files_with_row_range, key=(lambda x: x[1][1]))[1][1],
        )
    }

    with open(f"{args.out_dir}.json", "w") as f:
        json.dump(dump_value, f)


if __name__ == "__main__":
    description = """
    Convert .npz training data to .bin format, then tar, shardify, and shuffle them.
    """
    parser = ArgumentParser(description=description)
    parser.add_argument(
        "-dirs", metavar="DIR", nargs="+", help="Directories of training data files",
    )
    parser.add_argument("-out-dir", required=True, help="Dir to output training files")
    parser.add_argument(
        "-out-tmp-dir", required=True, help="Dir to use as scratch space"
    )
    parser.add_argument(
        "-summary-file", help="Summary json file for directory contents",
    )
    parser.add_argument(
        "-num-processes",
        type=int,
        required=True,
        help="Number of multiprocessing processes",
    )
    parser.add_argument(
        "-batch-size",
        type=int,
        required=True,
        help="Batch size to write training examples in",
    )
    parser.add_argument(
        "-compression",
        default="gz",
        choices=["None", "xz", "gz", "bz2"],
        help="Tar file compression format",
    )
    # start: training window size related parameters
    parser.add_argument(
        "-min-rows", type=int, help="Minimum training rows to use, default 250k",
    )
    parser.add_argument(
        "-max-rows", type=int, help="Maximum training rows to use, default unbounded",
    )
    parser.add_argument(
        "-keep-target-rows",
        type=int,
        help="Target number of rows to actually keep in the final data set, default 1.2M",
    )
    parser.add_argument(
        "-expand-window-per-row",
        type=float,
        required=True,
        help="Beyond min rows, initially expand the window by this much every post-random data row",
    )
    parser.add_argument(
        "-taper-window-exponent",
        type=float,
        required=True,
        help="Make the window size asymtotically grow as this power of the data rows",
    )
    parser.add_argument(
        "-taper-window-scale",
        type=float,
        help="The scale at which the power law applies",
    )
    parser.add_argument(
        "-add-to-window",
        type=float,
        default=0.0,
        help="Compute as if the window size were this much larger/smaller",
    )
    # end: training window size related parameters
    # start: sharding related
    parser.add_argument(
        "-worker-group-size",
        type=int,
        default=80000,
        help="Internally, target having many rows per parallel sharding worker",
    )
    parser.add_argument(
        "-approx-shard-size", type=int, required=True, help="Number of rows per shard",
    )
    # end: sharding related

    args = parser.parse_args()

    main(args)
