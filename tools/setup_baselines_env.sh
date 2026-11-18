# A bash script, started with the bash command (no interpreter line: no location is written here).
# Copy the two tool environments of the baselines out of an earlier install folder
# into the work folder, and check each copy as it will be used.
#
#   bash tools/setup_baselines_env.sh EARLIER WORK PYTHON [PROGRAM]
#
#   EARLIER  the earlier install folder; it holds venv-dolma and venv-ob. It is only
#            read: nothing is written there.
#   WORK     the work folder the copies go into (WORK/venv-dolma, WORK/venv-ob).
#   PYTHON   the interpreter that runs the checker, baselines.envcheck.
#   PROGRAM  optional: the program of the OriginBlame tool (a file). It is passed to the
#            checker for venv-ob only, which writes its sha256 into the identity.json of
#            that record as ob_binary_sha256; the file is only read, never started.
#
# Nothing is fetched and nothing is installed: this script only copies, compares and
# checks, and it starts nothing of either environment (the checker is run by PYTHON,
# which reads files). For each tool, in this order:
#   1. cp -a copies the environment, links kept as they are;
#   2. before anything else, every file, link and folder of the copy is compared with
#      the original's, including what the RECORD check leaves out (launcher scripts,
#      files in no RECORD, pyvenv.cfg, links); each difference is printed with its path
#      and its reason, and any difference stops THAT tool;
#   3. the sha256 of every package file is compared with the RECORD rows of the
#      environment's own dist-info folders (launcher scripts left out), and the
#      interpreter the copy binds to is read from files (the version from pyvenv.cfg
#      and from the interpreter folder's name, the sha256 of the binary);
#   4. the report goes to WORK/env-check-<name>.json and the record of what the copy
#      uses (its rows file, the files in no RECORD, the interpreter's identity, and
#      compare.json, the result of the compare of step 2, an equal result included) to
#      the new folder WORK/env-record-<name>. A copy whose compare found a difference
#      gets a record folder that holds compare.json alone (equal false, each difference
#      with its path and its reason); its differences are on standard error too.
#   The check of step 3 also refuses a pyvenv.cfg whose version_info the version of the
#   interpreter folder does not begin with (3.12 against 3.12.13 passes, 3.11 does not).
# A difference, a mismatch, a missing environment or a copy that is not as it should
# be stops THAT tool: the message names it, the exit status is nonzero, and nothing is
# repaired (the copy stays as it is). Both environments are always tried. A copy is
# never overwritten: a destination that exists stops its tool.

if [ "$#" -ne 3 ] && [ "$#" -ne 4 ]; then
    echo "usage: setup_baselines_env.sh EARLIER WORK PYTHON [PROGRAM]" >&2
    exit 2
fi
earlier=$1
work=$2
python=$3
program=${4:-}

here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
repo=$(dirname "$here")

status=0
for name in venv-dolma venv-ob; do
    source_dir="$earlier/$name"
    copy_dir="$work/$name"
    report="$work/env-check-$name.json"
    record="$work/env-record-$name"
    if [ ! -d "$source_dir" ]; then
        echo "$name: not found in the earlier install folder" >&2
        status=1
        continue
    fi
    if [ -e "$copy_dir" ] || [ -e "$report" ] || [ -e "$record" ]; then
        echo "$name: the copy, its report or its record exists in the work folder; nothing is overwritten" >&2
        status=1
        continue
    fi
    if ! cp -a -- "$source_dir" "$copy_dir"; then
        echo "$name: the copy failed" >&2
        status=1
        continue
    fi
    extra=()
    if [ "$name" = "venv-ob" ] && [ -n "$program" ]; then
        extra=(--program "$program")
    fi
    if PYTHONPATH="$repo" PYTHONDONTWRITEBYTECODE=1 "$python" -m baselines.envcheck "$copy_dir" --original "$source_dir" --out "$report" --record "$record" "${extra[@]}"; then
        echo "$name: compared with its original, checked and recorded"
    else
        echo "$name: stopped, its compare or its check failed" >&2
        status=1
    fi
done
exit "$status"
