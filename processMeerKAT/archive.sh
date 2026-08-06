#!/bin/bash
#SBATCH --job-name=reduction_archive
#SBATCH --output=reduction_archive_%j.out
#SBATCH --error=reduction_archive_%j.err
#SBATCH --time=00:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4GB
#
# archive.sh  --  processMeerKAT_llus reduction archiver
#
# Run this INSIDE a single galaxy/target reduction directory once you are
# completely finished with it. It sweeps the products worth keeping into one
# archive folder, then (only when you confirm) permanently deletes everything
# else in that directory to reclaim space.
#
# It is shipped with the pipeline and put on $PATH by setup.sh, so from any
# reduction directory you can simply run:  archive.sh   (or: archive.sh --help)
#
# What is KEPT (archived), matched at the TOP LEVEL of the current directory:
#   - *science_image*   all science-imaging products. Matching by name (not by
#                       target) automatically covers the plain images, the
#                       .alpha / .alpha.error maps (stokes != 'I'), the
#                       .cube.* products and SPW_MFSs/ (spw_cube = True), and the
#                       .katbeam* products (pbcorr = True).
#   - SPW_MFSs/         per-SPW image folder (spw_cube = True).
#   - plots/            diagnostic plots.
#   - the target MMS    the galaxy's own .mms/.ms, identified from the config
#                       [fields] targetfields (calibrator MMSs are NOT kept).
#   - *.png             stray plot images.
#
# What is DELETED (everything else at the top level), e.g.:
#   - per-SPW *MHz* directories, calibrator .mms, .gcal*/.bcal* tables,
#   - *.contcube / *.contcube.fits quick-look cross-cal diagnostics, logs, etc.
#
# Nothing is deleted unless you pass -d/--confirm-delete (or set
# CONFIRM_DELETE=true below); by default you only get a preview.
# The config file used is kept in place -- it is never archived or deleted.

set -uo pipefail

#############################  DEFAULTS  #############################
CONFIRM_DELETE=false           # false = preview only (safe). true = actually delete.
ARCHIVE_DIR="RC_PMKT_archive"  # folder that kept products are moved into.
CONFIG=""                      # config to read [fields] targetfields from (auto-detected).
TARGET=""                      # comma-separated target name(s); overrides the config.
#####################################################################

usage() {
cat <<EOF
Usage: archive.sh [options]   (run inside one galaxy/target reduction directory)

Archives the processMeerKAT_llus products worth keeping into ./${ARCHIVE_DIR}/,
then (only if enabled) permanently deletes everything else in this directory.

Kept (top level only): *science_image* (images, alpha, cube, katbeam),
SPW_MFSs/, plots/, *.png, and the target's own .mms/.ms.
Deleted: per-SPW *MHz* dirs, calibrator MMSs, .gcal*, *.contcube[.fits], logs, etc.

Options:
  -t, --target NAME[,NAME2,...]
                    Target field name(s) whose MMS to keep (e.g. --target PGC31359).
                    If omitted, the target is automatically read from .config.tmp
  -C, --config FILE
                    processMeerKAT config to read 'targetfields' from ([fields]).
                    Auto-detected if omitted (.config.tmp, then first *.txt).
  -d, --confirm-delete
                    Actually perform the cleanup deletion (default: preview only).
  -a, --archive-dir DIR
                    Name of the archive folder (default: ${ARCHIVE_DIR}).
  -h, --help        Show this help and exit.

Notes:
  * Safe by default: nothing is deleted until you pass -d/--confirm-delete.
  * The config file is kept in place -- it is never archived or deleted.
  * If no target can be resolved, ALL top-level .mms/.ms are kept.

Examples:
  archive.sh --help
  cd pgc31359 && archive.sh                 # preview (auto-reads .config.tmp)
  cd pgc31359 && archive.sh --confirm-delete
  archive.sh --target PGC31359 -d
EOF
}

#############################  ARG PARSING  #########################
while [[ $# -gt 0 ]]; do
    case "$1" in
        -t|--target)         TARGET="${2:-}"; shift 2 ;;
        -C|--config)         CONFIG="${2:-}"; shift 2 ;;
        -a|--archive-dir)    ARCHIVE_DIR="${2:-}"; shift 2 ;;
        -d|--confirm-delete) CONFIRM_DELETE=true; shift ;;
        -h|--help)           usage; exit 0 ;;
        *) echo "Unknown option: $1" >&2; echo "Try --help." >&2; exit 2 ;;
    esac
done

#############################  RESOLVE TARGET(S)  ###################
# Prefer --target; otherwise read 'targetfields' from the config's [fields] section.
if [[ -z "$TARGET" ]]; then
    if [[ -z "$CONFIG" ]]; then
        if [[ -f ".config.tmp" ]]; then
            CONFIG=".config.tmp"
        else
            CONFIG=$(ls -1 ./*.txt 2>/dev/null | head -n 1 || true)
        fi
    fi
    if [[ -n "$CONFIG" && -f "$CONFIG" ]]; then
        raw=$(grep -E '^[[:space:]]*targetfields[[:space:]]*=' "$CONFIG" | head -n 1 | cut -d= -f2-)
        raw="${raw%%#*}"                                                  # strip inline comment
        raw="$(echo "$raw" | tr -d "[]'\"" | xargs 2>/dev/null || true)"  # strip quotes/brackets/spaces
        TARGET="$raw"
        [[ -n "$TARGET" ]] && echo "Target(s) '$TARGET' read from config '$CONFIG'."
    fi
fi

declare -a TARGETS=()
[[ -n "$TARGET" ]] && IFS=',' read -r -a TARGETS <<< "$TARGET"

#############################  BUILD KEEP EXPRESSION  ###############
# Category matches that hold regardless of which pipeline features were enabled.
declare -a MATCH=( '(' '(' -type d -name 'plots' ')' \
                       -o '(' -type d -name 'SPW_MFSs' ')' \
                       -o -name '*science_image*' \
                       -o -name '*.png' )

if [[ ${#TARGETS[@]} -gt 0 ]]; then
    for t in "${TARGETS[@]}"; do
        [[ -z "$t" ]] && continue
        # keep the target's own MMS/MS (and its .flagversions); exclude per-SPW *MHz* copies
        MATCH+=( -o '(' -name "*${t}*.mms*" '!' -name '*MHz*' ')' \
                 -o -name "*${t}*.ms" )
    done
else
    echo "No target resolved (no --target and none in config); conservatively keeping ALL top-level .mms/.ms."
    MATCH+=( -o '(' -name '*.mms*' '!' -name '*MHz*' ')' -o -name '*.ms' )
fi
MATCH+=( ')' )

#############################  ARCHIVE SECTION  ####################
mkdir -p "$ARCHIVE_DIR"

echo "The following top-level items will be archived into $ARCHIVE_DIR/:"
find . -mindepth 1 -maxdepth 1 ! -name "$ARCHIVE_DIR" "${MATCH[@]}" -print
echo ""

echo "Archiving now..."
find . -mindepth 1 -maxdepth 1 ! -name "$ARCHIVE_DIR" "${MATCH[@]}" -exec mv -t "$ARCHIVE_DIR/" {} +
echo "Archive step complete."
echo ""

#############################  CLEANUP SECTION  ####################
CLEANUP_EXCLUDES=( ! -name "$ARCHIVE_DIR" ! -name "$(basename "$0")" \
                   ! -name "reduction_archive_*.out" ! -name "reduction_archive_*.err" )
[[ -n "$CONFIG" ]] && CLEANUP_EXCLUDES+=( ! -name "$(basename "$CONFIG")" )

echo "The following remaining items are NOT in $ARCHIVE_DIR/ and would be permanently deleted:"
find . -mindepth 1 -maxdepth 1 "${CLEANUP_EXCLUDES[@]}" -print
echo ""

if [ "$CONFIRM_DELETE" = true ]; then
    echo "CONFIRM_DELETE is true - deleting the above now..."
    find . -mindepth 1 -maxdepth 1 "${CLEANUP_EXCLUDES[@]}" -exec rm -rf {} +
    echo "Cleanup complete."
else
    echo "CONFIRM_DELETE is false - preview only, nothing deleted. Use -d/--confirm-delete (or set CONFIRM_DELETE=true) to actually delete."
fi
