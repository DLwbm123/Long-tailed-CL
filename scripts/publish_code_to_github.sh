#!/usr/bin/env bash
set -euo pipefail

REMOTE_URL="${1:-https://github.com/DLwbm123/Long-tailed-CL.git}"
BRANCH="${BRANCH:-main}"
COMMIT_MESSAGE="${COMMIT_MESSAGE:-Source-only sync}"

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TMP_DIR="$(mktemp -d "${TMPDIR:-/tmp}/long-tailed-cl-publish.XXXXXX")"

cleanup() {
  rm -rf "$TMP_DIR"
}
trap cleanup EXIT

SNAPSHOT_DIR="$TMP_DIR/snapshot"
REPO_DIR="$TMP_DIR/repo"

mkdir -p "$SNAPSHOT_DIR" "$REPO_DIR"

rsync -a --delete \
  --exclude-from="$PROJECT_ROOT/.gitignore" \
  --exclude='/.git/' \
  --exclude='**/.git/' \
  "$PROJECT_ROOT/" "$SNAPSHOT_DIR/"

rm -rf "$REPO_DIR"
if git -c http.version=HTTP/1.1 clone "$REMOTE_URL" "$REPO_DIR"; then
  if git -C "$REPO_DIR" show-ref --verify --quiet "refs/remotes/origin/$BRANCH"; then
    git -C "$REPO_DIR" checkout -B "$BRANCH" "origin/$BRANCH"
  else
    git -C "$REPO_DIR" checkout -B "$BRANCH"
  fi
else
  mkdir -p "$REPO_DIR"
  git -C "$REPO_DIR" init
  git -C "$REPO_DIR" checkout -B "$BRANCH"
  git -C "$REPO_DIR" remote add origin "$REMOTE_URL"
fi

git -C "$REPO_DIR" config user.name "${GIT_AUTHOR_NAME:-Codex}"
git -C "$REPO_DIR" config user.email "${GIT_AUTHOR_EMAIL:-codex@local}"

rsync -a --delete --exclude='/.git/' "$SNAPSHOT_DIR/" "$REPO_DIR/"

git -C "$REPO_DIR" add -A
git -C "$REPO_DIR" status --short
if git -C "$REPO_DIR" diff --cached --quiet; then
  echo "No source changes to publish."
  exit 0
fi

git -C "$REPO_DIR" commit -m "$COMMIT_MESSAGE"
git -C "$REPO_DIR" -c http.version=HTTP/1.1 push -u origin "$BRANCH"

echo "Published source-only snapshot to $REMOTE_URL on $BRANCH"
