#!/usr/bin/env bash
# merge-pr.sh - Merge PR with squash and delete both local & remote branches
# Usage: ./scripts/merge-pr.sh <pr-number>

set -euo pipefail

PR_NUMBER="${1:-}"
if [[ -z "$PR_NUMBER" ]]; then
    echo "Usage: $0 <pr-number>"
    exit 1
fi

# Get branch name before merging
BRANCH=$(gh pr view "$PR_NUMBER" --json headRefName -q .headRefName 2>/dev/null || echo "")

# Merge with squash
gh pr merge "$PR_NUMBER" --squash --delete-branch

# Delete remote branch if it still exists
if [[ -n "$BRANCH" ]]; then
    git push origin --delete "$BRANCH" 2>/dev/null || true
fi

# Clean up worktree if it exists
WORKTREE="./tree/$BRANCH"
if [[ -d "$WORKTREE" ]]; then
    git worktree remove "$WORKTREE" 2>/dev/null || true
fi