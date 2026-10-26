# Worktree Development Folder

This directory (`./tree/`) is used for **git worktree-based development** in this repository.

## Purpose

- Each feature/fix branch gets its own worktree under `./tree/<branch-name>/`
- Keeps the main worktree clean; enables parallel development without stashing
- Folder is **tracked by git** (exists in repo) but **contents are ignored** via `.gitignore`

## Workflow

```bash
# Create worktree for a new branch
git worktree add ./tree/<branch-name> -b <branch-name> main

# Work in the worktree
cd ./tree/<branch-name>
# ... make changes, run tests ...

# Push and create PR from worktree
gh pr create --title "..." --body-file ... --base main --head <branch-name>

# After PR merged and branch deleted
git worktree remove ./tree/<branch-name>
```

## Branch Naming

- `fix/...` — bug fixes
- `feat/...` — new features
- `refactor/...` — code improvements
- `chore/...` — maintenance

## Notes

- Never commit directly to `main`; all changes go through PRs
- CI runs on PRs; merge only after checks pass and explicit approval
- `.gitignore` excludes `./tree/*/` contents but keeps the folder itself