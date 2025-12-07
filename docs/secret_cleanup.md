# Secret Cleanup Guide

## Warning
These steps rewrite git history. After running, all collaborators must re-clone the repository. Force-pushing will overwrite remote history.

## Commands
```bash
# 1) Install git-filter-repo (one-time)
python -m pip install git-filter-repo

# 2) From repo root, scrub secrets using replacements.txt
git filter-repo --replace-text replacements.txt --force

# 3) Force-push cleaned history
git push origin --force
```

## After Cleanup
- Revoke old API keys (OpenAI, Polygon, FMP, Technic API) in their dashboards.
- Generate new keys and set them via environment variables or .env (see .env.example).
- Ask all users to re-clone the repo because history has been rewritten.
