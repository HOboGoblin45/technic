# Nuclear Option: Remove File from Git History

## The Problem

The file `data/training_data.parquet` is **still in Git history** even though you removed it from the current commit. Render is pulling the entire history and hitting the Git LFS error.

---

## THE NUCLEAR FIX

### Option 1: Use BFG Repo-Cleaner (Recommended)

```bash
# Download BFG
# Go to: https://rtyley.github.io/bfg-repo-cleaner/
# Or use: brew install bfg (Mac) or choco install bfg (Windows)

# Clone a fresh copy
cd ..
git clone --mirror https://github.com/HOboGoblin45/technic.git technic-mirror
cd technic-mirror

# Remove the file from ALL history
bfg --delete-files training_data.parquet

# Clean up
git reflog expire --expire=now --all
git gc --prune=now --aggressive

# Force push
git push --force
```

### Option 2: Use git filter-branch (Built-in)

```bash
# In your technic-clean directory
git filter-branch --force --index-filter \
  "git rm --cached --ignore-unmatch data/training_data.parquet" \
  --prune-empty --tag-name-filter cat -- --all

# Force push
git push origin --force --all
```

### Option 3: Simplest - Just Disable Git LFS in Render

Add this to your repo root:

**.gitattributes**
```
# Disable Git LFS
*.parquet !filter !diff !merge
```

Then:
```bash
git add .gitattributes
git commit -m "Disable Git LFS"
git push origin main
```

---

## RECOMMENDED: Option 3 (Simplest)

Just disable Git LFS entirely:

```bash
# Create/edit .gitattributes
echo "*.parquet !filter !diff !merge" > .gitattributes

# Commit
git add .gitattributes
git commit -m "Disable Git LFS for parquet files"
git push origin main
```

Then in Render: Clear cache & deploy again.

---

## ALTERNATIVE: Tell Render to Skip LFS

In Render, add this environment variable:

**Key:** `GIT_LFS_SKIP_SMUDGE`  
**Value:** `1`

This tells Git to skip LFS downloads entirely.

### Steps:
1. Render Dashboard → Your Service
2. Environment → Add Environment Variable
3. Key: `GIT_LFS_SKIP_SMUDGE`
4. Value: `1`
5. Save
6. Manual Deploy

---

## EASIEST FIX RIGHT NOW

Run these commands:

```bash
# Disable Git LFS
echo "" > .gitattributes
echo "# Disable Git LFS" >> .gitattributes  
echo "*.parquet !filter !diff !merge" >> .gitattributes

# Commit
git add .gitattributes
git commit -m "Disable Git LFS completely"
git push origin main
```

Then: Render → Manual Deploy (don't need to clear cache this time)

---

## WHY THIS KEEPS HAPPENING

Git LFS is configured in your repo's history. Even though you removed the file, Git still tries to download it from LFS when cloning. You need to either:

1. Remove from ALL history (nuclear)
2. Disable LFS (simple)
3. Tell Render to skip LFS (environment variable)

**Recommended: Just disable LFS with .gitattributes**
