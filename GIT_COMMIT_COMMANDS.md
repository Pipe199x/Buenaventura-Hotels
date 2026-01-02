# Git Commit Commands for Thematic Table Improvements

Run these commands **one at a time** in Git Bash:

## Commit 1: Improve create_thematic_table function

```bash
git add src/domain/aspect_normalization.py
git commit -m "feat: improve create_thematic_table with sentiment filtering and better year handling

- Add sentiment_col and sentiment_filter parameters
- Improve year extraction from year_month or publishedAtDate
- Better handling of datetime vs string year columns
- Add total row to pivot table output
- Improve documentation and examples"
```

## Commit 2: Add thematic table creation documentation

```bash
git add docs/THEMATIC_TABLE_CREATION.md
git commit -m "docs: add comprehensive guide for creating thematic tables

- Explain difference between manual dictionary and automated approach
- Show how create_thematic_table explodes combinations
- Provide usage examples with sentiment filtering
- Include SQL query alternative
- Add theme name mapping examples"
```

## Commit 3: Add example script for thematic table creation

```bash
git add examples/create_thematic_table_correct.py
git commit -m "feat: add example script for creating thematic tables

- Demonstrate correct usage of create_thematic_table()
- Show how to filter by sentiment (positive/negative)
- Include Spanish theme name mapping
- Add comparison with manual dictionary counts
- Export to Excel functionality"
```

## Commit 4: Add domain example script (if exists)

```bash
git add src/domain/create_thematic_table_example.py
git commit -m "docs: add example script in domain folder for thematic table creation"
```

## Verify Commits

After committing, verify with:

```bash
# View last commits
git log --oneline -4

# View detailed commit
git show HEAD

# View all changes
git status
```

## Push to Remote

After all commits are done:

```bash
git push origin main
```

## Quick Script Alternative

If you prefer to run all at once, use the script:

```bash
# Make script executable
chmod +x git_commit_thematic_table.sh

# Run the script
./git_commit_thematic_table.sh
```

