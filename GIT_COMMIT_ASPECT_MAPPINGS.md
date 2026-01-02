# Git Commit Commands for Aspect Mappings Update

Run this command in Git Bash:

## Commit: Update aspect mappings for all hotels

```bash
git add src/domain/aspect_mappings.py
git commit -m "feat: expand aspect mappings to cover all hotels' thematic categories

- Add mappings for Cosmos Pacífico, Steven Buenaventura, Torre Mar, and Magüipí
- Include food quality, staff consideration, organization, infrastructure details
- Add technology devices (phones, cards, intercoms, electronic billing)
- Map location details (beaches, port, natural environment, culture)
- Add maintenance aspects (plumbing, pests, painting, walls)
- Include guest experience categories (perception, profile, accessibility)
- Cover all manual thematic categories from analysis tables
- Ensure no aspects fall into generic categories
- Maintain consistency across all hotels for year-by-year comparisons"
```

## Verify Commit

After committing, verify with:

```bash
# View last commit
git log --oneline -1

# View detailed commit
git show HEAD

# View changes
git diff HEAD~1 src/domain/aspect_mappings.py
```

## Push to Remote

After committing:

```bash
git push origin main
```

## Quick Script Alternative

If you prefer to run the script:

```bash
# Make script executable
chmod +x git_commit_aspect_mappings.sh

# Run the script
./git_commit_aspect_mappings.sh
```

