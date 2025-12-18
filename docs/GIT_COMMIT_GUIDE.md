# Git Commit Guide - Individual Commits

This guide provides git commands to commit changes individually in logical groups.

## Commit Strategy

We'll commit changes in this order:
1. Language translation (Spanish → English)
2. Architecture refactoring (layered structure)
3. ACID compliance improvements
4. Aspect normalization system (thematic categories)
5. Aspect normalization fix (NULL handling)

## Git Commands

### 1. Language Translation - Configuration & Infrastructure

```bash
# Stage configuration and infrastructure files
git add src/config.py
git add src/db.py
git add src/utils.py
git add src/bronze_build.py
git add src/list_blob.py

# Commit
git commit -m "refactor: translate all Spanish comments and messages to English

- Translate comments in config.py, db.py, utils.py
- Translate error messages and log outputs
- Update bronze_build.py and list_blob.py messages
- Maintain all original functionality"
```

### 2. Language Translation - Data Processing

```bash
# Stage data processing files
git add src/silver_build.py
git add src/gold_build.py

# Commit
git commit -m "refactor: translate data processing modules to English

- Translate all comments and docstrings in silver_build.py
- Translate all comments and docstrings in gold_build.py
- Update print statements and error messages
- Preserve all business logic"
```

### 3. Architecture - Infrastructure Layer

```bash
# Stage new infrastructure layer
git add src/infrastructure/
git add src/infrastructure/__init__.py
git add src/infrastructure/config.py
git add src/infrastructure/database.py
git add src/infrastructure/blob_storage.py

# Commit
git commit -m "refactor: create infrastructure layer for external services

- Extract configuration to infrastructure/config.py
- Extract database operations to infrastructure/database.py
- Extract blob storage to infrastructure/blob_storage.py
- Maintain backward compatibility through re-exports
- Improve separation of concerns"
```

### 4. Architecture - Domain Layer

```bash
# Stage domain layer
git add src/domain/
git add src/domain/__init__.py
git add src/domain/transformations.py

# Commit
git commit -m "refactor: create domain layer for business logic

- Move data transformations to domain/transformations.py
- Separate business logic from infrastructure
- Improve testability and maintainability
- Follow clean architecture principles"
```

### 5. ACID Compliance - Database Operations

```bash
# Stage database improvements
git add src/nashor_to_supabase.py

# Commit
git commit -m "feat: improve ACID compliance in database operations

- Add explicit transaction management with rollback
- Implement batch processing with transaction boundaries
- Improve error handling and retry logic
- Add validation and logging
- Ensure data consistency and atomicity"
```

### 6. Aspect Normalization - Thematic System

```bash
# Stage aspect normalization system
git add src/domain/aspect_mappings.py
git add src/domain/aspect_normalization.py
git add docs/THEMATIC_ASPECT_NORMALIZATION.md
git add docs/THEMATIC_NORMALIZATION_REDESIGN.md

# Commit
git commit -m "feat: implement thematic aspect normalization system

- Create two-level model: aspect_raw → aspect_theme
- Define 15 meaningful thematic categories (no generic categories)
- Implement multi-step matching strategy
- Add single source of truth in aspect_mappings.py
- Support scalability and future-proofing
- Add comprehensive documentation"
```

### 7. Aspect Normalization - Integration

```bash
# Stage integration changes
git add src/domain/transformations.py
git add src/nashor_to_supabase.py

# Commit
git commit -m "feat: integrate thematic aspect normalization into pipeline

- Add aspect normalization to enrich_with_azure()
- Create aspect_raw and aspect_theme columns
- Add validation and statistics
- Update DATA_COLS to include new columns
- Automatic normalization during Gold layer processing"
```

### 8. Aspect Normalization - Fix for Reviews Without Text

```bash
# Stage fix for NULL handling
git add src/domain/aspect_normalization.py
git add src/domain/transformations.py
git add docs/ASPECT_NORMALIZATION_FIX.md

# Commit
git commit -m "fix: handle reviews without text in aspect normalization

- Set aspect_theme = NULL for reviews without text
- Only normalize aspects for reviews processed by Azure
- Fix aspect counts (733 vs 1193 for cordillera)
- Ensure SQL queries filter correctly
- Add documentation for the fix"
```

### 9. Documentation - Pipeline Guide

```bash
# Stage pipeline documentation
git add docs/PIPELINE_EXECUTION_GUIDE.md

# Commit
git commit -m "docs: add comprehensive pipeline execution guide

- Step-by-step instructions for Bronze → Silver → Gold → Database
- Troubleshooting section
- Command options and examples
- Data flow diagrams
- Testing strategies"
```

### 10. Backward Compatibility Updates

```bash
# Stage backward compatibility files
git add src/config.py
git add src/db.py
git add src/utils.py

# Commit
git commit -m "refactor: maintain backward compatibility through re-exports

- Update legacy files to import from new layers
- Preserve existing import paths
- No breaking changes for existing code
- Smooth migration path"
```

## Alternative: Single Command Script

If you prefer to run all commits at once, create a script:

```bash
# Create commit script
cat > commit_changes.sh << 'EOF'
#!/bin/bash

# 1. Language Translation - Config & Infrastructure
git add src/config.py src/db.py src/utils.py src/bronze_build.py src/list_blob.py
git commit -m "refactor: translate all Spanish comments and messages to English"

# 2. Language Translation - Data Processing
git add src/silver_build.py src/gold_build.py
git commit -m "refactor: translate data processing modules to English"

# 3. Architecture - Infrastructure Layer
git add src/infrastructure/
git commit -m "refactor: create infrastructure layer for external services"

# 4. Architecture - Domain Layer
git add src/domain/__init__.py src/domain/transformations.py
git commit -m "refactor: create domain layer for business logic"

# 5. ACID Compliance
git add src/nashor_to_supabase.py
git commit -m "feat: improve ACID compliance in database operations"

# 6. Aspect Normalization System
git add src/domain/aspect_mappings.py src/domain/aspect_normalization.py
git add docs/THEMATIC_ASPECT_NORMALIZATION.md docs/THEMATIC_NORMALIZATION_REDESIGN.md
git commit -m "feat: implement thematic aspect normalization system"

# 7. Aspect Normalization Integration
git add src/domain/transformations.py src/nashor_to_supabase.py
git commit -m "feat: integrate thematic aspect normalization into pipeline"

# 8. Aspect Normalization Fix
git add src/domain/aspect_normalization.py src/domain/transformations.py
git add docs/ASPECT_NORMALIZATION_FIX.md
git commit -m "fix: handle reviews without text in aspect normalization"

# 9. Documentation
git add docs/PIPELINE_EXECUTION_GUIDE.md
git commit -m "docs: add comprehensive pipeline execution guide"

echo "✅ All commits completed!"
EOF

chmod +x commit_changes.sh
./commit_changes.sh
```

## Verification

After committing, verify with:

```bash
# View commit history
git log --oneline -10

# View changes in a commit
git show <commit-hash>

# View all changes
git log --stat
```

## Push to Remote

After all commits are done:

```bash
# Push all commits to remote
git push origin main

# Or push to specific branch
git push origin <branch-name>
```

