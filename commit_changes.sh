#!/bin/bash
# Git commit script for individual changes
# Run this script to commit changes in logical groups

echo "🚀 Starting individual commits..."

# 1. Language Translation - Configuration & Infrastructure
echo "📝 Commit 1: Language translation - Config & Infrastructure"
git add src/config.py src/db.py src/utils.py src/bronze_build.py src/list_blob.py
git commit -m "refactor: translate all Spanish comments and messages to English

- Translate comments in config.py, db.py, utils.py
- Translate error messages and log outputs
- Update bronze_build.py and list_blob.py messages
- Maintain all original functionality"

# 2. Language Translation - Data Processing
echo "📝 Commit 2: Language translation - Data Processing"
git add src/silver_build.py src/gold_build.py
git commit -m "refactor: translate data processing modules to English

- Translate all comments and docstrings in silver_build.py
- Translate all comments and docstrings in gold_build.py
- Update print statements and error messages
- Preserve all business logic"

# 3. Architecture - Infrastructure Layer
echo "📝 Commit 3: Architecture - Infrastructure Layer"
git add src/infrastructure/
git commit -m "refactor: create infrastructure layer for external services

- Extract configuration to infrastructure/config.py
- Extract database operations to infrastructure/database.py
- Extract blob storage to infrastructure/blob_storage.py
- Maintain backward compatibility through re-exports
- Improve separation of concerns"

# 4. Architecture - Domain Layer (Transformations)
echo "📝 Commit 4: Architecture - Domain Layer"
git add src/domain/__init__.py src/domain/transformations.py
git commit -m "refactor: create domain layer for business logic

- Move data transformations to domain/transformations.py
- Separate business logic from infrastructure
- Improve testability and maintainability
- Follow clean architecture principles"

# 5. ACID Compliance - Database Operations
echo "📝 Commit 5: ACID Compliance"
git add src/nashor_to_supabase.py
git commit -m "feat: improve ACID compliance in database operations

- Add explicit transaction management with rollback
- Implement batch processing with transaction boundaries
- Improve error handling and retry logic
- Add validation and logging
- Ensure data consistency and atomicity"

# 6. Aspect Normalization - Thematic System
echo "📝 Commit 6: Aspect Normalization System"
git add src/domain/aspect_mappings.py src/domain/aspect_normalization.py
git add docs/THEMATIC_ASPECT_NORMALIZATION.md docs/THEMATIC_NORMALIZATION_REDESIGN.md
git commit -m "feat: implement thematic aspect normalization system

- Create two-level model: aspect_raw → aspect_theme
- Define 15 meaningful thematic categories (no generic categories)
- Implement multi-step matching strategy
- Add single source of truth in aspect_mappings.py
- Support scalability and future-proofing"

# 7. Aspect Normalization - Integration
echo "📝 Commit 7: Aspect Normalization Integration"
git add src/domain/transformations.py src/nashor_to_supabase.py
git commit -m "feat: integrate thematic aspect normalization into pipeline

- Add aspect normalization to enrich_with_azure()
- Create aspect_raw and aspect_theme columns
- Add validation and statistics
- Update DATA_COLS to include new columns
- Automatic normalization during Gold layer processing"

# 8. Aspect Normalization - Fix for Reviews Without Text
echo "📝 Commit 8: Aspect Normalization Fix"
git add src/domain/aspect_normalization.py src/domain/transformations.py
git add docs/ASPECT_NORMALIZATION_FIX.md
git commit -m "fix: handle reviews without text in aspect normalization

- Set aspect_theme = NULL for reviews without text
- Only normalize aspects for reviews processed by Azure
- Fix aspect counts (733 vs 1193 for cordillera)
- Ensure SQL queries filter correctly"

# 9. Documentation - Pipeline Guide
echo "📝 Commit 9: Documentation"
git add docs/PIPELINE_EXECUTION_GUIDE.md docs/GIT_COMMIT_GUIDE.md
git commit -m "docs: add comprehensive pipeline execution and git commit guides

- Step-by-step instructions for Bronze → Silver → Gold → Database
- Troubleshooting section
- Git commit guide for individual changes
- Command options and examples"

# 10. Backward Compatibility Updates
echo "📝 Commit 10: Backward Compatibility"
git add src/config.py src/db.py src/utils.py
git commit -m "refactor: maintain backward compatibility through re-exports

- Update legacy files to import from new layers
- Preserve existing import paths
- No breaking changes for existing code
- Smooth migration path"

echo ""
echo "✅ All commits completed!"
echo ""
echo "📊 Summary:"
git log --oneline -10
echo ""
echo "💡 To push to remote: git push origin <branch-name>"

