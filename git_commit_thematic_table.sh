#!/bin/bash
# Git commit script for thematic table improvements
# Run these commands in Git Bash

echo "🚀 Starting commits for thematic table improvements..."

# Commit 1: Improve create_thematic_table function
echo "📝 Commit 1: Improve thematic table creation function"
git add src/domain/aspect_normalization.py
git commit -m "feat: improve create_thematic_table with sentiment filtering and better year handling

- Add sentiment_col and sentiment_filter parameters
- Improve year extraction from year_month or publishedAtDate
- Better handling of datetime vs string year columns
- Add total row to pivot table output
- Improve documentation and examples"

# Commit 2: Add thematic table creation documentation
echo "📝 Commit 2: Add thematic table creation documentation"
git add docs/THEMATIC_TABLE_CREATION.md
git commit -m "docs: add comprehensive guide for creating thematic tables

- Explain difference between manual dictionary and automated approach
- Show how create_thematic_table explodes combinations
- Provide usage examples with sentiment filtering
- Include SQL query alternative
- Add theme name mapping examples"

# Commit 3: Add example script for thematic table creation
echo "📝 Commit 3: Add example script for thematic table creation"
git add examples/create_thematic_table_correct.py
git commit -m "feat: add example script for creating thematic tables

- Demonstrate correct usage of create_thematic_table()
- Show how to filter by sentiment (positive/negative)
- Include Spanish theme name mapping
- Add comparison with manual dictionary counts
- Export to Excel functionality"

# Commit 4: Add example in domain folder (if needed)
if [ -f "src/domain/create_thematic_table_example.py" ]; then
    echo "📝 Commit 4: Add domain example script"
    git add src/domain/create_thematic_table_example.py
    git commit -m "docs: add example script in domain folder for thematic table creation"
fi

echo ""
echo "✅ All commits completed!"
echo ""
echo "📊 Summary:"
git log --oneline -4
echo ""
echo "💡 To push to remote: git push origin main"

