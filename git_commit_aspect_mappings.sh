#!/bin/bash
# Git commit script for aspect mappings update
# Run these commands in Git Bash

echo "🚀 Starting commit for aspect mappings update..."

# Commit: Update aspect mappings for all hotels
echo "📝 Commit: Update aspect mappings for all hotels"
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

echo ""
echo "✅ Commit completed!"
echo ""
echo "📊 Summary:"
git log --oneline -1
echo ""
echo "💡 To push to remote: git push origin main"

