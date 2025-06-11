#!/bin/bash

# Script to remove sensitive files from git history
# WARNING: This will rewrite git history!

echo "⚠️  WARNING: This will rewrite git history!"
echo "Make sure all team members are aware before proceeding."
echo "Press Ctrl+C to cancel, or Enter to continue..."
read

# Files to remove from history
FILES_TO_REMOVE=(
    "secrets-policy.json"
    "ecs-task-final.json"
    "ecs-task-singularity.json"
    "ecs-task-v20250611-fix.json"
    "ecr-images-to-delete.json"
)

# Create backup branch
echo "Creating backup branch..."
git branch backup-before-cleanup

# Remove files from history
for file in "${FILES_TO_REMOVE[@]}"; do
    echo "Removing $file from history..."
    git filter-branch --force --index-filter \
        "git rm --cached --ignore-unmatch $file" \
        --prune-empty --tag-name-filter cat -- --all
done

echo "✅ Files removed from history"
echo ""
echo "Next steps:"
echo "1. Force push to remote: git push origin --force --all"
echo "2. Force push tags: git push origin --force --tags"
echo "3. Tell all team members to re-clone the repository"
echo "4. Consider using BFG Repo-Cleaner for larger cleanups"
echo ""
echo "Backup branch 'backup-before-cleanup' was created for safety"