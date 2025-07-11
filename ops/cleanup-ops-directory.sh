#!/bin/bash

# Clean up and organize the messy ops directory
# This directory has become a dumping ground - let's fix it

set -e

echo "🧹 Cleaning Up Ops Directory"
echo "============================="
echo ""

# Create organized subdirectories
echo "📁 Creating organized directory structure..."
mkdir -p {iam,deployment,monitoring,security,documentation,archive}

echo "🗂️  Moving files to appropriate directories..."

# Move IAM related files
echo "   Moving IAM files..."
mv aws-iam-policy.json iam/ 2>/dev/null || true
mv iam-*.json iam/ 2>/dev/null || true
mv iam-*.md iam/ 2>/dev/null || true
mv clarity-task-role-policy.json iam/ 2>/dev/null || true
mv ecs-execution-role-policy.json iam/ 2>/dev/null || true
mv ecs-task-s3-policy.json iam/ 2>/dev/null || true
mv task-role-policy.json iam/ 2>/dev/null || true
mv github-actions-*.json iam/ 2>/dev/null || true

# Move deployment related files
echo "   Moving deployment files..."
mv deploy*.sh deployment/ 2>/dev/null || true
mv ecs-task-definition.json deployment/ 2>/dev/null || true
mv check-deployment.sh deployment/ 2>/dev/null || true
mv test-deploy-script.sh deployment/ 2>/dev/null || true
mv validate-config.sh deployment/ 2>/dev/null || true

# Move security related files  
echo "   Moving security files..."
mv aws-waf-*.json security/ 2>/dev/null || true
mv waf-*.json security/ 2>/dev/null || true
mv s3-*.json security/ 2>/dev/null || true
mv test-waf-*.sh security/ 2>/dev/null || true
mv verify-s3-security.sh security/ 2>/dev/null || true

# Move monitoring related files
echo "   Moving monitoring files..."
mv test-*.sh monitoring/ 2>/dev/null || true
mv analyze-*.sh monitoring/ 2>/dev/null || true

# Move documentation
echo "   Moving documentation..."
mv *.md documentation/ 2>/dev/null || true

# Move old/unused files to archive
echo "   Archiving old files..."
mv env.sh archive/ 2>/dev/null || true
mv aws.sh archive/ 2>/dev/null || true
mv efs-models-setup.json archive/ 2>/dev/null || true
mv add-cofounder-aws-access.sh archive/ 2>/dev/null || true
mv create-presigned-urls-for-matt.sh archive/ 2>/dev/null || true
mv matt-cofounder-credentials.md archive/ 2>/dev/null || true

# Keep essential files in root
echo "   Keeping essential files in root..."
# fix-oidc-final-2025.sh stays in root
# README.md will be recreated

echo ""
echo "✅ Directory cleanup complete!"
echo ""
echo "📋 New structure:"
echo "   ops/"
echo "   ├── iam/           - IAM policies and roles"
echo "   ├── deployment/    - Deployment scripts and configs"
echo "   ├── security/      - Security configs (WAF, S3, etc)"
echo "   ├── monitoring/    - Testing and monitoring scripts"
echo "   ├── documentation/ - All documentation"
echo "   ├── archive/       - Old/unused files"
echo "   └── runbooks/      - Operational runbooks"
echo ""
echo "🗑️  Files that should be deleted (duplicates/obsolete):"
find . -name "*-old" -o -name "*-backup" -o -name "*-tmp" 2>/dev/null || true
echo ""
echo "💡 Next: Update all scripts to reference new file locations" 