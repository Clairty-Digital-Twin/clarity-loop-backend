#!/bin/bash
# 🔍 CHECK SWARM STATUS

echo "🔍 SWARM STATUS CHECK"
echo "===================="
echo ""

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Check each worktree
check_worktree() {
    local path=$1
    local name=$2
    if [ -d "$path" ]; then
        echo -e "${GREEN}✅ $name${NC}"
        echo "   Path: $path"
        echo "   Branch: $(cd $path && git branch --show-current)"
        if [ -f "$path/AGENTS.md" ]; then
            echo -e "   ${BLUE}AGENTS.md present${NC}"
        else
            echo -e "   ${YELLOW}⚠️  Missing AGENTS.md${NC}"
        fi
    else
        echo -e "${YELLOW}❌ $name - NOT FOUND${NC}"
    fi
    echo ""
}

echo "📂 Worktree Status:"
echo "-------------------"
check_worktree "pr-evaluator" "PR-EVALUATOR"
check_worktree "merge-strategist" "MERGE-STRATEGIST"
check_worktree "performance-optimizer" "PERFORMANCE-OPTIMIZER"
check_worktree "deployment-guardian" "DEPLOYMENT-GUARDIAN"
check_worktree "singularity-architect" "SINGULARITY-ARCHITECT"

echo ""
echo "📊 Latest Main Branch Status:"
echo "-----------------------------"
cd .. && git log --oneline origin/main -3
echo ""

echo "🚀 Remaining Branches to Process:"
echo "---------------------------------"
echo "- claude/issue-5-* (Dev Environment)"
echo "- claude/issue-7-* (Observability)"
echo "- claude/issue-8-* (Deployment v3)"
echo ""

echo "✨ Ready to launch agents in each worktree!"