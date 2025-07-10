# Claude Code Configuration

## Build Commands

- `npm run build`: Build the project
- `npm run test`: Run the full test suite
- `npm run lint`: Run ESLint and format checks
- `npm run typecheck`: Run TypeScript type checking
- `./tools/claude-flow --help`: Show all available commands

## Claude-Flow Complete Command Reference

### Core System Commands

- `./tools/claude-flow start [--ui] [--port 3000] [--host localhost]`: Start orchestration system with optional web UI
- `./tools/claude-flow status`: Show comprehensive system status
- `./tools/claude-flow monitor`: Real-time system monitoring dashboard
- `./tools/claude-flow config <subcommand>`: Configuration management (show, get, set, init, validate)

### Agent Management

- `./tools/claude-flow agent spawn <type> [--name <name>]`: Create AI agents (researcher, coder, analyst, etc.)
- `./tools/claude-flow agent list`: List all active agents
- `./tools/claude-flow spawn <type>`: Quick agent spawning (alias for agent spawn)

### Task Orchestration

- `./tools/claude-flow task create <type> [description]`: Create and manage tasks
- `./tools/claude-flow task list`: View active task queue
- `./tools/claude-flow workflow <file>`: Execute workflow automation files

### Memory Management

- `./tools/claude-flow memory store <key> <data>`: Store persistent data across sessions
- `./tools/claude-flow memory get <key>`: Retrieve stored information
- `./tools/claude-flow memory list`: List all memory keys
- `./tools/claude-flow memory export <file>`: Export memory to file
- `./tools/claude-flow memory import <file>`: Import memory from file
- `./tools/claude-flow memory stats`: Memory usage statistics
- `./tools/claude-flow memory cleanup`: Clean unused memory entries

### SPARC Development Modes

- `./tools/claude-flow sparc "<task>"`: Run orchestrator mode (default)
- `./tools/claude-flow sparc run <mode> "<task>"`: Run specific SPARC mode
- `./tools/claude-flow sparc tdd "<feature>"`: Test-driven development mode
- `./tools/claude-flow sparc modes`: List all 17 available SPARC modes

Available SPARC modes: orchestrator, coder, researcher, tdd, architect, reviewer, debugger, tester, analyzer, optimizer, documenter, designer, innovator, swarm-coordinator, memory-manager, batch-executor, workflow-manager

### Swarm Coordination

- `./tools/claude-flow swarm "<objective>" [options]`: Multi-agent swarm coordination
- `--strategy`: research, development, analysis, testing, optimization, maintenance
- `--mode`: centralized, distributed, hierarchical, mesh, hybrid
- `--max-agents <n>`: Maximum number of agents (default: 5)
- `--parallel`: Enable parallel execution
- `--monitor`: Real-time monitoring
- `--output <format>`: JSON, sqlite, csv, html

### MCP Server Integration

- `./tools/claude-flow mcp start [--port 3000] [--host localhost]`: Start MCP server
- `./tools/claude-flow mcp status`: Show MCP server status
- `./tools/claude-flow mcp tools`: List available MCP tools

### Claude Integration

- `./tools/claude-flow claude auth`: Authenticate with Claude API
- `./tools/claude-flow claude models`: List available Claude models
- `./tools/claude-flow claude chat`: Interactive chat mode

### Session Management

- `./tools/claude-flow session`: Manage terminal sessions
- `./tools/claude-flow repl`: Start interactive REPL mode

### Enterprise Features

- `./tools/claude-flow project <subcommand>`: Project management (Enterprise)
- `./tools/claude-flow deploy <subcommand>`: Deployment operations (Enterprise)
- `./tools/claude-flow cloud <subcommand>`: Cloud infrastructure management (Enterprise)
- `./tools/claude-flow security <subcommand>`: Security and compliance tools (Enterprise)
- `./tools/claude-flow analytics <subcommand>`: Analytics and insights (Enterprise)

### Project Initialization

- `./tools/claude-flow init`: Initialize Claude-Flow project
- `./tools/claude-flow init --sparc`: Initialize with full SPARC development environment

## Quick Start Workflows

### Research Workflow

```bash
# Start a research swarm with distributed coordination
./tools/claude-flow swarm "Research modern web frameworks" --strategy research --mode distributed --parallel --monitor

# Or use SPARC researcher mode for focused research
./tools/claude-flow sparc run researcher "Analyze React vs Vue vs Angular performance characteristics"

# Store findings in memory for later use
./tools/claude-flow memory store "research_findings" "Key insights from framework analysis"
```

### Development Workflow

```bash
# Start orchestration system with web UI
./tools/claude-flow start --ui --port 3000

# Run TDD workflow for new feature
./tools/claude-flow sparc tdd "User authentication system with JWT tokens"

# Development swarm for complex projects
./tools/claude-flow swarm "Build e-commerce API with payment integration" --strategy development --mode hierarchical --max-agents 8 --monitor

# Check system status
./tools/claude-flow status
```

### Analysis Workflow

```bash
# Analyze codebase performance
./tools/claude-flow sparc run analyzer "Identify performance bottlenecks in current codebase"

# Data analysis swarm
./tools/claude-flow swarm "Analyze user behavior patterns from logs" --strategy analysis --mode mesh --parallel --output sqlite

# Store analysis results
./tools/claude-flow memory store "performance_analysis" "Bottlenecks identified in database queries"
```

### Maintenance Workflow

```bash
# System maintenance with safety controls
./tools/claude-flow swarm "Update dependencies and security patches" --strategy maintenance --mode centralized --monitor

# Security review
./tools/claude-flow sparc run reviewer "Security audit of authentication system"

# Export maintenance logs
./tools/claude-flow memory export maintenance_log.json
```

## Integration Patterns

### Memory-Driven Coordination

Use Memory to coordinate information across multiple SPARC modes and swarm operations:

```bash
# Store architecture decisions
./tools/claude-flow memory store "system_architecture" "Microservices with API Gateway pattern"

# All subsequent operations can reference this decision
./tools/claude-flow sparc run coder "Implement user service based on system_architecture in memory"
./tools/claude-flow sparc run tester "Create integration tests for microservices architecture"
```

### Multi-Stage Development

Coordinate complex development through staged execution:

```bash
# Stage 1: Research and planning
./tools/claude-flow sparc run researcher "Research authentication best practices"
./tools/claude-flow sparc run architect "Design authentication system architecture"

# Stage 2: Implementation
./tools/claude-flow sparc tdd "User registration and login functionality"
./tools/claude-flow sparc run coder "Implement JWT token management"

# Stage 3: Testing and deployment
./tools/claude-flow sparc run tester "Comprehensive security testing"
./tools/claude-flow swarm "Deploy authentication system" --strategy maintenance --mode centralized
```

### Enterprise Integration

For enterprise environments with additional tooling:

```bash
# Project management integration
./tools/claude-flow project create "authentication-system"
./tools/claude-flow project switch "authentication-system"

# Security compliance
./tools/claude-flow security scan
./tools/claude-flow security audit

# Analytics and monitoring
./tools/claude-flow analytics dashboard
./tools/claude-flow deploy production --monitor
```

## Advanced Batch Tool Patterns

### TodoWrite Coordination

Always use TodoWrite for complex task coordination:

```javascript
TodoWrite([
  {
    id: "architecture_design",
    content: "Design system architecture and component interfaces",
    status: "pending",
    priority: "high",
    dependencies: [],
    estimatedTime: "60min",
    assignedAgent: "architect"
  },
  {
    id: "frontend_development", 
    content: "Develop React components and user interface",
    status: "pending",
    priority: "medium",
    dependencies: ["architecture_design"],
    estimatedTime: "120min",
    assignedAgent: "frontend_team"
  }
]);
```

### Task and Memory Integration

Launch coordinated agents with shared memory:

```javascript
// Store architecture in memory
Task("System Architect", "Design architecture and store specs in Memory");

// Other agents use memory for coordination
Task("Frontend Team", "Develop UI using Memory architecture specs");
Task("Backend Team", "Implement APIs according to Memory specifications");
```

## Code Style Preferences

- Use ES modules (import/export) syntax
- Destructure imports when possible
- Use TypeScript for all new code
- Follow existing naming conventions
- Add JSDoc comments for public APIs
- Use async/await instead of Promise chains
- Prefer const/let over var

## Workflow Guidelines

- Always run typecheck after making code changes
- Run tests before committing changes
- Use meaningful commit messages
- Create feature branches for new functionality
- Ensure all tests pass before merging

## Important Notes

- **Use TodoWrite extensively** for all complex task coordination
- **Leverage Task tool** for parallel agent execution on independent work
- **Store all important information in Memory** for cross-agent coordination
- **Use batch file operations** whenever reading/writing multiple files
- **Check .claude/commands/** for detailed command documentation
- **All swarm operations include automatic batch tool coordination**
- **Monitor progress** with TodoRead during long-running operations
- **Enable parallel execution** with --parallel flags for maximum efficiency

This configuration ensures optimal use of Claude Code's batch tools for swarm orchestration and parallel task execution with full Claude-Flow capabilities.
