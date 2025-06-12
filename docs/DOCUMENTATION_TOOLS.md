# Documentation Tools

## Why Node.js Files in a Python Project?

The `package.json` and `package-lock.json` files in this Python project are **ONLY** for documentation tooling:

- **Purpose**: Markdown linting for documentation quality
- **Tools**: markdownlint-cli2 for consistent documentation formatting
- **NOT**: Part of the Python application runtime

## Usage

```bash
# Install documentation tools (one-time)
npm install

# Lint markdown files
npm run lint:md

# Auto-fix markdown issues
npm run lint:md:fix
```

## For Python Developers

You can safely ignore these files unless you're:

1. Working on documentation
2. Want to ensure your markdown follows best practices

The Python application runs entirely without Node.js.
