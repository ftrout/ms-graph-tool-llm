# Security Policy

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| 1.x.x   | :white_check_mark: |
| < 1.0   | :x:                |

## Reporting a Vulnerability

We take the security of msgraph-tool-agent-8b seriously. If you discover a security vulnerability, please report it responsibly.

### How to Report

1. **Do NOT** open a public GitHub issue for security vulnerabilities
2. Email the security report to: ftrout@users.noreply.github.com
3. Include the following information:
   - Description of the vulnerability
   - Steps to reproduce
   - Potential impact
   - Any suggested fixes (optional)

### What to Expect

- **Acknowledgment**: We will acknowledge receipt of your report within 48 hours
- **Investigation**: We will investigate and validate the issue within 7 days
- **Resolution**: We aim to resolve critical issues within 30 days
- **Disclosure**: We will coordinate with you on public disclosure timing

### Scope

The following are in scope for security reports:

- Code injection vulnerabilities
- Unauthorized access to data
- Prompt injection attacks that bypass safety measures
- Dependency vulnerabilities
- Authentication/authorization bypasses

### Out of Scope

- Issues in dependencies that have already been reported upstream
- Theoretical attacks without proof of concept
- Social engineering attacks
- Physical security issues

## Security Best Practices for Users

When using msgraph-tool-agent-8b:

1. **Validate all outputs**: Never execute generated API calls without human review
2. **Use minimal permissions**: Configure OAuth scopes with least privilege
3. **Protect credentials**: Never include API keys or tokens in prompts
4. **Sanitize inputs**: Validate user inputs before passing to the model
5. **Monitor usage**: Log and audit all API calls made through the agent

## Security Features

- The model generates JSON payloads only; it does not execute API calls
- All generated outputs should be validated before execution
- The package does not store or transmit credentials
