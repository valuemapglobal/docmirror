# Security Policy

## Supported Versions

| Version | Supported          |
|---------|--------------------|
| 0.1.x   | ✅ Current release |

## Reporting a Vulnerability

If you discover a security vulnerability in DocMirror, please report it responsibly.

**Do NOT open a public GitHub issue for security vulnerabilities.**

### How to Report

1. Email: **docmirror-security@googlegroups.com**
2. Include:
   - Description of the vulnerability
   - Steps to reproduce
   - Potential impact
   - Suggested fix (if any)

### Response Timeline

- **Acknowledgment**: Within 48 hours
- **Assessment**: Within 7 days
- **Fix release**: Within 30 days for critical issues

### Scope

The following are in scope:
- Code execution vulnerabilities in document parsing
- Path traversal during file processing
- Denial of service through malformed documents
- Information disclosure through error messages
- Dependencies with known CVEs

### Out of Scope

- Issues in optional third-party dependencies (report to upstream)
- Social engineering attacks
- Issues requiring physical access to the machine

## Security Best Practices

When using DocMirror in production:

1. **Sandbox document parsing** — run in a container with restricted filesystem access
2. **Validate file sizes** — set `DOCMIRROR_MAX_PAGES` to limit resource usage
3. **Use Redis caching** — avoid re-parsing the same malicious file
4. **Keep dependencies updated** — run `pip audit` regularly
