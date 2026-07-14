# Security Policy

## Supported Versions

VecAdvisor is currently alpha software. Security fixes are applied to the
latest public release and the `main` branch.

## Reporting A Vulnerability

Please report security issues privately to the maintainer before opening a
public issue. Include:

- affected version or commit
- reproduction steps
- expected impact
- whether the issue requires a malicious database, malicious query input, or
  local filesystem access

VecAdvisor is a CLI advisor. It should never require superuser database
permissions for normal analysis, and it should not store raw query vectors in
its local selectivity cache.
