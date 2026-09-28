# Checkpoint and data safety

Use a supported, patched PyTorch release (minimum **2.10.0**) and matching
torchvision. Keep dependencies updated. Versions through 2.5.1 and through
2.9.1 have known restricted-loader vulnerabilities:
[CVE-2025-32434](https://github.com/pytorch/pytorch/security/advisories/GHSA-53q9-r3pm-6pq6)
and [CVE-2026-24747](https://github.com/pytorch/pytorch/security/advisories/GHSA-63cw-57p8-fm3p).

All project checkpoint, resume-state, decoder, text-embedding and feature-map
loads use `weights_only=True` and CPU placement. The project never adds custom
allowlisted globals; PyTorch's own built-in registrations are left unchanged.
The converter supports legacy **state-dict key layouts**, not arbitrary pickled
Python objects. A restricted-load failure must not be worked around by disabling
the restriction or registering unknown classes. There is no unsafe fallback.

Only load models and datasets from sources you trust. Check release checksums
against the values published through the trusted project repository. Checksums
detect changed bytes, not malicious authorship. Restricted loading is not a
sandbox: hostile files can still exhaust memory or exploit other native code.
If a file's provenance is uncertain, do not load it on your workstation/server.
Optional third-party packages have their own loading and download policies.

Do not include private data, credentials, raw tracking logs, machine metadata,
or unpublished model files in issues or pull requests. Report vulnerabilities
privately to the maintainers using GitHub's private reporting feature when
enabled, or the authors' contact details on the project page.
