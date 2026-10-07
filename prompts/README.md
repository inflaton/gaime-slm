# Prompt templates

**Approved for public release.** The Free-Thinking and JSON-Only templates retain the experimental instruction structure. These public copies omit the same three descriptor examples that were elided from the shared rules in [Appendix A of arXiv v4](https://arxiv.org/html/2606.08051v4).

All remaining merchant-related examples have been replaced with synthetic values created solely for illustration: intermediary and merchant names, store numbers, product/service names, transaction reference codes, street addresses, phone numbers, and websites. The phone examples use the 202-555-01xx range, and the website examples use `.example` domains. Generic state and country codes remain as schema illustrations.

These abridged templates with synthetic examples differ from the historical training prompts. The original experiment templates are retained internally; the reported aggregate results were produced with those original templates. Runs using these public copies are new experiments and do not reproduce the paper's exact inputs.

The transaction to classify is appended after `TRANSACTION:`. The Free-Thinking template supplies an explicit JSON schema; the JSON-Only template relies on the keys listed in the classification rules.
