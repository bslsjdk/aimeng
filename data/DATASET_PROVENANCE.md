# Dataset source and provenance

Default fallback dataset used by the Colab notebook:

- Dataset repository: `Salesforce/wikitext`
- Configuration: `wikitext-2-raw-v1`
- Split: `train`
- Intended use in this project: small English-only pipeline baseline
- Storage: Google Drive at `MyDrive/aimeng_corpus.txt`; raw corpus is not committed to Git
- License/provenance: check the upstream dataset card and license before use or redistribution; record the exact version and SHA-256 in each experiment report
- Limitations: does not establish Chinese, multilingual, dialogue, reasoning, or general language competence

When replacing the corpus, use only data you are authorized to use and record source URL/name, version/date, license, preprocessing, encoding, and SHA-256. Prefer document/source-level train-validation-test separation to reduce leakage.
