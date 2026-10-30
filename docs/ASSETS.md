# Checked datasets and model downloads

The [asset manifest](../assets/manifest.json) records SHA-256 hashes for external
teaching data, model weights, tokenizers and bundled polarity reviews. GitHub
sources use commit IDs; Hugging Face models use immutable revisions plus hashes
for each allowed file. Other hosts are fixed by content hash: their URL may remain
mutable, but changed bytes are rejected. Source and terms links are in the manifest
and [provenance register](PROVENANCE.md). A checksum establishes identity, not
permission to redistribute or scientific validity.

## Fetch and verify

Run these commands from the repository root in the relevant environment:

```bash
python scripts/manage_assets.py list
python scripts/manage_assets.py prefetch titanic_seaborn
python scripts/manage_assets.py prefetch bert_attention
python scripts/manage_assets.py verify titanic_seaborn bert_attention
# Explicit opt-in: several GB of public downloads, including neural weights.
python scripts/manage_assets.py prefetch --all
```

`verify` uses offline mode and fails for a missing or changed asset. File downloads
are written atomically under `validation/assets/` (override with `XAI_ASSET_CACHE`).
Model files use the Hugging Face cache. Failed checks never silently replace the
manifest. Review the upstream change, terms and example outputs before deliberately
updating a hash. Do not treat a hash failure as a request to disable verification.

The four native loaders (California housing, 20 Newsgroups, CIFAR-10 and OpenML
bike sharing) keep their own cache layout. Their notebooks check the decoded data,
including row/column order, against the manifest before fitting. The original
California/Newsgroups/CIFAR downloaders also check archive hashes. The manifest
records the OpenML data ID and raw ARFF hash; the notebook checks its decoded table.
Prefetching these archives with this CLI does **not** populate the native caches.
Run their notebooks once online to initialize those caches.

## Offline execution

New foundations, regression, exercises and the shared image/text comparisons use
bundled or generated data. They need no runtime downloads after package installation.
For legacy examples, first execute the selected notebooks online using the runner:

```bash
python scripts/check_notebooks.py --group all --output-dir validation/online
XAI_OFFLINE=1 HF_HUB_OFFLINE=1 python scripts/check_notebooks.py \
  --group all --output-dir validation/offline
```

Use the same checkout and cache directories. `XAI_OFFLINE=1` prevents network access
through this repository's asset helper, not arbitrary library code. Native loaders
can still attempt downloads if their own caches are missing. For a strict network
isolation test, disable network access at the operating-system/container boundary
after warming caches. The runner places native/Hugging Face caches beneath
`validation/cache/`; direct notebook execution may use library defaults instead.

No large model weights or downloaded archives are committed. Hashes for bundled
scikit-learn/scikit-image toy data are indirectly fixed by the platform package
locks; the externally downloaded datasets receive explicit content checks. Native
pickle caches are trusted local artifacts, not safe interchange formats for
untrusted files. Copies of third-party model/data assets retain upstream terms.
