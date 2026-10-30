"""Content-checked public teaching assets. Model/data bytes are never committed.

The manifest records reviewed bytes and upstream sources. A changed download or
cache is an error, not permission to silently update a scientific example.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parent
MANIFEST_PATH = ROOT / "assets/manifest.json"


def manifest():
    return json.loads(MANIFEST_PATH.read_text())


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_file(path, expected):
    actual = sha256(path)
    if actual != expected:
        raise ValueError(f"Asset checksum mismatch for {Path(path).name}: expected {expected}, got {actual}")
    return Path(path)


def asset_file(key, *, offline=False):
    """Fetch a registered file atomically, or verify its existing cached bytes.

    XAI_OFFLINE=1 forbids network access through this helper. It does not claim
    to disable network access in unrelated libraries or notebook code.
    """
    spec = manifest()["files"][key]
    cache = Path(os.environ.get("XAI_ASSET_CACHE", ROOT / "validation/assets"))
    target = cache / key / spec["filename"]
    if target.is_file():
        return verify_file(target, spec["sha256"])
    if offline or os.environ.get("XAI_OFFLINE") == "1":
        raise FileNotFoundError(f"Missing cached asset {key}; prefetch it before offline execution")
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix="download-", dir=target.parent)
    os.close(descriptor)
    temporary = Path(temporary)
    try:
        with urlopen(spec["url"], timeout=120) as response, temporary.open("wb") as destination:
            shutil.copyfileobj(response, destination, length=1024 * 1024)
        verify_file(temporary, spec["sha256"])
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)
    return target


def hf_model_path(key, *, offline=False):
    """Get only registered model/tokenizer files at an immutable Hub commit.

    Each file is checked independently after the Hub cache resolves the commit.
    Callers load from the returned directory with local_files_only=True and
    trust_remote_code=False where the model API supports these flags.
    """
    from huggingface_hub import snapshot_download

    spec = manifest()["huggingface"][key]
    path = Path(snapshot_download(
        repo_id=spec["repo_id"], revision=spec["revision"],
        allow_patterns=list(spec["files"]),
        local_files_only=offline or os.environ.get("XAI_OFFLINE") == "1",
    ))
    for name, expected in spec["files"].items():
        verify_file(path / name, expected)
    return path


def fingerprint(*objects):
    """Hash decoded dataset contents, including shape/order; independent of pickle.

    DataFrames/Series use UTF-8 CSV with 17-digit floats. Numeric arrays use
    explicit little-endian bytes; text sequences use a length-preserving JSON
    encoding. The pinned loaders/packages define this representation.
    """
    import numpy as np
    import pandas as pd

    digest = hashlib.sha256()
    for obj in objects:
        if isinstance(obj, (pd.DataFrame, pd.Series)):
            payload = obj.to_csv(index=True, float_format="%.17g", lineterminator="\n").encode()
            kind = f"{type(obj).__name__}|{obj.shape}"
        else:
            array = np.asarray(obj)
            kind = f"array|{array.shape}|{array.dtype.kind}"
            if array.dtype.kind in "biufc":
                dtype = array.dtype.newbyteorder("<")
                payload = np.ascontiguousarray(array.astype(dtype)).tobytes()
                kind += f"|{dtype.str}"
            else:
                payload = json.dumps(array.tolist(), ensure_ascii=False, separators=(",", ":")).encode()
        header = kind.encode()
        digest.update(len(header).to_bytes(8, "little"))
        digest.update(header)
        digest.update(len(payload).to_bytes(8, "little"))
        digest.update(payload)
    return digest.hexdigest()


def verify_dataset(key, *objects):
    expected = manifest()["decoded_datasets"][key]["sha256"]
    actual = fingerprint(*objects)
    if actual != expected:
        raise ValueError(f"Decoded dataset {key} changed: expected {expected}, got {actual}")
    return actual


def load_nhanesi(display=False):
    """Same packaged values as SHAP's loader, with registered input bytes."""
    import numpy as np
    import pandas as pd
    X = pd.read_csv(asset_file("nhanesi_X"), index_col=0)
    y = pd.read_csv(asset_file("nhanesi_y"), index_col=0)["y"]
    return X.copy() if display else X, np.asarray(y)


def load_imagenet50():
    import numpy as np
    return (np.load(asset_file("imagenet50_images"), allow_pickle=False).astype(np.float32),
            np.loadtxt(asset_file("imagenet50_labels")))


def load_adult_dataframe():
    """DiCE 0.12's teaching preparation, reading a verified ZIP without extraction.

    Adapted from DiCE's MIT-licensed helpers.load_adult_income_dataset; preserve
    its category mappings and initial 80% subset for continuity with this example.
    """
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from zipfile import ZipFile

    names = ["age", "workclass", "fnlwgt", "education", "educational-num", "marital-status",
             "occupation", "relationship", "race", "gender", "capital-gain", "capital-loss",
             "hours-per-week", "native-country", "income"]
    with ZipFile(asset_file("adult_zip")) as archive, archive.open("adult.data") as source:
        data = pd.read_csv(source, names=names, skipinitialspace=True)
    mappings = {
        "workclass": {"Without-pay": "Other/Unknown", "Never-worked": "Other/Unknown", "?": "Other/Unknown",
                      "Federal-gov": "Government", "State-gov": "Government", "Local-gov": "Government",
                      "Self-emp-not-inc": "Self-Employed", "Self-emp-inc": "Self-Employed"},
        "occupation": {"Adm-clerical": "White-Collar", "Craft-repair": "Blue-Collar", "Exec-managerial": "White-Collar",
                       "Farming-fishing": "Blue-Collar", "Handlers-cleaners": "Blue-Collar", "Machine-op-inspct": "Blue-Collar",
                       "Other-service": "Service", "Priv-house-serv": "Service", "Prof-specialty": "Professional",
                       "Protective-serv": "Service", "Tech-support": "Service", "Transport-moving": "Blue-Collar",
                       "Unknown": "Other/Unknown", "Armed-Forces": "Other/Unknown", "?": "Other/Unknown"},
        "marital-status": {"Married-civ-spouse": "Married", "Married-AF-spouse": "Married", "Married-spouse-absent": "Married", "Never-married": "Single"},
        "race": {"Black": "Other", "Asian-Pac-Islander": "Other", "Amer-Indian-Eskimo": "Other"},
        "education": {"Assoc-voc": "Assoc", "Assoc-acdm": "Assoc", **{x: "School" for x in
                      ["11th", "10th", "7th-8th", "9th", "12th", "5th-6th", "1st-4th", "Preschool"]}},
    }
    data = data.replace(mappings)
    data["income"] = data["income"].map({"<=50K": 0, ">50K": 1})
    if data["income"].isna().any():
        raise ValueError("Unexpected Adult target labels")
    data = data[["age", "workclass", "education", "marital-status", "occupation", "race", "gender", "hours-per-week", "income"]]
    data = data.rename(columns={"marital-status": "marital_status", "hours-per-week": "hours_per_week"})
    subset, _ = train_test_split(data, test_size=0.2, random_state=17)
    return subset.reset_index(drop=True)
