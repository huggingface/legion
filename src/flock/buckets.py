"""Spec bucket IDs, resolved by the launcher; URLs may add flock's internal prefixes."""

import re
from typing import Annotated, Any

from pydantic import AfterValidator


def validate_bucket_id(value: str) -> str:
    """An optional `[namespace/]name`, without a URL or a path inside the bucket."""
    parts = value.split("/")
    if value and (len(parts) > 2 or any(
            not re.fullmatch(r"[\w.-]+", p, flags=re.ASCII) or p in (".", "..")
            for p in parts)):
        raise ValueError("bucket_id must be [namespace/]name, without hf://buckets/ or a prefix")
    return value


BucketId = Annotated[str, AfterValidator(validate_bucket_id)]


def resolve_bucket_id(bucket_id: str, namespace: str, api: Any) -> str:
    """Qualify a bare name using the explicit namespace, else HF_TOKEN's account."""
    validate_bucket_id(bucket_id)
    if not bucket_id or "/" in bucket_id:
        return bucket_id
    return f"{namespace or api.whoami()['name']}/{bucket_id}"


def bucket_url(bucket_id: str) -> str:
    """A qualified spec ID as a Hub URL, ready for flock's checkpoint/agent prefixes."""
    validate_bucket_id(bucket_id)
    if "/" not in bucket_id:
        raise ValueError("bucket_id must be qualified by flock launch before use")
    return f"hf://buckets/{bucket_id}"
