"""Harness adapters: one class per harness behind the `HarnessAdapter` protocol."""

from legion.worker.adapters.base import AdapterEvent, HarnessAdapter, ResumeError

__all__ = ["AdapterEvent", "HarnessAdapter", "ResumeError"]
