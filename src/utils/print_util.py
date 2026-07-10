"""Shared print utility — flushed stdout for all pipeline scripts."""
import functools
print = functools.partial(print, flush=True)
