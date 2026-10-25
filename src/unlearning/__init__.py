"""Infrastructure for the specialization/unlearning pilot."""

from unlearning.runtime.settings import apply_hf_cache_default, load_local_env

__all__ = ["__version__"]
__version__ = "0.1.0"

# Load machine-local settings (.env) and point the Hugging Face cache into the
# storage root before any Hugging Face library is imported.
load_local_env()
apply_hf_cache_default()
