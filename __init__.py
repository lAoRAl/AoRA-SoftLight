"""ComfyUI entry point for AoRA Soft Light."""
if __package__:
    from .aora_soft_light import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
else:
    # Allow pytest to collect a checkout whose folder name contains a hyphen.
    from aora_soft_light import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
