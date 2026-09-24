"""Utilities shared by ATREX public API modules."""

from .device_target import DeviceTarget, detect_device_target

__all__ = ("DeviceTarget", "detect_device_target")
