"""Relationships between entity populations."""

from .link import Link
from .many2one import Many2OneLink
from .one2many import One2ManyLink

__all__ = ["Link", "Many2OneLink", "One2ManyLink"]
