"""Adversarial attacks for face verification."""

from .base import AdversarialAttack, AttackMode
from .carlini_wagner import CarliniWagnerAttack

__all__ = [
    "AdversarialAttack",
    "AttackMode",
    "CarliniWagnerAttack",
]
