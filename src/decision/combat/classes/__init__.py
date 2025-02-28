"""
Class-specific combat implementations for different character classes.

Each class has a dedicated combat module that inherits from BaseCombatModule
and implements class-specific combat logic and rotations.
"""

# Import all class modules to make them available
from src.decision.combat.classes.warrior import WarriorCombatModule
from src.decision.combat.classes.mage import MageCombatModule
from src.decision.combat.classes.priest import PriestCombatModule
from src.decision.combat.classes.hunter import HunterCombatModule
from src.decision.combat.classes.rogue import RogueCombatModule
from src.decision.combat.classes.shaman import ShamanCombatModule
from src.decision.combat.classes.paladin import PaladinCombatModule
from src.decision.combat.classes.warlock import WarlockCombatModule
from src.decision.combat.classes.druid import DruidCombatModule