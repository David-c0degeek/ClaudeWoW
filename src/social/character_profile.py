# src/social/character_profile.py

import logging
import json
import os
import random
from typing import Dict, List, Any, Optional

class CharacterProfile:
    """
    Manages the character personality profile for consistent social interactions
    """
    
    def __init__(self, config: Dict):
        """
        Initialize the character profile
        
        Args:
            config: Configuration dictionary
        """
        self.logger = logging.getLogger("wow_ai.social.character_profile")
        self.config = config
        
        # Load or create character profile
        self.profile = self._load_profile()
        
        # Character basics
        self.name = config.get("player_name", "Adventurer")
        self.race = config.get("player_race", "Human")
        self.character_class = config.get("player_class", "Warrior")
        self.level = config.get("player_level", 1)
        
        # Update basics in profile
        self.profile["basics"]["name"] = self.name
        self.profile["basics"]["race"] = self.race
        self.profile["basics"]["class"] = self.character_class
        self.profile["basics"]["level"] = self.level
        
        # Apply configured personality traits
        personality = config.get("player_personality", "friendly and helpful")
        self._apply_personality(personality)
        
        # Save updated profile
        self._save_profile()
        
        self.logger.info(f"CharacterProfile initialized for {self.name}, {self.race} {self.character_class}")
    
    def _load_profile(self) -> Dict:
        """
        Load character profile from file or create default
        
        Returns:
            Dict: Character profile
        """
        profile_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            "data", "character_profile.json"
        )
        
        try:
            if os.path.exists(profile_path):
                with open(profile_path, 'r') as f:
                    profile = json.load(f)
                self.logger.info(f"Loaded character profile from {profile_path}")
                return profile
            else:
                profile = self._create_default_profile()
                
                # Ensure directory exists
                os.makedirs(os.path.dirname(profile_path), exist_ok=True)
                
                # Save default profile
                with open(profile_path, 'w') as f:
                    json.dump(profile, f, indent=2)
                
                self.logger.info(f"Created default character profile at {profile_path}")
                return profile
        except Exception as e:
            self.logger.error(f"Error loading character profile: {e}")
            return self._create_default_profile()
    
    def _create_default_profile(self) -> Dict:
        """
        Create default character profile
        
        Returns:
            Dict: Default character profile
        """
        return {
            "basics": {
                "name": self.config.get("player_name", "Adventurer"),
                "race": self.config.get("player_race", "Human"),
                "class": self.config.get("player_class", "Warrior"),
                "level": self.config.get("player_level", 1),
                "faction": "Alliance" if self.config.get("player_race", "Human") in [
                    "Human", "Dwarf", "Night Elf", "Gnome", "Draenei", "Worgen"
                ] else "Horde"
            },
            "background": {
                "origin": "",
                "backstory": "",
                "motivations": [],
                "achievements": []
            },
            "personality": {
                "traits": [],
                "values": [],
                "likes": [],
                "dislikes": [],
                "communication_style": "",
                "temperament": ""
            },
            "social": {
                "default_greeting": "",
                "farewell": "",
                "common_phrases": [],
                "attitude_to_strangers": "",
                "attitude_to_friends": "",
                "attitude_to_enemies": ""
            },
            "class_specific": {
                "role_preference": "",
                "fighting_style": "",
                "specialty": "",
                "class_pride": 0
            }
        }
    
    def _save_profile(self) -> None:
        """
        Save character profile to file
        """
        profile_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            "data", "character_profile.json"
        )
        
        try:
            with open(profile_path, 'w') as f:
                json.dump(self.profile, f, indent=2)
            self.logger.debug(f"Saved character profile to {profile_path}")
        except Exception as e:
            self.logger.error(f"Error saving character profile: {e}")
    
    def _apply_personality(self, personality_desc: str) -> None:
        """
        Apply personality description to character profile
        
        Args:
            personality_desc: Personality description
        """
        # Parse personality description
        traits = [trait.strip() for trait in personality_desc.split(' and ')]
        traits.extend([trait.strip() for trait in personality_desc.split(',')])
        traits = list(set(traits))  # Remove duplicates
        
        # Update personality traits
        self.profile["personality"]["traits"] = traits
        
        # Map common traits to other personality aspects
        trait_mappings = {
            "friendly": {
                "values": ["friendship", "community"],
                "communication_style": "warm and approachable",
                "attitude_to_strangers": "welcoming",
                "common_phrases": ["Hello friend!", "Good to see you!"]
            },
            "helpful": {
                "values": ["service", "assistance"],
                "attitude_to_strangers": "supportive",
                "common_phrases": ["Need a hand?", "I can help with that."]
            },
            "brave": {
                "values": ["courage", "honor"],
                "temperament": "courageous",
                "fighting_style": "bold and direct"
            },
            "cautious": {
                "values": ["safety", "preparation"],
                "temperament": "careful",
                "fighting_style": "strategic and measured"
            },
            "funny": {
                "communication_style": "humorous",
                "common_phrases": ["That's hilarious!", "Here's a good one..."]
            },
            "serious": {
                "communication_style": "direct and to the point",
                "temperament": "focused"
            },
            "loyal": {
                "values": ["loyalty", "commitment"],
                "attitude_to_friends": "fiercely devoted"
            },
            "intelligent": {
                "communication_style": "thoughtful and articulate",
                "values": ["knowledge", "wisdom"]
            }
        }
        
        # Apply mappings for recognized traits
        for trait in traits:
            if trait in trait_mappings:
                mapping = trait_mappings[trait]
                for key, value in mapping.items():
                    if isinstance(value, list):
                        self.profile["personality"].setdefault(key, []).extend(value)
                    else:
                        if not self.profile["personality"].get(key):
                            self.profile["personality"][key] = value
        
        # Apply class-specific characteristics
        self._apply_class_characteristics()
        
        # Generate backstory elements if empty
        if not self.profile["background"]["origin"]:
            self._generate_backstory()
        
        # Set default greeting and farewell if not already set
        if not self.profile["social"]["default_greeting"]:
            self._set_default_social_phrases()
    
    def _apply_class_characteristics(self) -> None:
        """
        Apply class-specific characteristics to the profile
        """
        class_characteristics = {
            "warrior": {
                "role_preference": "tank" if random.random() < 0.7 else "dps",
                "fighting_style": "direct and powerful",
                "specialty": random.choice(["Arms", "Fury", "Protection"]),
                "class_pride": random.randint(7, 10),
                "values": ["strength", "bravery", "combat prowess"]
            },
            "paladin": {
                "role_preference": random.choice(["tank", "healer", "dps"]),
                "fighting_style": "righteous and disciplined",
                "specialty": random.choice(["Holy", "Protection", "Retribution"]),
                "class_pride": random.randint(8, 10),
                "values": ["justice", "righteousness", "service"]
            },
            "hunter": {
                "role_preference": "dps",
                "fighting_style": "ranged and tactical",
                "specialty": random.choice(["Beast Mastery", "Marksmanship", "Survival"]),
                "class_pride": random.randint(6, 10),
                "values": ["nature", "precision", "self-reliance"]
            },
            "rogue": {
                "role_preference": "dps",
                "fighting_style": "stealthy and opportunistic",
                "specialty": random.choice(["Assassination", "Combat", "Subtlety"]),
                "class_pride": random.randint(7, 10),
                "values": ["cunning", "efficiency", "freedom"]
            },
            "priest": {
                "role_preference": "healer" if random.random() < 0.7 else "dps",
                "fighting_style": "thoughtful and controlled",
                "specialty": random.choice(["Discipline", "Holy", "Shadow"]),
                "class_pride": random.randint(6, 10),
                "values": ["faith", "balance", "knowledge"]
            },
            "shaman": {
                "role_preference": random.choice(["healer", "dps"]),
                "fighting_style": "elemental and spiritual",
                "specialty": random.choice(["Elemental", "Enhancement", "Restoration"]),
                "class_pride": random.randint(7, 10),
                "values": ["elements", "ancestry", "balance"]
            },
            "mage": {
                "role_preference": "dps",
                "fighting_style": "arcane and calculated",
                "specialty": random.choice(["Arcane", "Fire", "Frost"]),
                "class_pride": random.randint(8, 10),
                "values": ["intellect", "power", "knowledge"]
            },
            "warlock": {
                "role_preference": "dps",
                "fighting_style": "dark and controlling",
                "specialty": random.choice(["Affliction", "Demonology", "Destruction"]),
                "class_pride": random.randint(7, 10),
                "values": ["power", "knowledge", "control"]
            },
            "druid": {
                "role_preference": random.choice(["tank", "healer", "dps"]),
                "fighting_style": "adaptive and natural",
                "specialty": random.choice(["Balance", "Feral", "Guardian", "Restoration"]),
                "class_pride": random.randint(7, 10),
                "values": ["nature", "balance", "versatility"]
            },
            "death knight": {
                "role_preference": "tank" if random.random() < 0.5 else "dps",
                "fighting_style": "grim and relentless",
                "specialty": random.choice(["Blood", "Frost", "Unholy"]),
                "class_pride": random.randint(8, 10),
                "values": ["strength", "dominance", "vengeance"]
            }
        }
        
        class_lower = self.character_class.lower()
        if class_lower in class_characteristics:
            # Apply class characteristics
            for key, value in class_characteristics[class_lower].items():
                if key == "values":
                    self.profile["personality"].setdefault("values", []).extend(value)
                else:
                    self.profile["class_specific"][key] = value
    
    def _generate_backstory(self) -> None:
        """
        Generate simple backstory elements based on race and class
        """
        race_origins = {
            "human": ["Stormwind", "Westfall", "Redridge", "Elwynn Forest"],
            "dwarf": ["Ironforge", "Dun Morogh", "Loch Modan"],
            "night elf": ["Darnassus", "Teldrassil", "Darkshore"],
            "gnome": ["Gnomeregan", "Dun Morogh"],
            "draenei": ["The Exodar", "Azuremyst Isle"],
            "worgen": ["Gilneas"],
            "orc": ["Orgrimmar", "Durotar"],
            "undead": ["Tirisfal Glades", "Lordaeron"],
            "tauren": ["Mulgore", "Thunder Bluff"],
            "troll": ["Echo Isles", "Durotar"],
            "blood elf": ["Silvermoon", "Eversong Woods"],
            "goblin": ["Kezan"]
        }
        
        class_motivations = {
            "warrior": ["prove my strength", "protect the weak", "master combat", "seek glory in battle"],
            "paladin": ["uphold justice", "vanquish evil", "serve the Light", "protect the innocent"],
            "hunter": ["explore the wilderness", "track rare beasts", "perfect my marksmanship", "live freely"],
            "rogue": ["acquire wealth", "perfect my skills", "live by my own rules", "outfox my enemies"],
            "priest": ["spread faith", "heal the wounded", "seek enlightenment", "banish darkness"],
            "shaman": ["commune with the elements", "restore balance", "honor ancestors", "understand nature"],
            "mage": ["master arcane knowledge", "research new spells", "discover magical artifacts", "prove my intellect"],
            "warlock": ["gain forbidden knowledge", "increase my power", "command demons", "uncover secrets"],
            "druid": ["protect nature", "maintain balance", "master shapeshifting", "preserve ancient knowledge"],
            "death knight": ["seek redemption", "find purpose", "exact vengeance", "master death magic"]
        }
        
        # Set origin
        race_lower = self.race.lower()
        if race_lower in race_origins:
            self.profile["background"]["origin"] = random.choice(race_origins[race_lower])
        
        # Set motivations
        class_lower = self.character_class.lower()
        if class_lower in class_motivations:
            self.profile["background"]["motivations"] = random.sample(class_motivations[class_lower], 
                                                                     k=min(2, len(class_motivations[class_lower])))
        
        # Generate simple achievements based on level
        level = self.level
        achievements = []
        
        if level >= 10:
            achievements.append(f"Mastered basic {self.character_class} abilities")
        if level >= 20:
            achievements.append(f"Explored multiple regions of {self.profile['basics']['faction']} territory")
        if level >= 30:
            achievements.append("Joined my first dungeon groups")
        if level >= 40:
            achievements.append("Learned to ride a mount")
        if level >= 50:
            achievements.append("Survived countless battles")
        if level >= 60:
            achievements.append("Reached the pinnacle of my abilities")
        
        self.profile["background"]["achievements"] = achievements
    
    def _set_default_social_phrases(self) -> None:
        """
        Set default social phrases based on race and personality
        """
        race_lower = self.race.lower()
        class_lower = self.character_class.lower()
        
        # Greetings by race
        race_greetings = {
            "human": ["Hello there!", "Greetings, friend.", "Well met!"],
            "dwarf": ["Hail, friend!", "Well met!", "How are ye?"],
            "night elf": ["Ishnu-alah.", "Greetings, traveler.", "The goddess smiles upon you."],
            "gnome": ["Hi there!", "Hello hello!", "Great to meet you!"],
            "draenei": ["The Light be with you.", "Good health to you.", "May the Naaru protect you."],
            "worgen": ["Greetings.", "Well met.", "What brings you here?"],
            "orc": ["Lok'tar!", "Strength and honor.", "Blood and thunder!"],
            "undead": ["Greetings.", "What do you want?", "State your business."],
            "tauren": ["Hail, traveler.", "Earth Mother bless you.", "Walk with the Earthmother."],
            "troll": ["Hey mon!", "What you be needin'?", "Stay away from the voodoo."],
            "blood elf": ["The eternal sun guides us.", "State your business.", "Well met."],
            "goblin": ["Hey there!", "Time is money, friend!", "What's the deal?"]
        }
        
        # Class-specific phrases
        class_phrases = {
            "warrior": ["For glory!", "Stand and fight!", "My blade is sharp."],
            "paladin": ["The Light protects.", "Justice shall prevail.", "For the Light!"],
            "hunter": ["The hunt calls.", "Tracked you from miles away.", "My aim is true."],
            "rogue": ["Watch your back.", "The shadows hide me.", "Quick and quiet."],
            "priest": ["Light be with you.", "Find peace in the Light.", "Faith guides me."],
            "shaman": ["The elements guide me.", "Earth, fire, air, water.", "The spirits speak."],
            "mage": ["Knowledge is power.", "The arcane calls.", "Fascinating, isn't it?"],
            "warlock": ["Your soul will serve.", "Power has its price.", "Knowledge is... valuable."],
            "druid": ["Nature's balance must be preserved.", "The wild calls.", "Balance in all things."],
            "death knight": ["Death comes for all.", "No rest for the dead.", "Suffering lingers."]
        }
        
        # Set greeting
        if race_lower in race_greetings:
            self.profile["social"]["default_greeting"] = random.choice(race_greetings[race_lower])
        else:
            self.profile["social"]["default_greeting"] = "Hello there!"
        
        # Set farewell
        farewells = ["Farewell!", "Until next time.", "Safe travels.", "Be seeing you!"]
        self.profile["social"]["farewell"] = random.choice(farewells)
        
        # Set common phrases
        common_phrases = []
        
        # Add race-specific phrases
        if race_lower in race_greetings:
            common_phrases.extend(race_greetings[race_lower])
        
        # Add class-specific phrases
        if class_lower in class_phrases:
            common_phrases.extend(class_phrases[class_lower])
        
        # Add general phrases
        general_phrases = [
            "Interesting.", 
            "I see.", 
            "Indeed.", 
            "Is that so?",
            "Thanks!",
            "Good to know.",
            "Haha!",
            "Of course.",
            "Not a problem.",
            "I understand."
        ]
        
        common_phrases.extend(general_phrases)
        
        # Remove duplicates and limit
        self.profile["social"]["common_phrases"] = list(set(common_phrases))[:10]
        
        # Set attitudes
        temperament = self.profile["personality"].get("temperament", "")
        
        if "friendly" in self.profile["personality"]["traits"]:
            self.profile["social"]["attitude_to_strangers"] = "welcoming and open"
            self.profile["social"]["attitude_to_friends"] = "loyal and supportive"
        elif "cautious" in self.profile["personality"]["traits"]:
            self.profile["social"]["attitude_to_strangers"] = "reserved but polite"
            self.profile["social"]["attitude_to_friends"] = "protective and reliable"
        else:
            self.profile["social"]["attitude_to_strangers"] = "neutral but respectful"
            self.profile["social"]["attitude_to_friends"] = "reliable and honest"
        
        # Set enemy attitude based on class
        if class_lower in ["warrior", "death knight", "paladin"]:
            self.profile["social"]["attitude_to_enemies"] = "confrontational and direct"
        elif class_lower in ["rogue", "hunter", "warlock"]:
            self.profile["social"]["attitude_to_enemies"] = "calculating and opportunistic"
        else:
            self.profile["social"]["attitude_to_enemies"] = "cautious but determined"
    
    def get_profile_as_prompt(self) -> str:
        """
        Get character profile formatted as an LLM prompt
        
        Returns:
            str: Character profile as prompt text
        """
        basics = self.profile["basics"]
        personality = self.profile["personality"]
        social = self.profile["social"]
        background = self.profile["background"]
        class_specific = self.profile["class_specific"]
        
        prompt = f"""
Character Profile for WoW Player Character:

NAME: {basics['name']}
RACE: {basics['race']}
CLASS: {basics['class']}
LEVEL: {basics['level']}
FACTION: {basics['faction']}

BACKGROUND:
- Origin: {background['origin']}
- Main motivations: {', '.join(background['motivations'])}
- Notable achievements: {', '.join(background['achievements'])}

PERSONALITY:
- Key traits: {', '.join(personality['traits'])}
- Values: {', '.join(personality['values'])}
- Communication style: {personality['communication_style']}
- Temperament: {personality['temperament']}

CLASS IDENTITY:
- Role preference: {class_specific['role_preference']}
- Fighting style: {class_specific['fighting_style']}
- Specialty: {class_specific['specialty']}

SOCIAL BEHAVIOR:
- Default greeting: "{social['default_greeting']}"
- Common farewell: "{social['farewell']}"
- Attitude to strangers: {social['attitude_to_strangers']}
- Attitude to friends: {social['attitude_to_friends']}
- Attitude to enemies: {social['attitude_to_enemies']}
- Frequently uses phrases like: {', '.join([f'"{phrase}"' for phrase in social['common_phrases'][:3]])}
"""
        
        return prompt
    
    def get_greeting(self, target: str = "", relationship: str = "neutral") -> str:
        """
        Get an appropriate greeting
        
        Args:
            target: Target name
            relationship: Relationship status
        
        Returns:
            str: Appropriate greeting
        """
        default_greeting = self.profile["social"]["default_greeting"]
        
        if not target:
            return default_greeting
        
        # Add target name for personalization
        if relationship == "friendly":
            greetings = [
                f"Hey {target}! Good to see you!",
                f"{default_greeting} {target}, how have you been?",
                f"There you are, {target}! How's it going?"
            ]
            return random.choice(greetings)
        
        elif relationship == "unfriendly":
            greetings = [
                f"Oh. It's {target}.",
                f"What do you want, {target}?",
                f"{default_greeting} {target}."  # More neutral
            ]
            return random.choice(greetings)
        
        else:  # neutral
            greetings = [
                f"{default_greeting} {target}.",
                f"Hello there, {target}.",
                f"Greetings, {target}."
            ]
            return random.choice(greetings)
    
    def get_random_phrase(self) -> str:
        """
        Get a random common phrase
        
        Returns:
            str: Random phrase
        """
        common_phrases = self.profile["social"]["common_phrases"]
        if common_phrases:
            return random.choice(common_phrases)
        else:
            return "Interesting."
    
    def is_appropriate_response(self, message: str, response: str) -> bool:
        """
        Check if a response is appropriate for the character's personality
        
        Args:
            message: Original message
            response: Proposed response
        
        Returns:
            bool: True if the response is appropriate
        """
        # This is a simple implementation
        # In a real system, this could use more sophisticated NLP
        
        # Check for contradictions with values
        values = self.profile["personality"].get("values", [])
        
        # Check character consistency
        traits = self.profile["personality"].get("traits", [])
        
        # Example checks
        if "friendly" in traits and "rude" in response.lower():
            return False
        
        if "serious" in traits and any(word in response.lower() for word in ["lol", "haha", "lmao"]):
            return False
        
        if "honor" in values and any(word in response.lower() for word in ["cheat", "steal", "lie"]):
            return False
        
        # By default, assume the response is appropriate
        return True