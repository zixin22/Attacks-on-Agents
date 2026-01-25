"""
AutoDan Evolutionary Optimization System

基于进化算法的Prompt优化框架，用于生成能够绕过安全过滤器的trigger instructions。
"""

__version__ = "1.0.0"
__author__ = "AutoDan Team"

from .config import Config
from .population import Population, Individual
from .proposer import Proposer
from .evaluator import Evaluator
from .evolutionary_optimizer import EvolutionaryOptimizer

__all__ = [
    'Config',
    'Population',
    'Individual',
    'Proposer',
    'Evaluator',
    'EvolutionaryOptimizer'
]
