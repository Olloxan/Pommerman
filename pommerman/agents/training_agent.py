from .. import characters
from . import BaseAgent

class TrainingAgent(BaseAgent):

    def __init__(self, *args, **kwargs):
        super(TrainingAgent, self).__init__(*args, **kwargs)

    def act(self, obs, action_space):
        """This agent has its own way of inducing actions."""
        return None
