
from coup_env.coup_env import CoupEnv

class CurriculumEnv(CoupEnv):
    """Wrapper around environment to modify reward for curriculum learning.

    :param env: Environment to learn in
    :type env: PettingZoo-style environment
    :param LESSON: LESSON settings for curriculum learning
    :type LESSON: dict
    """

    def __init__(self, LESSON):
            super().__init__(LESSON["n_players"])  # Corrected `super()` call
            self.LESSON = LESSON

    def _get_reward(self, agent, prev_state) -> int:
        """Processes and returns reward from environment according to LESSON criteria.

        :param done: Environment has terminated
        :type done: bool
        :param player: Player who we are checking, 0 or 1
        :type player: int
        """
        
        # Get rewards for each game state change
        win  = self.LESSON.get('rewards').get('win')
        lose = self.LESSON.get('rewards').get('lose')
        coins = self.LESSON.get('rewards').get('coins')
        kill = self.LESSON.get('rewards').get('kill')
        lose_life = self.LESSON.get('rewards').get('lose_life')
        play_continues = self.LESSON.get('rewards').get('play_continues')
        
        # init reward for this agent
        reward = 0
        
        # check if agent just won game
        if self._agent_won(agent):
            reward+=win
        
        # check if agent just lost game
        if self._agent_lost(agent):
            reward+=lose
        
        # check if agent got coins. Get coins value reward for each coin you get
        if self._agent_gained_coins(agent, prev_state)>0:
            reward+=coins*self._agent_gained_coins(agent,prev_state)
        
        # check if agent killed someone
        if self._agent_killed(agent, prev_state):
            reward+=kill
            
        # check if agent lost life
        if self._agent_lost_life(agent, prev_state):
            reward+=lose_life
        return round(float(reward),2)

