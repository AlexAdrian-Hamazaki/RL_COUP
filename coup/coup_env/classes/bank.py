class CoinBank():
    def __init__(self):
        self._n = 50      
    def __repr__(self):
        return str(self._n)
    
    def remove(self, n):
        self._n -= n
    def add(self, n):
        self._n += n  
    
    @property
    def n(self):
        return self._n