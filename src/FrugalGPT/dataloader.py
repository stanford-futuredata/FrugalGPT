class DataLoader(object):
    def __init__(self,
                 queries,
                 prefix=""
                 ):
        self.queries = queries
        self.prefix = prefix
        return