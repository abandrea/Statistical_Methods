import pandas as pd


class Handler:
    TARGET = "doctorco"
    DROP_COLS = ['constant']

    def __init__(self, data_path='data/HealthCareAustralia.csv',
                 train_indices_path='data/train_indices.csv',
                 test_indices_path='data/test_indices.csv'):
        """read the data"""
        self.df = pd.read_csv(data_path, index_col=0)
        self.train_indices_path = train_indices_path
        self.test_indices_path = test_indices_path
        self.train_indices = pd.read_csv(train_indices_path)
        self.test_indices = pd.read_csv(test_indices_path)
        self.x_train = None
        self.x_test = None
        self.y_train = None
        self.y_test = None
        self.drop_cols(Handler.DROP_COLS)

    def __repr__(self):
        return str(self.df)

    def drop_cols(self, col_names):
        """drop columns"""
        for name in col_names:
            if name in self.df.columns:
                self.df = self.df.drop([name], axis=1)

    def shuffle(self):
        """change the order of the data"""
        self.df = self.df.sample(frac=1).reset_index(drop=True)

    def get_df(self):
        """get the data"""
        return self.df

    def set_df(self, df):
        """set the data"""
        self.df = df

    def train_test_split(self):
        x = self.get_df().drop([Handler.TARGET], axis=1)
        y = self.get_df()[Handler.TARGET]

        train_indices = self.train_indices['indices']
        test_indices = self.test_indices['indices']
        self.x_train = x.iloc[train_indices]
        self.x_test = x.iloc[test_indices]
        self.y_train = y.iloc[train_indices]
        self.y_test = y.iloc[test_indices]

        return self.x_train, self.x_test, self.y_train, self.y_test


def example():
    handler = Handler()
    x_train, x_test, y_train, y_test = handler.train_test_split()

    print(handler)
    print(x_train)
    print(y_train)


if __name__ == '__main__':
    example()
