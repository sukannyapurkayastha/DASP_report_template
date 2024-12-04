from loaders.jitsupeer_loader import JitsupeerLoader

if __name__ == '__main__':
    loader = JitsupeerLoader()

    data = loader.load_data()
    X_train, X_val, X_test, y_train, y_val, y_test = loader.load_data_with_splits()
