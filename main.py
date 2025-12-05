# Main script to run train + evaluation

from data_loading import load_dataset_df
from modules.classifier import SKClassifier
from utils import load_config, prepare_data



if __name__ == "__main__":
    # load config
    config = load_config()
    
    # load data
    dataset_df = load_dataset_df(config)
    # prepare data
    X, y = prepare_data(dataset_df)

    # load classifier model
    classifier = SKClassifier(config['model']['classifier'], config)
    classifier.cross_validate(X, y, k=10)

    classifier.fit(X, y)

    classifier.score(X, y)
    # train model
    train_classifier()
    # evaluate model
    evaluate_classifier()
    # save model
    save_model()
    ...