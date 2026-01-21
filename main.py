"""
Main script for feature extraction, evaluation, and classification
=================================================================

Supports:
- Multiple pretrained audio encoders
- Supervised (linear / attentive) evaluation
- Unsupervised evaluation (clustering, ranking)

"""

import os
import torch
import argparse
from sklearn.metrics import accuracy_score, confusion_matrix

# ============================
# Imports from your project
# ============================

from Unsupervised_evaluation import (
    clustering_MI,
    rank_similarity_score
)

from Supervised_evaluation import (
    fit_attentive_layer,
    fit_linear_layer
)

# Feature extractors (assumed to exist in your project)
from Load_model import (
    load_train_data_loader,
    load_test_data_loader,
    prediction_BEATS,
    prediction_HuBERT,
    prediction_AudioMAE,
    prediction_WavLM,
    prediction_Data2vec,
    prediction_Perch2,
    prediction_BirdMAE,
    prediction_wav2vec,
    prediction_HuBERTAS,
    MelSpec,
    get_Bioacoustic_features
)

# ============================
# Configuration
# ============================

# DEFAULT_TRAIN_DIR = "/projects/0/vusr0637/shipsEar/train/"
# DEFAULT_TEST_DIR = "/projects/0/vusr0637/shipsEar/test/"

NUM_CLASSES = 5
BATCH_SIZE = 32
SAMPLE_LEN = 10


# ============================
# Feature extraction
# ============================

def extract_features(model_name, train_loader, test_loader):
    """
    Run the selected feature extractor.
    """

    if model_name == "BEATS":
        return (
            prediction_BEATS(train_loader, mean_pooling=True),
            prediction_BEATS(test_loader, mean_pooling=True)
        )

    elif model_name == "HuBERT":
        return (
            prediction_HuBERT(train_loader),
            prediction_HuBERT(test_loader)
        )

    elif model_name == "AudioMAE":
        return (
            prediction_AudioMAE(train_loader, mean_pooling=True),
            prediction_AudioMAE(test_loader, mean_pooling=True)
        )

    elif model_name == "WavLM":
        return (
            prediction_WavLM(train_loader),
            prediction_WavLM(test_loader)
        )

    elif model_name == "Data2vec":
        return (
            prediction_Data2vec(train_loader),
            prediction_Data2vec(test_loader)
        )

    elif model_name == "Perch2":
        return (
            prediction_Perch2(train_loader),
            prediction_Perch2(test_loader)
        )

    elif model_name == "BirdMAE":
        return (
            prediction_BirdMAE(train_loader),
            prediction_BirdMAE(test_loader)
        )

    elif model_name == "Wav2Vec":
        return (
            prediction_wav2vec(train_loader),
            prediction_wav2vec(test_loader)
        )

    elif model_name == "HuBERTAS":
        return (
            prediction_HuBERTAS(train_loader),
            prediction_HuBERTAS(test_loader)
        )

    elif model_name == "MelSpec":
        return (
            MelSpec(train_loader),
            MelSpec(test_loader)
        )

    elif model_name == "Bioacoustic":
        return (
            get_Bioacoustic_features(train_loader),
            get_Bioacoustic_features(test_loader)
        )

    else:
        raise ValueError(f"Unknown model: {model_name}")


# ============================
# Supervised evaluation
# ============================

def supervised_evaluation(
    X_train,
    y_train,
    X_test,
    y_test,
    classifier_type,
    model_path
):
    """
    Train and evaluate a classifier.
    """

    if classifier_type == "linear":
        clf = fit_linear_layer(
            X_train, y_train, NUM_CLASSES, model_path
        )
    elif classifier_type == "attentive":
        clf = fit_attentive_layer(
            X_train, y_train, NUM_CLASSES, model_path
        )
    else:
        raise ValueError("classifier_type must be 'linear' or 'attentive'")

    X_test_tensor = torch.from_numpy(X_test).float()

    with torch.no_grad():
        logits = clf(X_test_tensor)
        predictions = torch.argmax(logits, dim=1).cpu().numpy()

    print("Accuracy:", accuracy_score(y_test, predictions))
    print("Confusion matrix:\n", confusion_matrix(y_test, predictions))

    return logits, predictions


# ============================
# Unsupervised evaluation
# ============================

def unsupervised_evaluation(
    X_train,
    X_test,
    labels_test,
    recordings_test
):
    """
    Run unsupervised metrics.
    """

    clustering_MI(X_train, X_test, labels_test, n_clusters=NUM_CLASSES)

    rank_similarity_score(X_test, labels_test)
    rank_similarity_score(X_test, recordings_test)


# ============================
# Main
# ============================

def main(args):

    # ------------------------
    # Data loaders
    # ------------------------
    train_loader = load_train_data_loader(
        args.train_dir,
        BATCH_SIZE,
        sample_len=SAMPLE_LEN,
        return_recording=True
    )

    test_loader = load_test_data_loader(
        args.test_dir,
        BATCH_SIZE,
        sample_len=SAMPLE_LEN,
        return_recording=True
    )

    # ------------------------
    # Feature extraction
    # ------------------------
    (
        result_train,
        labels_train,
        recordings_train,
        timestamps_train
    ), (
        result_test,
        labels_test,
        recordings_test,
        timestamps_test
    ) = extract_features(args.model, train_loader, test_loader)

    # ------------------------
    # Supervised evaluation
    # ------------------------
    if args.supervised:
        logits_test, preds_test = supervised_evaluation(
            result_train,
            labels_train,
            result_test,
            labels_test,
            args.classifier,
            args.model_out
        )

    # ------------------------
    # Unsupervised evaluation
    # ------------------------
    if args.unsupervised:
        unsupervised_evaluation(
            result_train,
            result_test,
            labels_test,
            recordings_test
        )


# ============================
# CLI
# ============================

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Audio representation evaluation"
    )

    parser.add_argument("--train_dir", type=str)
    parser.add_argument("--test_dir", type=str)

    parser.add_argument(
        "--model",
        type=str,
        default="BEATS",
        choices=[
            "BEATS", "HuBERT", "AudioMAE",
            "WavLM", "Data2vec", "MelSpec",
            "Perch2", "BirdMAE", "Wav2Vec",
            "HuBERTAS", "Bioacoustic"
        ]
    )

    parser.add_argument(
        "--classifier",
        type=str,
        default="linear",
        choices=["linear", "attentive"]
    )

    parser.add_argument(
        "--model_out",
        type=str,
        default="models/Classifiers/model.pt"
    )

    parser.add_argument("--supervised", action="store_true")
    parser.add_argument("--unsupervised", action="store_true")

    args = parser.parse_args()
    main(args)
