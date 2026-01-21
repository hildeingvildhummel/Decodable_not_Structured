import pandas as pd
from sklearn.manifold import TSNE
import seaborn as sns

import matplotlib.pyplot as plt
import torch
from AttentivePooling import AttentionPoolingClassifier

def tSNE_visualization(data_subset_train, data_subset_test, labels_train, labels_test, save_base_name):
    ship_types = ['Cargo', 'Passenger', 'Tanker', 'Tug', 'Background']
    markers = {'Cargo': 'o', 'Passenger': 's', 'Tanker': 'X', 'Tug': '^', 'Background': '*'}
    tsne = TSNE(n_components=2, verbose=1, perplexity=40)
    tsne_results = tsne.fit_transform(data_subset_train)
    df_subset = pd.DataFrame()

    df_subset['tsne-2d-one'] = tsne_results[:, 0]
    df_subset['tsne-2d-two'] = tsne_results[:, 1]
    df_subset['labels'] = [ship_types[x] for x in labels_train]

    plt.figure(figsize=(16, 10))
    sns.scatterplot(x='tsne-2d-one', y = 'tsne-2d-two',
        palette=sns.color_palette("Set2"),
        data=df_subset,
        markers=markers,
        style='labels',
        hue='labels',
        legend="full",
        alpha=1,
        sizes=(20, 200)
    )
    plt.savefig('Plots/{}_train.svg'.format(save_base_name))
    plt.show()

    tsne_results = tsne.fit_transform(data_subset_test)
    df_subset = pd.DataFrame()

    df_subset['tsne-2d-one'] = tsne_results[:, 0]
    df_subset['tsne-2d-two'] = tsne_results[:, 1]
    df_subset['labels'] = [ship_types[x] for x in labels_test]

    plt.figure(figsize=(16, 10))
    cmap = sns.cubehelix_palette(as_cmap=True)
    sns.scatterplot(x='tsne-2d-one', y='tsne-2d-two',
                    palette=sns.color_palette("Set2"),
                    data=df_subset,
                    markers=markers,
                    style='labels',
                    hue='labels',
                    legend="full",
                    alpha=1,
                    sizes=(20, 200)
                    )
    plt.savefig('Plots/{}_test.svg'.format(save_base_name))
    plt.show()

def fit_linear_layer(samples, labels, output_size, save_path):
    from torch.utils.data import TensorDataset, DataLoader
    print('SIZE: ', samples.shape)

    classifier = torch.nn.Linear(samples.shape[1], output_size)

    # Convert to PyTorch tensors
    X_tensor = torch.from_numpy(samples).float()
    y_tensor = torch.from_numpy(labels).long()

    # Create a dataset
    dataset = TensorDataset(X_tensor, y_tensor)

    # Create a DataLoader for batching
    batch_size = 32
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    import torch.optim as optim

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.Adam(classifier.parameters(), lr=0.001)

    num_epochs = 30
    prev_loss = torch.inf

    for epoch in range(num_epochs):
        for X_batch, y_batch in loader:
            optimizer.zero_grad()
            y_pred = classifier(X_batch)
            loss = criterion(y_pred, y_batch)
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item():.4f}")
        if loss.item() < prev_loss:

            torch.save(classifier.state_dict(), save_path)
            final_classifier = classifier
            final_classifier.eval()
    return final_classifier


def fit_attentive_layer(samples, labels, output_size, save_path):
    from torch.utils.data import TensorDataset, DataLoader
    print('SIZE: ', samples.shape)

    classifier = AttentionPoolingClassifier(samples.shape[1], output_size)

    # Convert to PyTorch tensors
    X_tensor = torch.from_numpy(samples).float()
    y_tensor = torch.from_numpy(labels).long()

    # Create a dataset
    dataset = TensorDataset(X_tensor, y_tensor)

    # Create a DataLoader for batching
    batch_size = 32
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    import torch.optim as optim

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.Adam(classifier.parameters(), lr=0.001)

    num_epochs = 30
    prev_loss = torch.inf

    for epoch in range(num_epochs):
        for X_batch, y_batch in loader:
            optimizer.zero_grad()
            y_pred = classifier(X_batch)
            loss = criterion(y_pred, y_batch)
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item():.4f}")
        if loss.item() < prev_loss:

            torch.save(classifier.state_dict(), save_path)
            final_classifier = classifier
            final_classifier.eval()
    return final_classifier

