import numpy as np
from numpy.linalg import inv
import pandas as pd

# Files path setting (don't change this)
TRAIN_PATH = "inputs/training_dataset.csv"
TEST_PATH = "inputs/testing_dataset.csv"
SAVE_PATH = "outputs/result_3.csv"
SMALL_TEST_PATH = "inputs/(additional_small)testing_dataset.csv"

def save_result(preds: np.ndarray, weights: np.ndarray):
    """Save prediction and weights to a CSV file
    - `preds`: predicted values with shape (n_samples,) 
    - `weights`: model weights with shape (n_basis,)
    """
    max_length = max(len(preds), len(weights))
    result = np.full((max_length, 2), "", dtype=object)
    result[:len(preds), 0] = preds.astype(str)
    result[:len(weights), 1] = weights.astype(str)
    np.savetxt(SAVE_PATH, result, delimiter=",", fmt="%s")


# Load data
train_set = pd.read_csv(TRAIN_PATH, header=None).values.astype(np.float32)
# test_set = pd.read_csv(TEST_PATH, header=None).values.astype(np.float32)
small_test_set = pd.read_csv(SMALL_TEST_PATH, header=None).values.astype(np.float32)

X_train, t_train = train_set[:,0:2], train_set[:,2]
X_small_test, t_small_test = small_test_set[:,0:2], small_test_set[:,2]


# Step 1: k-means find mu and sigma in basis functions
def initialize_random_centroids(K, X):
    '''choose K centroids from 6000 data entries'''
    np.random.seed(44)
    m, n = np.shape(X) # (6000, 2)
    centroids = np.empty((K, n)) # 1 centroid (1, 2) => K centroids (K, 2)
    for i in range(K):
        centroids[i] = X[np.random.choice(m)]
    return centroids

def euclidean_distance(x1, x2):
    return np.linalg.norm(x1 - x2)

def closest_centroid(centroids, K, X):
    '''return the index of the closest centroid'''
    distances = np.empty(K)
    for i in range(K):
        distances[i] = euclidean_distance(centroids[i], X)
    return np.argmin(distances)

def create_clusters(centroids, K, X):
    '''return an array of cluster indices'''
    m, _ = np.shape(X)
    cluster_index = np.empty(m)
    for i in range(m):
        cluster_index[i] = closest_centroid(centroids, K, X[i])
    return cluster_index

def compute_means(cluster_index, K, X):
    '''compute and return the new centroids of clusters'''
    _, n = np.shape(X)
    centroids = np.empty((K, n))
    for i in range(K):
        points = X[cluster_index == i]
        if len(points) > 0:
            centroids[i] = np.mean(points, axis=0)
    return centroids

def run_kmeans(K, X, max_iterations=500, epsilon=1e-4):
    '''run kmeans to find Gaussian basis function mu'''
    centroids = initialize_random_centroids(K, X)
    for _ in range(max_iterations):
        clusters = create_clusters(centroids, K, X)
        prev_centroids = centroids.copy()
        centroids = compute_means(clusters, K, X)

        # break if centroids converges
        if np.linalg.norm(prev_centroids - centroids) < epsilon:
            break
    return centroids

def compute_sigma(centers):
    '''return average nearest neighbor distance'''
    # calculate distances between clusters
    # subtract using broadcasting (M * 1 * N) - (1 * M * N) -> (M * M * N)
    M, _ = np.shape(centers)
    x_diffs = np.abs(centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0])
    y_diffs = np.abs(centers[:, np.newaxis, 1] - centers[np.newaxis, :, 1]) 

    # ignore self-distances by setting to infinity
    np.fill_diagonal(x_diffs, np.inf)
    np.fill_diagonal(y_diffs, np.inf)

    # compute mean nearest neighbor distances
    sigma_x = np.mean(np.min(x_diffs, axis=1)) / 2 + 1e-5 
    sigma_y = np.mean(np.min(y_diffs, axis=1)) / 2 + 1e-5  

    return sigma_x, sigma_y

# Step 2: define basis functions and calculate Phi (from p.29)
def gaussian_bf(x, y, mu_x, mu_y, sigma_x, sigma_y):
    diff_x = np.clip(x - mu_x, -1e5, 1e5)
    diff_y = np.clip(y - mu_y, -1e5, 1e5)
    exponent = - ((diff_x**2) / (2 * sigma_x)) - ((diff_y**2) / (2 * sigma_y))
    exponent = np.clip(exponent, -500, None)
    return np.exp(exponent)

def laplace_bf(x, y, mu_x, mu_y, sigma_x, sigma_y):
    diff_x = np.clip(x - mu_x, -1e5, 1e5)
    diff_y = np.clip(y - mu_y, -1e5, 1e5)
    exponent = - np.abs(diff_x) / sigma_x - np.abs(diff_y) / sigma_y
    exponent = np.clip(exponent, -500, None)
    return np.exp(exponent)

def sigmoid_bf(x, y, mu_x, mu_y, sigma_x, sigma_y):
    diff_x = np.clip(x - mu_x, -1e5, 1e5)
    diff_y = np.clip(y - mu_y, -1e5, 1e5)
    exponent = - ((diff_x) / sigma_x) - ((diff_y) / sigma_y)
    exponent = np.clip(exponent, -500, None)
    return 1 / (1 + np.exp(exponent))

def compute_Phi(X, centers, sigma_x, sigma_y):
    N, _ = np.shape(X)
    M, _ = np.shape(centers)
    Phi = np.zeros((N, M + 1))

    Phi[:, 0] = 1 # bias term
    for i, (x, y) in enumerate(X): 
        for j, (mu_x, mu_y) in enumerate(centers): 
            Phi[i, j + 1] = gaussian_bf(x, y, mu_x, mu_y, sigma_x, sigma_y)
    return Phi


# Step 3: Bayesian linear regression p.35
def W_Bayesian(alpha, m_0, sigma_x, sigma_y, Phi, t):
    '''Bayesian linear regression formula'''
    sigma = (sigma_x + sigma_y) / 2
    beta = 1 / sigma

    S_0 = alpha * np.eye(M + 1)
    S_N_inv = inv(S_0) + beta * (Phi.T @ Phi)
    S_N = inv(S_N_inv)
    m_N = S_N @ (inv(S_0) @ m_0 + beta * (Phi.T @ t[:, None]))
    return m_N


# Step 4: s-fold cross validation and mse
def mean_square_error(y, t):
    squared_errors = []
    N, _ = np.shape(t)
    for i in range(N):
        squared_errors.append((t[i] - y[i])**2)
    return np.mean(squared_errors)

def model_train(X_train, t_train, centers, sigma_x, sigma_y, alpha):
    '''W calculated using training set with maximum likelihood'''
    m_0 = np.zeros((M + 1, 1))
    Phi_train = compute_Phi(X_train, centers, sigma_x, sigma_y)
    return W_Bayesian(alpha, m_0, sigma_x, sigma_y, Phi_train, t_train)

def model_pred(X_val, W, centers, sigma_x, sigma_y):
    '''predict y'''
    Phi_val = compute_Phi(X_val, centers, sigma_x, sigma_y)
    pred = Phi_val @ W
    return np.maximum(pred, 0)

def s_fold_cross_validation(X, t, S, centers, sigma_x, sigma_y, alpha):
    '''return average validation scores, optimal weights, and best predictions'''
    np.random.seed(42)
    n, _ = np.shape(X)  # (6000, 2)
    indices = np.random.permutation(n)  # shuffle indices
    folds = np.array_split(indices, S)  # split into S folds

    val_scores = []
    W_list = []
    t_pred_list = []

    for i in range(S):
        # ith split as validation set, other as training sets
        val_set_indices = folds[i]
        train_set_indices = np.concatenate([folds[j] for j in range(S) if j != i])

        X_train, t_train = X[train_set_indices], t[train_set_indices]
        X_val, t_val = X[val_set_indices], t[val_set_indices]

        W = model_train(X_train, t_train, centers, sigma_x, sigma_y, alpha)
        t_pred = model_pred(X_val, W, centers, sigma_x, sigma_y)

        mse = mean_square_error(t_val, t_pred)
        val_scores.append(mse)
        W_list.append(W)
        t_pred_list.append(t_pred)
    
    best_idx = np.argmin(val_scores)
    return np.mean(val_scores), W_list[best_idx], t_pred_list[best_idx]


M = 505 # number of clusters, also number of basis functions
S = 5 # number of folds (s-fold cross validation)
alpha = 0.015 

# Load data
train_set = pd.read_csv(TRAIN_PATH, header=None).values.astype(np.float32)
test_set = pd.read_csv(TEST_PATH, header=None).values.astype(np.float32)
s_test_set = pd.read_csv(SMALL_TEST_PATH, header=None).values.astype(np.float32)

X_train, t_train = train_set[:,0:2], train_set[:,2]
X_test = test_set[:,0:2]
X_s_test, t_s_test = s_test_set[:,0:2], s_test_set[:,2]

centers = run_kmeans(M, X_train)
sigma_x, sigma_y = compute_sigma(centers)
Phi = compute_Phi(X_train, centers, sigma_x, sigma_y)

avg_mse, W, t_pred = s_fold_cross_validation(X_train, t_train, S, centers, sigma_x, sigma_y, alpha)
print(avg_mse)

t_test_pred = model_pred(X_test, W, centers, sigma_x, sigma_y)
t_test_pred = t_test_pred.ravel()
W = W.ravel()
save_result(t_test_pred, W)