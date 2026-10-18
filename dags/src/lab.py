import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from kneed import KneeLocator
import pickle
import os
import base64

def load_data(filename: str = "file.csv"):
    """
    Loads data from a CSV file in the data folder, serializes it, and returns the serialized data.
    Returns:
        str: Base64-encoded serialized data (JSON-safe).
    """
    print(f"Loading data/{filename}")
    df = pd.read_csv(os.path.join(os.path.dirname(__file__), "../data", filename))
    serialized_data = pickle.dumps(df)                    # bytes
    return base64.b64encode(serialized_data).decode("ascii")  # JSON-safe string

# Columns the clustering model needs, and limits the data must meet
REQUIRED_COLUMNS = ["BALANCE", "PURCHASES", "CREDIT_LIMIT"]
MIN_ROWS = 100
MAX_MISSING_SHARE = 0.05

def validate_data(data_b64: str):
    """
    Checks the loaded data before it is used for training.
    Returns a list of problems found (an empty list means the data passed).
    """
    # decode -> bytes -> DataFrame
    data_bytes = base64.b64decode(data_b64)
    df = pickle.loads(data_bytes)

    problems = []
    missing_cols = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing_cols:
        problems.append(f"Missing required columns: {missing_cols}")
        return problems  # the remaining checks need these columns

    if len(df) < MIN_ROWS:
        problems.append(f"Only {len(df)} rows; need at least {MIN_ROWS}")

    for col in REQUIRED_COLUMNS:
        missing_share = df[col].isna().mean()
        if missing_share > MAX_MISSING_SHARE:
            problems.append(f"{col} is {missing_share:.1%} missing; limit is {MAX_MISSING_SHARE:.0%}")
        negatives = int((df[col] < 0).sum())
        if negatives:
            problems.append(f"{col} has {negatives} negative values")

    print(f"Validation checked {len(df)} rows: {len(problems)} problem(s) found")
    for problem in problems:
        print(f"  - {problem}")
    return problems

def data_preprocessing(data_b64: str):
    """
    Deserializes base64-encoded pickled data, performs preprocessing,
    and returns base64-encoded pickled clustered data.
    """
    # decode -> bytes -> DataFrame
    data_bytes = base64.b64decode(data_b64)
    df = pickle.loads(data_bytes)

    df = df.dropna()
    clustering_data = df[["BALANCE", "PURCHASES", "CREDIT_LIMIT"]]

    min_max_scaler = MinMaxScaler()
    clustering_data_minmax = min_max_scaler.fit_transform(clustering_data)

    # bytes -> base64 string for XCom
    clustering_serialized_data = pickle.dumps(clustering_data_minmax)
    return base64.b64encode(clustering_serialized_data).decode("ascii")


def build_save_model(data_b64: str, filename: str):
    """
    Builds a KMeans model on the preprocessed data and saves it.
    Returns the SSE list (JSON-serializable).
    """
    # decode -> bytes -> numpy array
    data_bytes = base64.b64decode(data_b64)
    df = pickle.loads(data_bytes)

    kmeans_kwargs = {"init": "random", "n_init": 10, "max_iter": 300, "random_state": 42}
    sse = []
    for k in range(1, 50):
        kmeans = KMeans(n_clusters=k, **kmeans_kwargs)
        kmeans.fit(df)
        sse.append(kmeans.inertia_)

    # NOTE: This saves the last-fitted model (k=49), matching your original intent.
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "model")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, filename)
    with open(output_path, "wb") as f:
        pickle.dump(kmeans, f)

    return sse  # list is JSON-safe


def load_model_elbow(filename: str, sse: list):
    """
    Loads the saved model and uses the elbow method to report k.
    Returns the first prediction (as a plain int) for test.csv.
    """
    # load the saved (last-fitted) model
    output_path = os.path.join(os.path.dirname(__file__), "../model", filename)
    loaded_model = pickle.load(open(output_path, "rb"))

    # elbow for information/logging
    kl = KneeLocator(range(1, 50), sse, curve="convex", direction="decreasing")
    print(f"Optimal no. of clusters: {kl.elbow}")

    # predict on raw test data (matches your original code)
    df = pd.read_csv(os.path.join(os.path.dirname(__file__), "../data/test.csv"))
    pred = loaded_model.predict(df)[0]

    # ensure JSON-safe return
    try:
        return int(pred)
    except Exception:
        # if not numeric, still return a JSON-friendly version
        return pred.item() if hasattr(pred, "item") else pred
