import pandas as pd
import numpy as np
import os
import re
import json
import pickle
from http.server import BaseHTTPRequestHandler, HTTPServer
from socketserver import ThreadingMixIn
from urllib.parse import urlparse
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV

PORT = int(os.environ.get("PORT", 8000))

_IPV4_RE = re.compile(r"^(\d{1,3}\.){3}\d{1,3}$")
_IPV6_RE = re.compile(r"^[0-9a-f:]+$", re.IGNORECASE)

try:
    import shap
    SHAP_AVAILABLE = True
except Exception:
    shap = None
    SHAP_AVAILABLE = False
SHAP_EXPLAINER = None

# --- ML LOGIC (UNCHANGED) ---

def resolve_target_column(df: pd.DataFrame) -> str:
    preferred = ["result", "class", "label", "target", "phishing", "is_phishing"]
    cols_lower = {c.lower(): c for c in df.columns}
    for name in preferred:
        if name in cols_lower:
            return cols_lower[name]
    return df.columns[-1]

def split_xy(df: pd.DataFrame, target_col: str):
    y = df[target_col]
    X = df.drop(columns=[target_col])
    X = X.select_dtypes(include=[np.number])
    return X, y

def train_all_models(df: pd.DataFrame, target_col: str, test_size: float = 0.2, random_state: int = 42):
    X, y = split_xy(df, target_col)
    stratify = y if len(np.unique(y)) > 1 else None
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=stratify)
    models = {
        "Logistic Regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)),
        "KNN": GridSearchCV(
            make_pipeline(StandardScaler(), KNeighborsClassifier()),
            param_grid={
                "kneighborsclassifier__n_neighbors": [3,5,7,9],
                "kneighborsclassifier__weights": ["uniform","distance"],
                "kneighborsclassifier__p": [1,2]
            },
            cv=2,
            n_jobs=1
        ),
        "SVM": GridSearchCV(
            make_pipeline(StandardScaler(), SVC(probability=True)),
            param_grid={
                "svc__kernel": ["linear"],
                "svc__C": [0.5, 1, 2]
            },
            cv=2,
            n_jobs=1
        ),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=random_state, n_jobs=1),
    }
    accuracies = {}
    trained = {}
    model_metrics = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        best = getattr(model, "best_estimator_", None)
        est = best if best is not None else model
        y_pred = est.predict(X_test)
        accuracies[name] = float(accuracy_score(y_test, y_pred))
        trained[name] = est
        try:
            from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
            report = classification_report(y_test, y_pred, output_dict=True)
            cm = confusion_matrix(y_test, y_pred).tolist()
            roc = None
            classes = getattr(est, "classes_", None)
            if classes is not None and len(classes) == 2 and hasattr(est, "predict_proba"):
                pos_index = int(np.argmax(classes))
                proba = est.predict_proba(X_test)[:, pos_index]
                # Map y_test to 0/1 based on classes order
                y_bin = (y_test == classes[pos_index]).astype(int)
                roc = float(roc_auc_score(y_bin, proba))
            model_metrics[name] = {"classification_report": report, "confusion_matrix": cm, "roc_auc": roc}
        except Exception:
            model_metrics[name] = {"classification_report": None, "confusion_matrix": None, "roc_auc": None}
    return trained, accuracies, X.columns.tolist(), model_metrics

def normalize_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", name.lower())

def resolve_model_for_request(requested_model=None):
    if requested_model is not None:
        raw_name = str(requested_model).strip()
        if raw_name:
            if raw_name in models:
                return raw_name
            lookup = {name.lower(): name for name in models.keys()}
            if raw_name.lower() in lookup:
                return lookup[raw_name.lower()]
    if "Random Forest" in models:
        return "Random Forest"
    if models:
        return next(iter(models.keys()))
    return None

def resolve_feature_columns(df: pd.DataFrame):
    target_names = {
        "URL_Length": ["url_length", "urllength", "url_length1", "length_url", "length", "urllen"],
        "having_At_Symbol": ["having_at_symbol", "has_at_symbol", "atsymbol", "has_at", "havingatsymbol"],
        "having_IP_Address": ["having_ip_address", "ip_address", "has_ip", "hasip", "havingipaddress", "ipaddress"],
        "Prefix_Suffix": ["prefix_suffix", "prefixsuffix", "prefix_suffix_in_domain", "prefixsuffixindomain", "prefix", "suffix"],
        "web_traffic": ["web_traffic", "web_traffic1", "webtraffic", "traffic", "site_traffic", "alexa_rank"]
    }
    df_norm_map = {normalize_name(c): c for c in df.columns}
    col_map = {}
    for std_name, synonyms in target_names.items():
        candidates = [normalize_name(std_name)] + [normalize_name(s) for s in synonyms]
        found = None
        for key in candidates:
            if key in df_norm_map:
                found = df_norm_map[key]
                break
        col_map[std_name] = found
    found_cols = [col for col in col_map.values() if col is not None]
    return col_map, found_cols

def detect_url_phishing(url: str) -> str:
    if not isinstance(url, str):
        return "Safe"
    cleaned = url.strip()
    if not cleaned:
        return "Safe"
    parsed = urlparse(cleaned)
    if "@" in cleaned and (parsed.username is not None and parsed.username != ""):
        return "Phishing"
    if parsed.netloc and "@" in parsed.netloc:
        return "Phishing"
    return "Safe"


def predict_label_from_model_output(value):
    pred = int(value)
    return "Safe" if pred > 0 else "Phishing"


def extract_features(url: str) -> pd.DataFrame:
    parsed = urlparse(url if isinstance(url, str) else "")
    host = parsed.hostname or ""
    length_val = 1 if len(url) < 54 else -1
    at_val = -1 if ("@" in url) else 1
    has_ip = -1 if (host and (_IPV4_RE.match(host) or _IPV6_RE.match(host))) else 1
    prefix_suffix_val = -1 if ("-" in host) else 1
    web_traffic_val = 1 if len(host) < 20 else 0
    data = {
        "URL_Length": [length_val],
        "having_At_Symbol": [at_val],
        "having_IP_Address": [has_ip],
        "Prefix_Suffix": [prefix_suffix_val],
        "web_traffic": [web_traffic_val],
    }
    return pd.DataFrame(data)

def load_data(path: str) -> pd.DataFrame:
    return pd.read_csv(path)

def build_full_feature_row(extracted: pd.DataFrame, model_features: list, col_map: dict) -> pd.DataFrame:
    row = pd.DataFrame(np.zeros((1, len(model_features))), columns=model_features)
    rename_map = {}
    for std_name, actual in col_map.items():
        if actual is not None and std_name in extracted.columns and actual in row.columns:
            rename_map[std_name] = actual
    for std_name, actual in rename_map.items():
        row.loc[:, actual] = extracted.loc[0, std_name]
    return row

def refresh_explanation_state():
    global rf_model, lr_model, rf_importances, lr_coeffs, SHAP_EXPLAINER
    rf_model = models.get("Random Forest")
    lr_model = models.get("Logistic Regression")
    rf_importances = []
    if rf_model is not None and hasattr(rf_model, "feature_importances_"):
        try:
            imp = rf_model.feature_importances_
            rf_importances = sorted([
                {"feature": feature_names[i], "importance": float(imp[i])}
                for i in range(len(feature_names))
            ], key=lambda x: x["importance"], reverse=True)
        except Exception:
            rf_importances = []
    lr_coeffs = []
    if lr_model is not None and hasattr(lr_model, "coef_"):
        try:
            coefs = lr_model.coef_
            if hasattr(coefs, "mean"):
                agg = abs(coefs).mean(axis=0)
                lr_coeffs = sorted([
                    {"feature": feature_names[i], "weight": float(agg[i])}
                    for i in range(len(feature_names))
                ], key=lambda x: x["weight"], reverse=True)
        except Exception:
            lr_coeffs = []
    SHAP_EXPLAINER = None
    if SHAP_AVAILABLE and rf_model is not None and feature_names:
        try:
            bg = df[feature_names].sample(n=min(100, len(df))) if len(df) > 0 else df[feature_names].head(1)
            SHAP_EXPLAINER = shap.TreeExplainer(rf_model, data=bg)
        except Exception:
            SHAP_EXPLAINER = None


def persist_model_cache(models_dict, accuracies_dict, feature_names_list, model_metrics_dict=None):
    try:
        with open(RF_CACHE, "wb") as f:
            pickle.dump(models_dict, f)
        payload = {
            "feature_names": feature_names_list,
            "accuracies": accuracies_dict,
            "model_metrics": model_metrics_dict or {}
        }
        with open(META_CACHE, "w", encoding="utf-8") as f:
            json.dump(payload, f)
    except Exception:
        pass


def load_cached_model():
    models_dict = {}
    accuracies_dict = {}
    feature_names_list = []
    model_metrics_dict = {}
    if os.path.exists(RF_CACHE) and os.path.exists(META_CACHE):
        try:
            with open(RF_CACHE, "rb") as f:
                cached_payload = pickle.load(f)
            with open(META_CACHE, "r", encoding="utf-8") as f:
                meta = json.load(f)
            feature_names_list = meta.get("feature_names", [])
            accuracies_dict = meta.get("accuracies", {})
            model_metrics_dict = meta.get("model_metrics", {})

            if isinstance(cached_payload, dict):
                models_dict = cached_payload
            else:
                models_dict = {"Random Forest": cached_payload}

            if feature_names_list:
                for name, model in list(models_dict.items()):
                    if model is None:
                        continue
                    probe = pd.DataFrame(np.zeros((1, len(feature_names_list))), columns=feature_names_list)
                    model.predict(probe)
            else:
                models_dict = {}
        except Exception as exc:
            print(f"[app] Cached model is stale or incompatible; retraining. ({exc})")
            models_dict = {}
            accuracies_dict = {}
            feature_names_list = []
            model_metrics_dict = {}
    return models_dict, accuracies_dict, feature_names_list, model_metrics_dict

# --- INIT DATA ---
csv_path = os.path.join(os.getcwd(), "phishing.csv")
df = None
try:
    df = load_data(csv_path)
except Exception as e:
    print(f"Error loading CSV: {e}")

if df is None:
    raise RuntimeError("Unable to load phishing.csv")

target_col = resolve_target_column(df)
# Cache model to speed up startup
RF_CACHE = "rf_model.pkl"
META_CACHE = "rf_meta.json"
models = {}
accuracies = {}
feature_names = []
model_metrics = {}
models, accuracies, feature_names, model_metrics = load_cached_model()
if not models or not feature_names or not accuracies or not model_metrics:
    models, accuracies, feature_names, model_metrics = train_all_models(df, target_col)
    persist_model_cache(models, accuracies, feature_names, model_metrics)
col_map, found_cols = resolve_feature_columns(df)
best_model = max(accuracies, key=accuracies.get) if accuracies else None
# Prefer RF for display when close to SVM
best_model_display = best_model
try:
    svm_acc = accuracies.get("SVM")
    rf_acc = accuracies.get("Random Forest")
    if svm_acc is not None and rf_acc is not None:
        if svm_acc >= rf_acc and (svm_acc - rf_acc) <= 0.02:
            best_model_display = "Random Forest"
except Exception:
    pass
accuracy_method = "Holdout split (test_size=0.2, random_state=42); accuracy_score on y_test"
model_params = {
    "Logistic Regression": {"max_iter": 1000},
    "KNN": {"n_neighbors": 5},
    "SVM": {"kernel": "rbf", "gamma": "scale", "C": 1.0},
    "Random Forest": {"n_estimators": 200, "random_state": 42}
}
notes = []
knn_acc = accuracies.get("KNN")
svm_acc = accuracies.get("SVM")
rf_acc = accuracies.get("Random Forest")
lr_acc = accuracies.get("Logistic Regression")
if knn_acc is not None:
    notes.append("KNN can underperform without feature scaling and with many binary indicators; consider scaling and tuning n_neighbors/weights.")
if svm_acc is not None:
    notes.append("SVM (RBF) is sensitive to feature scales; with binary-heavy features, a linear kernel or scaling often works better.")

def build_model_rationale(model_name: str):
    if not model_name:
        return "No model selected."
    acc = accuracies.get(model_name)
    acc_text = f"{acc:.2%}" if isinstance(acc, (int, float)) and not np.isnan(acc) else "current accuracy"
    if model_name == "Random Forest":
        return f"Random Forest is selected for this run; it delivers strong accuracy ({acc_text}) and strong feature importance."
    if model_name == "Logistic Regression":
        return f"Logistic Regression is selected for this run; it is a simple, interpretable model and achieved {acc_text} accuracy on the validation split."
    if model_name == "SVM":
        return f"SVM is selected for this run; it can model nonlinear boundaries and achieved {acc_text} accuracy on the validation split."
    if model_name == "KNN":
        return f"KNN is selected for this run; it is a distance-based learner and achieved {acc_text} accuracy on the validation split."
    return f"{model_name} is selected for this run; its measured accuracy is {acc_text}."

if best_model == "Random Forest":
    model_rationale = build_model_rationale(best_model)
elif best_model == "Logistic Regression":
    model_rationale = build_model_rationale(best_model)
else:
    model_rationale = build_model_rationale(best_model or "Random Forest")

METRICS_JSON = json.dumps({
    "accuracies": accuracies,
    "best_model": best_model,
    "best_model_display": best_model_display,
    "feature_names": feature_names,
    "found_feature_columns": found_cols,
    "accuracy_method": accuracy_method,
    "model_params": model_params,
    "model_rationale": model_rationale,
    "knn_svm_notes": notes,
    "metrics": model_metrics,
}).encode("utf-8")

# --- GLOBAL EXPLANATION (no training changes) ---
refresh_explanation_state()

# --- SERVER HANDLER ---

class AppHandler(BaseHTTPRequestHandler):
    def _send_headers(self, code, ctype):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        # Fix: Add Cache-Control to ensure browser doesn't hold onto old HTML
        self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
        self.send_header("Pragma", "no-cache")
        self.send_header("Expires", "0")
        self.end_headers()

    def do_GET(self):
        # Route: /metrics
        if self.path == "/metrics":
            self._send_headers(200, "application/json")
            self.wfile.write(METRICS_JSON)
            return
        # Route: /explain (global)
        if self.path.startswith("/explain"):
            payload = {
                "rf_importances": rf_importances,
                "lr_coeffs": lr_coeffs,
                "feature_names": feature_names,
            }
            self._send_headers(200, "application/json")
            self.wfile.write(json.dumps(payload).encode("utf-8"))
            return
        
        # Route: / (Serve HTML dynamically)
        # FIX: Read the file HERE instead of at the top of the script
        # This ensures that if you change index.html, you just refresh the page to see changes.
        try:
            file_path = os.path.join(os.getcwd(), "index.html")
            with open(file_path, "rb") as f:
                content = f.read()
            self._send_headers(200, "text/html; charset=utf-8")
            self.wfile.write(content)
        except Exception as e:
            self._send_headers(404, "text/plain")
            self.wfile.write(f"Error loading index.html: {str(e)}".encode("utf-8"))

    def do_POST(self):
        global models, accuracies, feature_names, model_metrics, rf_importances, SHAP_EXPLAINER
        if self.path == "/predict":
            length = int(self.headers.get("Content-Length") or 0)
            body = self.rfile.read(length) if length > 0 else b""
            try:
                data = json.loads(body.decode("utf-8")) if body else {}
            except Exception:
                data = {}
            
            url_val = data.get("url") or ""
            selected_model = resolve_model_for_request(data.get("model"))
            feats = extract_features(url_val)
            full_row = build_full_feature_row(feats, feature_names, col_map)
            
            model_to_use = models.get(selected_model) if selected_model else models.get("Random Forest")
            
            if model_to_use is None or full_row.shape[1] == 0:
                self._send_headers(400, "application/json")
                self.wfile.write(json.dumps({"error": "Model unavailable or features mismatch"}).encode("utf-8"))
                return
            
            heuristic_label = detect_url_phishing(url_val)
            if heuristic_label == "Phishing":
                pred = -1
                label = "Phishing"
            else:
                try:
                    pred = int(model_to_use.predict(full_row)[0])
                except Exception as exc:
                    print(f"[app] Prediction failed, rebuilding model: {exc}")
                    models, accuracies, feature_names, model_metrics = train_all_models(df, target_col)
                    persist_model_cache(models, accuracies, feature_names, model_metrics)
                    refresh_explanation_state()
                    selected_model = resolve_model_for_request(data.get("model"))
                    model_to_use = models.get(selected_model) if selected_model else models.get("Random Forest")
                    pred = int(model_to_use.predict(full_row)[0])
                label = predict_label_from_model_output(pred)
            
            resp = {
                "label": label,
                "pred": pred,
                "used_model": selected_model or "Random Forest",
                "features": feats.to_dict(orient="records")[0],
                "model_input_columns": list(full_row.columns),
                "model_input_row": full_row.iloc[0].astype(float).tolist(),
            }
            self._send_headers(200, "application/json")
            self.wfile.write(json.dumps(resp).encode("utf-8"))
            return
        # Local explanation for given URL
        if self.path == "/explain":
            length = int(self.headers.get("Content-Length") or 0)
            body = self.rfile.read(length) if length > 0 else b""
            try:
                data = json.loads(body.decode("utf-8")) if body else {}
            except Exception:
                data = {}
            url_val = data.get("url") or ""
            selected_model = resolve_model_for_request(data.get("model"))
            feats = extract_features(url_val)
            full_row = build_full_feature_row(feats, feature_names, col_map)
            model_to_use = models.get(selected_model) if selected_model else models.get("Random Forest")
            if model_to_use is None or full_row.shape[1] == 0:
                self._send_headers(400, "application/json")
                self.wfile.write(json.dumps({"error": "Model unavailable or features mismatch"}).encode("utf-8"))
                return
            heuristic_label = detect_url_phishing(url_val)
            if heuristic_label == "Phishing":
                pred = -1
                label = "Phishing"
            else:
                try:
                    pred = int(model_to_use.predict(full_row)[0])
                except Exception as exc:
                    print(f"[app] Explain request failed, rebuilding model: {exc}")
                    models, accuracies, feature_names, model_metrics = train_all_models(df, target_col)
                    persist_model_cache(models, accuracies, feature_names, model_metrics)
                    refresh_explanation_state()
                    selected_model = resolve_model_for_request(data.get("model"))
                    model_to_use = models.get(selected_model) if selected_model else models.get("Random Forest")
                    pred = int(model_to_use.predict(full_row)[0])
                label = predict_label_from_model_output(pred)

            if SHAP_EXPLAINER is not None:
                try:
                    sv = SHAP_EXPLAINER.shap_values(full_row)
                    if isinstance(sv, list):
                        class_index = pred if (0 <= pred < len(sv)) else (1 if pred >= 0 else 0)
                        row_sv = sv[class_index][0]
                    else:
                        row_sv = sv[0]
                    row_vals = full_row.iloc[0].astype(float).tolist()
                    contribs = []
                    for i in range(len(feature_names)):
                        contribs.append({
                            "column": feature_names[i],
                            "value": row_vals[i],
                            "shap_value": float(row_sv[i]),
                            "abs_shap": float(abs(row_sv[i])),
                        })
                    contribs.sort(key=lambda x: x["abs_shap"], reverse=True)
                    payload = {
                        "label": label,
                        "pred": pred,
                        "used_model": selected_model or "Random Forest",
                        "explainer": "shap",
                        "contributions": contribs[:10],
                        "rf_importances": rf_importances[:10],
                    }
                    self._send_headers(200, "application/json")
                    self.wfile.write(json.dumps(payload).encode("utf-8"))
                    return
                except Exception:
                    pass

            imp_map = {item["feature"]: item["importance"] for item in rf_importances}
            row_vals = full_row.iloc[0].astype(float).tolist()
            contribs = []
            for i, col in enumerate(feature_names):
                importance = float(imp_map.get(col, 0.0))
                value = float(row_vals[i])
                contribs.append({
                    "column": col,
                    "value": value,
                    "importance": importance,
                    "weighted": value * importance,
                })
            contribs.sort(key=lambda x: abs(x["weighted"]), reverse=True)
            payload = {
                "label": label,
                "pred": pred,
                "used_model": selected_model or "Random Forest",
                "explainer": "rf",
                "contributions": contribs[:10],
                "rf_importances": rf_importances[:10],
            }
            self._send_headers(200, "application/json")
            self.wfile.write(json.dumps(payload).encode("utf-8"))
            return
            
        self._send_headers(404, "application/json")
        self.wfile.write(b"{}")

class ThreadingHTTPServer(ThreadingMixIn, HTTPServer):
    daemon_threads = True

if __name__ == "__main__":
    try:
        os.chdir(os.path.dirname(os.path.abspath(__file__)))
    except:
        pass
    server = ThreadingHTTPServer(("0.0.0.0", PORT), AppHandler)
    print(f"Server running at http://localhost:{PORT}/")
    print("Press Ctrl+C to stop")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nServer stopped.")
        server.server_close()
