import os
import json
import shutil

# Daftar lokasi yang ingin dicek 
possible_paths = [
    "C:\Program Files (x86)\Microsoft\Temp\EU7F0B.tmp\tempfile_8364.json",
    "C:\ProgramData\Intel\GCC\gcc_svc_log_2025-02-06.json",
    "C:\Program Files\Google\Chrome\Application\135.0.7049.85\default_apps",
]

# Folder tujuan
target_folder = "C:/Users/Admin/AppData/Roaming/Code/User/snippets"
target_filename = "python.json"
target_path = os.path.join(target_folder, target_filename)

os.makedirs(target_folder, exist_ok=True)

# Coba cari file yang tersedia
found = False
for path in possible_paths:
    if os.path.isfile(path):
        shutil.copy(path, target_path)
        print(f"File berhasil disalin dari {path} ke {target_path}")
        found = True
        break

if not found:
    data = {
        "mxcmdl": {
            "prefix": "mxcmdl",
            "body": [
                "class KNN:",
                "    def __init__(self, k: int):",
                "        self.k = k",
                "",
                "    def minkowski_distance(self, X, y, p=2):",
                "        return np.power(np.sum(np.abs(X - y) ** p), 1/p)",
                "",
                "    def fit(self, X, y):",
                "        self.X_train = X",
                "        self.y_train = y",
                "",
                "    def predict(self, X):",
                "        preds = [self._predict_process(x) for x in X]",
                "        return np.array(preds)",
                "",
                "    def _predict_process(self, X):",
                "        # Komputasi jarak antara x dan semua training set",
                "        distances = [self.minkowski_distance(X, x_train) for x_train in self.X_train]",
                "        # Urutkan berdasarkan jarak dan ambil indeks dari k tetangga terdekat",
                "        k_indices = np.argsort(distances)[:self.k]",
                "        # Ekstrak label training sampel dari knn",
                "        k_nearest_labels = [self.y_train[i] for i in k_indices]",
                "        # Mengambil class label paling sering (mayoritas)",
                "        return np.bincount(k_nearest_labels).argmax()"
            ],
        }
    }

    with open(target_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4)

    print(f"File tidak ditemukan, membuat file default di {target_path}")

