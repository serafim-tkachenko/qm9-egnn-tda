from src.data.qm9_data import load_qm9
from src.data.tda_features import TDACache, TDAConfig

def main():
    # Here we save everything TDA related to the Google Drive as it's a heavy operation and reruning is costly
    ds = load_qm9("/content/drive/MyDrive/topo-egnn-qm9/data/qm9")
    cfg = TDAConfig(cache_dir="/content/drive/MyDrive/topo-egnn-qm9/tda_cache", n_bins=16, max_homology_dim=1)
    tda = TDACache(cfg)

    print("TDA feature dim:", tda.feature_dim())
    tda.build_for_dataset(ds)
    print("Done. Cached to:", cfg.cache_dir)

if __name__ == "__main__":
    main()
