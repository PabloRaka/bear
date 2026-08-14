import pytest
from data.dataset import DATASET_REGISTRY, get_verified_shards


def test_registry_contains_four_stages():
    expected_stages = {"pretrain", "cpt", "sft", "safety"}
    assert set(DATASET_REGISTRY.keys()) == expected_stages


def test_stage_weights_sum_to_one():
    for stage_name, stage_info in DATASET_REGISTRY.items():
        total_weight = sum(dom["weight"] for dom in stage_info["domains"].values())
        assert abs(total_weight - 1.0) < 1e-5, f"Stage {stage_name} weights do not sum to 1.0: {total_weight}"


def test_pretrain_proportions():
    pretrain = DATASET_REGISTRY["pretrain"]["domains"]
    # Bahasa (Umum): 50% (ID 25% + EN 25%)
    assert pretrain["bahasa_umum_id"]["weight"] + pretrain["bahasa_umum_en"]["weight"] == 0.50
    # Code (Python, TS, JS, PHP, C++, C, C#): 20%
    code_weight = (
        pretrain["code_python"]["weight"]
        + pretrain["code_typescript"]["weight"]
        + pretrain["code_javascript"]["weight"]
        + pretrain["code_php"]["weight"]
        + pretrain["code_cpp"]["weight"]
        + pretrain["code_c"]["weight"]
        + pretrain["code_csharp"]["weight"]
    )
    assert abs(code_weight - 0.20) < 1e-5
    # Matematika: 20%
    assert pretrain["matematika"]["weight"] == 0.20
    # Terminal: 10%
    assert pretrain["terminal"]["weight"] == 0.10


def test_cpt_proportions():
    cpt = DATASET_REGISTRY["cpt"]["domains"]
    # Bahasa (Umum): 10%
    assert cpt["bahasa_umum"]["weight"] == 0.10
    # Code: 40%
    assert cpt["code_multilang"]["weight"] == 0.40
    # Matematika: 30%
    assert cpt["matematika"]["weight"] == 0.30
    # Terminal: 20%
    assert cpt["terminal"]["weight"] == 0.20


def test_sft_proportions():
    sft = DATASET_REGISTRY["sft"]["domains"]
    # Percakapan Bahasa ID & EN: 80% (ID 40% + EN 40%)
    assert sft["percakapan_id"]["weight"] + sft["percakapan_en"]["weight"] == 0.80
    # Instruksi: 10%
    assert sft["instruksi"]["weight"] == 0.10
    # Tooling calls: 10%
    assert sft["tooling_calls"]["weight"] == 0.10


def test_safety_proportions():
    safety = DATASET_REGISTRY["safety"]["domains"]
    assert safety["safety_pku"]["weight"] == 0.50
    assert safety["safety_anthropic"]["weight"] == 0.30
    assert safety["safety_jailbreak"]["weight"] == 0.20
    total_safety = sum(dom["weight"] for dom in safety.values())
    assert abs(total_safety - 1.0) < 1e-5


def test_split_dataset_file(tmp_path):
    from data.dataset import split_dataset_file
    import pyarrow as pa
    import pyarrow.parquet as pq
    import json

    # 1. Test Parquet Split (100 rows -> 95 train / 5 val)
    parquet_path = tmp_path / "sample.parquet"
    data = {"text": [f"Sample row {i}" for i in range(100)]}
    table = pa.Table.from_pydict(data)
    pq.write_table(table, parquet_path)

    train_dir = tmp_path / "train"
    val_dir = tmp_path / "val"
    train_file, val_file = split_dataset_file(parquet_path, train_dir, val_dir, val_ratio=0.05)

    train_table = pq.read_table(train_file)
    val_table = pq.read_table(val_file)
    assert len(train_table) == 95
    assert len(val_table) == 5

    # 2. Test JSONL Split (100 lines -> 95 train / 5 val)
    jsonl_path = tmp_path / "sample.jsonl"
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for i in range(100):
            f.write(json.dumps({"id": i, "content": f"code snippet {i}"}) + "\n")

    train_j, val_j = split_dataset_file(jsonl_path, train_dir, val_dir, val_ratio=0.05)
    with open(train_j, "r", encoding="utf-8") as f:
        assert len(f.readlines()) == 95
    with open(val_j, "r", encoding="utf-8") as f:
        assert len(f.readlines()) == 5


def test_deterministic_shuffle_reproducibility(tmp_path):
    from data.dataset import split_dataset_file
    import pyarrow as pa
    import pyarrow.parquet as pq

    parquet_path = tmp_path / "shuffle_source.parquet"
    data = {"id": list(range(200)), "content": [f"doc_{i}" for i in range(200)]}
    table = pa.Table.from_pydict(data)
    pq.write_table(table, parquet_path)

    # Split with seed=42
    dir_a_train = tmp_path / "a_train"
    dir_a_val = tmp_path / "a_val"
    train_a, val_a = split_dataset_file(parquet_path, dir_a_train, dir_a_val, val_ratio=0.05, seed=42, shuffle=True)

    # Split with seed=42 again (must be 100% identical)
    dir_b_train = tmp_path / "b_train"
    dir_b_val = tmp_path / "b_val"
    train_b, val_b = split_dataset_file(parquet_path, dir_b_train, dir_b_val, val_ratio=0.05, seed=42, shuffle=True)

    table_a = pq.read_table(train_a)
    table_b = pq.read_table(train_b)
    assert table_a["id"].to_pylist() == table_b["id"].to_pylist()
    # Check that it is actually shuffled (not just sequential 0..189)
    assert table_a["id"].to_pylist() != list(range(190))

    # Split with seed=999 (must differ from seed=42)
    dir_c_train = tmp_path / "c_train"
    dir_c_val = tmp_path / "c_val"
    train_c, val_c = split_dataset_file(parquet_path, dir_c_train, dir_c_val, val_ratio=0.05, seed=999, shuffle=True)
    table_c = pq.read_table(train_c)
    assert table_a["id"].to_pylist() != table_c["id"].to_pylist()


def test_domain_folder_structure_mapping():
    from data.dataset import DATASET_REGISTRY
    from pathlib import Path

    base_dir = Path("storage/dataset")
    # Verify every stage and domain has unique isolated paths
    all_domain_paths = set()
    for stage_key, stage_info in DATASET_REGISTRY.items():
        for domain_key in stage_info["domains"].keys():
            expected_path = base_dir / stage_key / domain_key
            assert expected_path not in all_domain_paths, f"Duplicate folder path: {expected_path}"
            all_domain_paths.add(expected_path)
    
    # Exactly 22 distinct isolated domain folders (11 Pretrain + 4 CPT + 4 SFT + 3 Safety)
    assert len(all_domain_paths) == 22
