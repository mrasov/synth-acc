from collections import defaultdict
from rdkit import Chem
import pandas as pd
from tqdm.auto import tqdm
from joblib import Parallel, delayed
import multiprocessing as mp
import math


def build_hierarchical_tree(df_library):
    """
    Строит оптимизированное дерево для обхода:
    (smarts_str, rdkit_mol, children_list)
    """

    # сборка кешей SMARTS -> Mol
    unique_smarts = set()
    # теперь только до layer4
    cols = [
        "layer0_smarts",
        "layer1_smarts",
        "layer2_smarts",
        "layer3_smarts",
        "layer4_smarts",
    ]

    if "layer0_smarts" not in df_library.columns:
        raise ValueError("В библиотеке отсутствует столбец layer0_smarts")

    l0_str = df_library["layer0_smarts"].iloc[0]
    unique_smarts.add(l0_str)

    for c in cols[1:]:
        unique_smarts.update(df_library[c].dropna().unique())

    mol_cache = {}
    for s in tqdm(unique_smarts, desc="Pre-compiling SMARTS"):
        if not s:
            continue
        try:
            mol_cache[s] = Chem.MolFromSmarts(s)
        except Exception:
            mol_cache[s] = None

    # сборка структуры словарей (промежуточный шаг)
    # структура: L1 -> L2 -> L3 -> set(L4)
    tree_dict = {}
    for row in df_library.itertuples(index=False):
        d1 = tree_dict.setdefault(row.layer1_smarts, {})
        d2 = d1.setdefault(row.layer2_smarts, {})
        s3 = d2.setdefault(row.layer3_smarts, set())
        s3.add(row.layer4_smarts)  # L4 теперь лист

    # преобразование в список узлов (txt, mol, children)
    optimized_roots = []

    for l1_txt, l1_children in tree_dict.items():
        l1_mol = mol_cache.get(l1_txt)
        if l1_mol is None:
            continue

        l2_nodes = []
        for l2_txt, l2_children in l1_children.items():
            l2_mol = mol_cache.get(l2_txt)
            if l2_mol is None:
                continue

            l3_nodes = []
            for l3_txt, l4_set in l2_children.items():
                l3_mol = mol_cache.get(l3_txt)
                if l3_mol is None:
                    continue

                l4_nodes = []
                # L4 - последний уровень (листья)
                for l4_txt in l4_set:
                    l4_mol = mol_cache.get(l4_txt)
                    if l4_mol:
                        l4_nodes.append((l4_txt, l4_mol, None))  # детей нет

                l3_nodes.append((l3_txt, l3_mol, l4_nodes))
            l2_nodes.append((l2_txt, l2_mol, l3_nodes))
        optimized_roots.append((l1_txt, l1_mol, l2_nodes))

    return mol_cache.get(l0_str), optimized_roots


def _worker_process_chunk(chunk_dicts, l0_pat, search_tree, store_matches):
    local_l0_count = 0
    local_pattern_counts = {
        "L1": defaultdict(int), "L2": defaultdict(int),
        "L3": defaultdict(int), "L4": defaultdict(int),
    }

    local_matches = None
    if store_matches:
        local_matches = {
            "L0": set(), "L1": defaultdict(set),
            "L2": defaultdict(set), "L3": defaultdict(set), "L4": defaultdict(set),
        }

    for row in chunk_dicts:
        try:
            mol_id = row.get("molecule_chembl_id")
            smi = row.get("canonical_smiles")
            if not isinstance(smi, str): continue
            mol = Chem.MolFromSmiles(smi)
            mol = Chem.AddHs(mol)
        except Exception: continue

        if mol is None: continue
        if not mol.HasSubstructMatch(l0_pat): continue

        local_l0_count += 1
        if store_matches: local_matches["L0"].add(mol_id)

        for l1_txt, l1_mol, l2_list in search_tree:
            if not mol.HasSubstructMatch(l1_mol): continue
            local_pattern_counts["L1"][l1_txt] += 1
            if store_matches: local_matches["L1"][l1_txt].add(mol_id)

            for l2_txt, l2_mol, l3_list in l2_list:
                if not mol.HasSubstructMatch(l2_mol): continue
                local_pattern_counts["L2"][l2_txt] += 1
                if store_matches: local_matches["L2"][l2_txt].add(mol_id)

                for l3_txt, l3_mol, l4_list in l3_list:
                    if not mol.HasSubstructMatch(l3_mol): continue
                    local_pattern_counts["L3"][l3_txt] += 1
                    if store_matches: local_matches["L3"][l3_txt].add(mol_id)

                    for l4_txt, l4_mol, _ in l4_list:
                        if mol.HasSubstructMatch(l4_mol):
                            local_pattern_counts["L4"][l4_txt] += 1
                            if store_matches: local_matches["L4"][l4_txt].add(mol_id)

    return local_l0_count, local_pattern_counts, local_matches, len(chunk_dicts)


def collect_fragment_statistics(molecules_df, df_library, store_matches=False):
    """
    Иерархический подсчет частот фрагментов (L0 -> L4).
    """

    # подготовка дерева поиска
    l0_pat, search_tree = build_hierarchical_tree(df_library)
    if l0_pat is None:
        raise ValueError("Ошибка компиляции L0 SMARTS")

    pattern_counts = {
        "L1": defaultdict(int),
        "L2": defaultdict(int),
        "L3": defaultdict(int),
        "L4": defaultdict(int),
    }
    l0_count = 0

    matches = None
    if store_matches:
        matches = {
            "L0": set(),
            "L1": defaultdict(set),
            "L2": defaultdict(set),
            "L3": defaultdict(set),
            "L4": defaultdict(set),
        }

    # основной цикл сканирования
    if num_workers is None:
        num_workers = mp.cpu_count()
    
    num_chunks = min(len(molecules_df), num_workers * 4)
    chunk_data = []
    if num_chunks > 0:
        chunk_size = math.ceil(len(molecules_df) / num_chunks)
        cols_to_keep = [c for c in ["molecule_chembl_id", "canonical_smiles"] if c in molecules_df.columns]
        
        for i in range(0, len(molecules_df), chunk_size):
            chunk = molecules_df[cols_to_keep].iloc[i : i + chunk_size]
            chunk_data.append(chunk.to_dict('records'))

    if chunk_data:
        results = Parallel(n_jobs=num_workers)(
            delayed(_worker_process_chunk)(
                chunk, l0_pat, search_tree, store_matches
            ) for chunk in tqdm(chunk_data, desc=f"Sending chunks to {num_workers} processes")
        )
        
        for loc_l0, loc_pat, loc_mat, _ in results:
            l0_count += loc_l0
            for lvl in ["L1", "L2", "L3", "L4"]:
                for pat, count in loc_pat[lvl].items():
                    pattern_counts[lvl][pat] += count
                    
            if store_matches:
                matches["L0"].update(loc_mat["L0"])
                for lvl in ["L1", "L2", "L3", "L4"]:
                    for pat, ids in loc_mat[lvl].items():
                        matches[lvl][pat].update(ids)


    def dict_to_df(d):
        if not d:
            return pd.DataFrame(columns=["pattern", "count"])
        return (
            pd.DataFrame(list(d.items()), columns=["pattern", "count"])
            .sort_values("count", ascending=False)
            .reset_index(drop=True)
        )

    result_dfs = {k: dict_to_df(v) for k, v in pattern_counts.items()}

    print(f"L0 Matches: {l0_count}")
    for lvl in ["L1", "L2", "L3", "L4"]:
        found = len(result_dfs[lvl])
        total = df_library[f"layer{lvl[-1]}_smarts"].nunique()
        print(f"{lvl}: found {found} / {total} patterns")

    per_level_counts = {"L0": l0_count, **result_dfs}

    return {"per_level_counts": per_level_counts, "matches": matches}
