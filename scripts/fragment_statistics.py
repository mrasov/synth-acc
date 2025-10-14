from collections import defaultdict
from rdkit import Chem
import pandas as pd
from tqdm import tqdm


L0_SMARTS = "[a:1]:1:a:a:a:a1"

def build_hierarchical_tree(df_library):
    """
    Строит дерево SMARTS-подструктур 
    от базового паттерна [a:1]:1:a:a:a:a1 к дочерним на 1 уровне и далее
    {l1: {l2: {l3: {l4: [l5, ...]}}}}
    
    Сохраняет кеш SMARTS -> Mol
    """
    # сборка дерева подструктур
    tree = {}
    for _, r in df_library.iterrows():
        l1 = r['layer1_smarts']
        l2 = r['layer2_smarts']
        l3 = r['layer3_smarts']
        l4 = r['layer4_smarts']
        l5 = r['layer5_smarts']
        tree.setdefault(l1, {}).setdefault(l2, {}).setdefault(l3, {}).setdefault(l4, []).append(l5)

    # подготовка к сборке кешей SMARTS -> Mol
    def unique_patterns(col):
        return set(df_library[col].dropna().astype(str).unique())

    set_l1 = unique_patterns('layer1_smarts')
    set_l2 = unique_patterns('layer2_smarts')
    set_l3 = unique_patterns('layer3_smarts')
    set_l4 = unique_patterns('layer4_smarts')
    set_l5 = unique_patterns('layer5_smarts')

    caches = {'L0': None, 'l1': {}, 'l2': {}, 'l3': {}, 'l4': {}, 'l5': {}}
    try:
        caches['L0'] = Chem.MolFromSmarts(L0_SMARTS)
    except Exception:
        caches['L0'] = None

    def compile_set(sset):
        c = {}
        for s in sset:
            if s is None or (isinstance(s, str) and s.strip() == ""):
                c[s] = None
                continue
            try:
                m = Chem.MolFromSmarts(s)
            except Exception:
                m = None
            c[s] = m
        return c

    caches['l1'] = compile_set(set_l1)
    caches['l2'] = compile_set(set_l2)
    caches['l3'] = compile_set(set_l3)
    caches['l4'] = compile_set(set_l4)
    caches['l5'] = compile_set(set_l5)

    return tree, caches


def collect_fragment_statistics(molecules_df, df_library):
    """
    Подсчет статистик по уровням L0...L5.
    """
    tree, caches = build_hierarchical_tree(df_library)
    pattern_counts = {
        'L1': defaultdict(int),
        'L2': defaultdict(int),
        'L3': defaultdict(int),
        'L4': defaultdict(int),
        'L5': defaultdict(int)
    }
    l0_count = 0

    matches = None
    matches = {'L0': set(), 'L1': defaultdict(set), 'L2': defaultdict(set),
                'L3': defaultdict(set), 'L4': defaultdict(set), 'L5': defaultdict(set)}

    iterrows = tqdm(molecules_df.itertuples(index=False), total=len(molecules_df), desc="Scanning molecules set")

    for row in iterrows:
        mol_id = getattr(row, 'molecule_chembl_id')
        smi = getattr(row, 'canonical_smiles')

        try:
            mol = Chem.MolFromSmiles(smi)
        except Exception:
            continue
        if mol is None:
            continue

        # проверка вхождения базового паттерна
        if caches['L0'] is not None and mol.HasSubstructMatch(caches['L0']):
            l0_count += 1
            matches['L0'].add(mol_id)
        else:
            continue

        seen_L1 = set()
        seen_L2 = set()
        seen_L3 = set()
        seen_L4 = set()
        seen_L5 = set()

        # первый уровень
        for l1, d2 in tree.items():
            pat1 = caches['l1'].get(l1)
            if pat1 is None:
                continue
            if not mol.HasSubstructMatch(pat1):
                continue
            if l1 not in seen_L1:
                pattern_counts['L1'][l1] += 1
                seen_L1.add(l1)
                matches['L1'][l1].add(mol_id)
        
        # второй уровень
            for l2, d3 in d2.items():
                pat2 = caches['l2'].get(l2)
                if pat2 is None:
                    continue
                if not mol.HasSubstructMatch(pat2):
                    continue
                if l2 not in seen_L2:
                    pattern_counts['L2'][l2] += 1
                    seen_L2.add(l2)
                    matches['L2'][l2].add(mol_id)

        # третий уровень
                for l3, d4 in d3.items():
                    pat3 = caches['l3'].get(l3)
                    if pat3 is None:
                        continue
                    if not mol.HasSubstructMatch(pat3):
                        continue
                    if l3 not in seen_L3:
                        pattern_counts['L3'][l3] += 1
                        seen_L3.add(l3)
                        matches['L3'][l3].add(mol_id)

        # четвертый уровень
                    for l4, l5s in d4.items():
                        pat4 = caches['l4'].get(l4)
                        if pat4 is None:
                            continue
                        if not mol.HasSubstructMatch(pat4):
                            continue
                        if l4 not in seen_L4:
                            pattern_counts['L4'][l4] += 1
                            seen_L4.add(l4)
                            matches['L4'][l4].add(mol_id)

        # пятый уровень, листья дерева
                        for l5 in l5s:
                            pat5 = caches['l5'].get(l5)
                            if pat5 is None:
                                continue
                            if l5 in seen_L5:
                                continue
                            if mol.HasSubstructMatch(pat5):
                                pattern_counts['L5'][l5] += 1
                                seen_L5.add(l5)
                                matches['L5'][l5].add(mol_id)

    # подготовка результатов по каждому уровню L1...L5
    def dict_to_df(d):
        df = pd.DataFrame([{'pattern': k, 'count': v} for k, v in d.items()])
        if df.shape[0] == 0:
            return df
        return df.sort_values('count', ascending=False).reset_index(drop=True)

    df_L1 = dict_to_df(pattern_counts['L1'])
    df_L2 = dict_to_df(pattern_counts['L2'])
    df_L3 = dict_to_df(pattern_counts['L3'])
    df_L4 = dict_to_df(pattern_counts['L4'])
    df_L5 = dict_to_df(pattern_counts['L5'])

    for lvl, df_lvl in [('L1', df_L1), ('L2', df_L2), ('L3', df_L3), ('L4', df_L4), ('L5', df_L5)]:
        n_ge1 = (df_lvl['count'] >= 1).sum() if not df_lvl.empty else 0
        n_ge10 = (df_lvl['count'] >= 10).sum() if not df_lvl.empty else 0
        print(f"{lvl}: patterns seen >=1: {n_ge1}, >=10: {n_ge10}, total patterns in lib: {len(df_library[f'layer{lvl[-1]}_smarts'].unique()) if lvl!='L0' else 'NA'}")

    print(f"L0: molecules containing L0 pattern: {l0_count}")

    per_level_counts = {'L0': l0_count, 'L1': df_L1, 'L2': df_L2, 'L3': df_L3, 'L4': df_L4, 'L5': df_L5}
    result = {
        'per_level_counts': per_level_counts,
        'pattern_counts': pattern_counts,
        'matches': matches
    }
    return result