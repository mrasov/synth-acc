import pandas as pd
from itertools import product, combinations
from tqdm import tqdm


# заместители и ключевые элементы (таблицы 3, 4)
C_STAR = "[#6](*)"
N_NOSTAR = "[#7+0]"
N_STAR = "[#7+0](*)"
O_ATOM = "[#8]"
S_ATOM = "[#16]"
H_TOKEN = "[#1]"
L0_SMARTS = "[a:1]1:a:a:a:a:1"

SUBSTITUENT_CATALOG = {
    "C1": "[CH3,CH2]", "C2": "[CH,CH0]", "C3": "[c]", "N1": "[NH2,NH,N+]",
    "N2": "[N+0H0]", "N3": "[n+0]", "O1": "[O]", "S1": "[S]", "F": "[F]", "Cl": "[Cl]"
}
APPLICABLE_SUBSTITUENTS = list(SUBSTITUENT_CATALOG.keys())


#  вспомогательные функции
def canonical_reflect_tuple(seq):
    return min(tuple(seq), tuple(reversed(seq)))

def canonical_necklace(seq):
    """
    Канонизация с поворотами
    """
    n = len(seq)
    reps = {tuple(seq), tuple(reversed(seq))}
    fwd = list(seq)
    rev = list(reversed(seq))
    for _ in range(n - 1):
        fwd = fwd[1:] + fwd[:1]
        rev = rev[1:] + rev[:1]
        reps.add(tuple(fwd))
        reps.add(tuple(rev))
    return min(reps)

def build_smarts(pattern, level):
    """
    Сборка SMARTS
    """
    if level == 1: return L0_SMARTS

    def add_ring_label(atom_str):
        if ":1" in atom_str: return atom_str
        return atom_str.replace("]", ":1]", 1) if "]" in atom_str else f"[{atom_str}:1]"

    parts = []
    for i, p_str in enumerate(pattern):
        atom, tag = (p_str.split('|', 1) + [''])[:2]
        
        current_part = atom
        if tag == "H":
            current_part = atom.replace("(*)", f"({H_TOKEN})")
        elif tag and tag != "*":
            current_part = atom.replace("(*)", f"({SUBSTITUENT_CATALOG[tag]})")
        
        if i == 0:
            if level == 2:
                current_part = "[a:1]"
            else:
                current_part = add_ring_label(current_part)

        parts.append(current_part)

    return parts[0] + "1:" + ":".join(parts[1:]) + ":1"

#  основной генератор
def generate_hierarchical_library():

    # L1: скелеты
    choices = [C_STAR, N_NOSTAR]
    layer1_skeletons = list({canonical_reflect_tuple(s): s for s in product(choices, repeat=4) if s.count(N_NOSTAR) <= 2}.values())

    # L2: ядра
    centers = [C_STAR, N_STAR, O_ATOM, S_ATOM]
    layer2_cores_h = [((skel, (center,) + skel)) for skel in layer1_skeletons for center in centers]

    # L3: маски
    layer3_masks_h = []
    for skel, core in tqdm(layer2_cores_h, desc="Generating L3 (Masks)"):
        free_pos = [i for i, token in enumerate(core) if "(*)" in token]
        for k in (2, 3):
            if len(free_pos) < k: continue
            for indices in combinations(free_pos, k):
                mask = tuple(f"{atom}|*" if i in indices else f"{atom}|H" if "(*)" in atom else f"{atom}|" for i, atom in enumerate(core))
                layer3_masks_h.append((skel, core, mask))
    
    unique_masks = {canonical_necklace(m): (s, c, m) for s, c, m in layer3_masks_h}
    
    # L4: добавление заместителей
    final_hierarchy = []
    for skel, core, mask in tqdm(unique_masks.values(), desc="Generating L4 (Substituents)"):
        star_pos = [i for i, p in enumerate(mask) if p.endswith("|*")]
        for labels in product(APPLICABLE_SUBSTITUENTS, repeat=len(star_pos)):
            final_pat = list(mask)
            for i, pos in enumerate(star_pos):
                final_pat[pos] = f"{mask[pos].split('|')[0]}|{labels[i]}"
            final_hierarchy.append((skel, core, mask, tuple(final_pat)))

    # сборка таблицы
    final_data = {}
    parent_smarts_cache = {}
    
    for skeleton_pat, core_pat, mask_pat, final_pat in tqdm(final_hierarchy, desc="Building DataFrame"):
        canon_final_pat = canonical_necklace(final_pat)
        if canon_final_pat not in final_data:
            key_cache = (skeleton_pat, core_pat, mask_pat)
            if key_cache not in parent_smarts_cache:
                parent_smarts_cache[key_cache] = {
                    "layer1_smarts": build_smarts(None, level=1),
                    "layer2_smarts": build_smarts(('a',) + skeleton_pat, level=2),
                    "layer3_smarts": build_smarts(core_pat, level=3),
                    "layer4_smarts": build_smarts(mask_pat, level=4),
                }
            
            final_data[canon_final_pat] = {
                **parent_smarts_cache[key_cache],
                "layer5_smarts": build_smarts(final_pat, level=5)
            }

    df = pd.DataFrame.from_dict(final_data, orient='index')
    return df.sort_values(by=list(df.columns)).reset_index(drop=True)

if __name__ == "__main__":
    df = generate_hierarchical_library()
    print(f"собрана таблица из {len(df)} уникальных подструктур.")
    
    print("первые 5 строк:")
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    print(df.head())

    output_filename = "smarts_hierarchical_library.csv"
    df.to_csv(output_filename, index=False)
