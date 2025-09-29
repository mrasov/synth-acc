import pandas as pd
from itertools import product, combinations
from tqdm import tqdm


# базовый SMARTS
L0_SMARTS = "[a:1]1:a:a:a:a:1"

# атомные токены (таблица 4)
C_STAR = "[#6](*)"
N_NOSTAR = "[#7+0]"
N_STAR = "[#7+0](*)"
O_ATOM = "[#8]"
S_ATOM = "[#16]"
H_TOKEN = "[#1]"

# каталог заместителей (таблица 3)
SUBSTITUENT_CATALOG = {
    "C1": "[CH3,CH2]", "C2": "[CH,CH0]", "C3": "[c]",
    "N1": "[NH2,NH,N+]", "N2": "[N+0H0]", "N3": "[n+0]",
    "O1": "[O]", "S1": "[S]", "F": "[F]", "Cl": "[Cl]",
}
APPLICABLE_SUBSTITUENTS = list(SUBSTITUENT_CATALOG.keys())

#  вспомогательные функции
def canonical_reflect_tuple(seq):
    """канонизация (отражение последовательности)"""
    return min(tuple(seq), tuple(reversed(seq)))

def canonical_necklace(seq):
    """полная канонизация (вращения + отражения)"""
    n = len(seq)
    reps = []
    current_fwd = list(seq)
    current_rev = list(reversed(seq))
    for _ in range(n):
        reps.append(tuple(current_fwd))
        reps.append(tuple(current_rev))
        current_fwd = current_fwd[1:] + current_fwd[:1]
        current_rev = current_rev[1:] + current_rev[:1]
    return min(reps)

def build_smarts(pattern):
    """сборка SMARTS"""
    if not pattern: return ""
    if pattern[0] == 'a': return L0_SMARTS

    def add_ring_label(atom_str):
        if ":1" in atom_str: return atom_str
        return atom_str.replace("]", ":1]", 1) if "]" in atom_str else f"[{atom_str}:1]"

    parts = []
    for i, p_str in enumerate(pattern):
        atom, tag = (p_str.split('|', 1) + [''])[:2]
        
        current_part = atom.replace("(*)", "") if tag else atom
        if tag == "H":
            current_part += f"({H_TOKEN})"
        elif tag and tag != "*":
            current_part += f"({SUBSTITUENT_CATALOG[tag]})"
        if i == 0:
            current_part = add_ring_label(current_part)
        parts.append(current_part)
    return parts[0] + "1:" + ":".join(parts[1:]) + ":1"


#  основной пайплайн генерации
def generate_hierarchical_library():
    """генерирует с нуля иерархическую библиотеку, сохраняет таблицу с соответствующей иерархией подструктур"""

    # слой 1: скелеты C, N
    choices = [C_STAR, N_NOSTAR]
    layer1_skeletons = list({canonical_reflect_tuple(s): s for s in product(choices, repeat=4) if s.count(N_NOSTAR) <= 2}.values())

    # слой 3: ядра с конкретным гетероатомом
    centers = [C_STAR, N_STAR, O_ATOM, S_ATOM]
    layer2_cores = [((skel, (center,) + skel)) for skel in layer1_skeletons for center in centers]

    # слой 3: маски (H/*)
    layer3_masks_h = []
    for skel, core in tqdm(layer2_cores, desc="Generating L3 (Masks)"):
        free_pos = [i for i, token in enumerate(core) if "(*)" in token]
        for k in (2, 3):
            if len(free_pos) < k: continue
            for indices in combinations(free_pos, k):
                mask = tuple(f"{atom}|*" if i in indices else f"{atom}|H" if "(*)" in atom else f"{atom}|" for i, atom in enumerate(core))
                layer3_masks_h.append((skel, core, mask))
    
    unique_masks = {canonical_necklace(m): (s, c, m) for s, c, m in layer3_masks_h}
    
    # слой 4: с заместителями из таблицы 3
    final_hierarchy = []
    for skel, core, mask in tqdm(unique_masks.values(), desc="Generating L4 (Substituents)"):
        star_pos = [i for i, p in enumerate(mask) if p.endswith("|*")]
        choices = [APPLICABLE_SUBSTITUENTS] * len(star_pos)
        for labels in product(*choices):
            final_pat = list(mask)
            for i, pos in enumerate(star_pos):
                final_pat[pos] = f"{mask[pos].split('|')[0]}|{labels[i]}"
            final_hierarchy.append((skel, core, mask, tuple(final_pat)))

    # сборка датафрейма с кешированием и дедупликацией
    print("Building final hierarchical table with caching and deduplication...")
    final_data = {}
    parent_smarts_cache = {}
    
    for skeleton_pat, core_pat, mask_pat, final_pat in tqdm(final_hierarchy, desc="Building DataFrame"):
        canon_final_pat = canonical_necklace(final_pat)
        if canon_final_pat not in final_data:
            # используем кеширование для родительских SMARTS
            if mask_pat not in parent_smarts_cache:
                parent_smarts_cache[mask_pat] = {
                    "layer1_smarts": L0_SMARTS,
                    "layer2_smarts": build_smarts(('a',) + skeleton_pat),
                    "layer3_smarts": build_smarts(core_pat),
                    "layer4_smarts": build_smarts(mask_pat),
                }
            
            final_data[canon_final_pat] = {
                **parent_smarts_cache[mask_pat],
                "layer5_smarts": build_smarts(final_pat)
            }

    df = pd.DataFrame.from_dict(final_data, orient='index')
    return df.sort_values(by=list(df.columns)).reset_index(drop=True)

if __name__ == "__main__":
    df = generate_hierarchical_library()
    print(f"сгенерировано {len(df)} уникальных подструктур.")
    
    output_filename = "smarts_hierarchical_library.csv"
    df.to_csv(output_filename, index=False)
    print(f"таблица сохранена в файл: {output_filename}")