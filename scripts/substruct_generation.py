import pandas as pd
from itertools import product, combinations
from tqdm import tqdm


# заместители и ключевые элементы (таблицы 3, 4)
SUBSTITUENT_CATALOG = {
    "C1": ('!=!@', '[CH3,CH2]'),  
    "C2": ('!=!@', '[CH,CH0]'),   
    "C3": ('!=!:', '[c]'),        
    "C4": ('!=@',  '[C]'),        
    "C5": (':',    '[c]'),        
    "N1": ('!=!@', '[NH2,NH,N+]'),
    "N2": ('!=!@', '[N+0H0]'),    
    "N3": ('!=!:', '[n+0]'),      
    "N4": (':',    '[n+0]'),      
    "N5": ('!=@',  '[N+0]'),      
    "O1": ('!=!@', '[O]'),        
    "O2": ('!=@',  '[O]'),        
    "O3": (':',    '[o]'),        
    "S1": ('!=!@', '[S]'),        
    "S2": ('!=@',  '[S]'),        
    "S3": (':',    '[s]'),        
    "F":  ('-',    '[F]'),        
    "Cl": ('-',    '[Cl]')        
}

ATOM_CONFIG = {
    "C_STAR":   {"smarts": "[#6]",   "has_sub": True},  
    "N_NOSTAR": {"smarts": "[#7+0]", "has_sub": False}, 
    "N_STAR":   {"smarts": "[#7+0]", "has_sub": True},  
    "O_ATOM":   {"smarts": "[#8]",   "has_sub": False}, 
    "S_ATOM":   {"smarts": "[#16]",  "has_sub": False}, 
}

BACKBONE_POOL = ["C_STAR", "N_NOSTAR"]  
CENTER_POOL   = ["N_STAR", "O_ATOM", "S_ATOM"] 
H_TOKEN = "[#1]"


class HeterocycleGenerator:
    def __init__(self, ring_size=5, min_subs=2, max_subs=3, max_n_skeleton=2, substituents=None):
        self.ring_size = ring_size
        self.backbone_len = ring_size - 1
        self.min_subs = min_subs
        self.max_subs = max_subs
        self.max_n_skeleton = max_n_skeleton
        self.substituents = substituents if substituents else list(SUBSTITUENT_CATALOG.keys())

    # вспомогательные функции
    @staticmethod
    def _canonical_necklace(seq):
        n = len(seq)
        reps = set()
        fwd = list(seq)
        rev = list(reversed(seq))
        for _ in range(n):
            reps.add(tuple(fwd))
            reps.add(tuple(rev))
            fwd = fwd[1:] + fwd[:1]
            rev = rev[1:] + rev[:1]
        return min(reps)

    def _align_to_center(self, sequence):
        # сдвиг цикла так, чтобы первым атомом был центр (S, O, N*)
        center_idx = -1
        for i, (atom, _) in enumerate(sequence):
            if atom in CENTER_POOL:
                center_idx = i
                break
        return list(sequence) if center_idx == -1 else sequence[center_idx:] + sequence[:center_idx]

    def _build_smarts(self, sequence, level):
        if level == 0:
            # L0: базовый SMARTS
            return f"[a:1]:1{':a' * (self.ring_size - 1)}:1"

        parts = []
        for i, (atom_key, sub_tag) in enumerate(sequence):
            config = ATOM_CONFIG.get(atom_key, {"smarts": atom_key, "has_sub": True})
            base_smarts = config['smarts']
            has_sub = config['has_sub']
            
            current_part = base_smarts

            # L1: скелеты 
            if level == 1:
                if i == 0:
                    current_part = "[a:1]"
                else:
                    current_part = f"{base_smarts}(*)" if has_sub else base_smarts
            
            # L2: ядра
            elif level == 2:
                current_part = f"{base_smarts}(*)" if has_sub else base_smarts
            
            # L3 (маски) и L4 (финальные)
            else:
                if sub_tag == '*':
                    current_part = f"{base_smarts}(*)"
                elif sub_tag == 'H' and has_sub:
                    current_part = f"{base_smarts}({H_TOKEN})"
                elif sub_tag in SUBSTITUENT_CATALOG:
                    bond, group = SUBSTITUENT_CATALOG[sub_tag]
                    current_part = f"{base_smarts}({bond}{group})"
            
            # Добавление метки [atom:1] к первому атому, если её нет
            if i == 0 and ":1" not in current_part:
                # Вставка метки перед закрывающей скобкой или в конец
                if "]" in current_part:
                    current_part = current_part.replace("]", ":1]", 1)
                else:
                    current_part = f"[{current_part}:1]"
            
            parts.append(current_part)

        # Сборка: [Start]:1 : [Next] : ... :1
        return parts[0] + ":1:" + ":".join(parts[1:]) + ":1"

    # основной генератор
    def generate_library(self):
        
        # L1: скелеты
        valid_backbones = []
        for skel in product(BACKBONE_POOL, repeat=self.backbone_len):
            if skel.count("N_NOSTAR") <= self.max_n_skeleton:
                valid_backbones.append(skel)
        
        # L2: ядра
        unique_cores = []
        seen_cores = set()
        for center in CENTER_POOL:
            for backbone in valid_backbones:
                raw_core = (center,) + backbone
                canon_core = self._canonical_necklace(raw_core)
                if canon_core not in seen_cores:
                    seen_cores.add(canon_core)
                    unique_cores.append(canon_core)

        # L3: маски
        unique_masks = []
        seen_masks_reps = set()
        
        for core in tqdm(unique_cores, desc="Generating L3 (Masks)"):
            free_pos = [i for i, atom in enumerate(core) if ATOM_CONFIG[atom]['has_sub']]
            for k in range(self.min_subs, self.max_subs + 1):
                if len(free_pos) < k: continue
                for indices in combinations(free_pos, k):
                    mask_struct = []
                    for i, atom in enumerate(core):
                        tag = '*' if i in indices else ('H' if ATOM_CONFIG[atom]['has_sub'] else None)
                        mask_struct.append((atom, tag))
                    
                    canon_mask = self._canonical_necklace(tuple(mask_struct))
                    if canon_mask not in seen_masks_reps:
                        seen_masks_reps.add(canon_mask)
                        unique_masks.append(canon_mask)

        # L4: добавление заместителей
        final_data = []
        seen_final_structs = set()
        
        for mask in tqdm(unique_masks, desc="Generating L4 (Substituents)"):
            mask_list = list(mask)
            
            # выравнивание и выбор направления для корректной генерации L1/L2
            aligned_mask = self._align_to_center(mask_list)
            aligned_mask_rev = [aligned_mask[0]] + list(reversed(aligned_mask[1:]))

            def get_base_smarts(seq):
                pure_seq = [(atom, None) for atom, _ in seq]
                return (self._build_smarts(pure_seq, level=1), 
                        self._build_smarts(pure_seq, level=2))

            l1_fwd, l2_fwd = get_base_smarts(aligned_mask)
            l1_rev, l2_rev = get_base_smarts(aligned_mask_rev)

            if l1_fwd < l1_rev:
                l1_smarts, l2_smarts = l1_fwd, l2_fwd
                target_seq = aligned_mask
            else:
                l1_smarts, l2_smarts = l1_rev, l2_rev
                target_seq = aligned_mask_rev

            l0_smarts = self._build_smarts(target_seq, level=0)
            l3_smarts = self._build_smarts(target_seq, level=3)

            star_pos = [i for i, (_, tag) in enumerate(target_seq) if tag == '*']
            
            for labels in product(self.substituents, repeat=len(star_pos)):
                final_struct_resolved = []
                iter_labels = iter(labels)
                for atom, tag in target_seq:
                    if tag == '*':
                        final_struct_resolved.append((atom, next(iter_labels)))
                    else:
                        final_struct_resolved.append((atom, tag))
                
                final_hash = self._canonical_necklace(tuple(final_struct_resolved))
                
                if final_hash not in seen_final_structs:
                    seen_final_structs.add(final_hash)
                    
                    final_data.append({
                        "layer0_smarts": l0_smarts,
                        "layer1_smarts": l1_smarts,
                        "layer2_smarts": l2_smarts,
                        "layer3_smarts": l3_smarts,
                        "layer4_smarts": self._build_smarts(final_struct_resolved, level=4)
                    })

        # сборка таблицы
        df = pd.DataFrame(final_data)
        if not df.empty:
            cols = sorted(list(df.columns))
            return df[cols].sort_values(by=cols).reset_index(drop=True)
        return df


if __name__ == "__main__":
    generator = HeterocycleGenerator(
        ring_size=5,
        min_subs=2, 
        max_subs=3,
        max_n_skeleton=2
    )
    
    df = generator.generate_library()
    print(f"собрана таблица из {len(df)} уникальных подструктур.")
    
    print("первые 5 строк:")
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    print(df.head())

    output_filename = "smarts_hierarchical_library.csv"
    df.to_csv(output_filename, index=False)