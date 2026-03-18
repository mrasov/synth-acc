import pandas as pd
import re
from collections import defaultdict
from tqdm import tqdm


class LibraryOptimizer:
    def __init__(self, df_library, l4_counts=None, threshold=10):
        self.df_library = df_library
        self.l4_counts = l4_counts if l4_counts is not None else defaultdict(int)
        self.threshold = threshold

    # вспомогательные функции
    def _extract_subs(self, l3_smarts, l4_smarts):
        escaped_l3 = re.escape(l3_smarts)
        pattern = "^" + escaped_l3.replace(r"\(\*\)", r"\((.*?)\)") + "$"
        match = re.fullmatch(pattern, l4_smarts)
        return match.groups() if match else None

    def _merge_substituent_strings(self, sub_list):
        if not sub_list:
            return ""
        if len(sub_list) == 1:
            return sub_list[0]

        parsed = []
        for s in sub_list:
            m = re.match(r"^([^\[]*)\[(.*)\]$", s)
            if m:
                parsed.append((m.group(1), m.group(2)))
            else:
                parsed.append(("", s.strip("[]")))

        base_bond = parsed[0][0]

        all_atoms = []
        for _, atoms in parsed:
            for a in atoms.split(","):
                if a not in all_atoms:
                    all_atoms.append(a)

        merged_inner = ",".join(all_atoms)
        return f"{base_bond}[{merged_inner}]"

    def _compress_tuples(self, tuples):
        if not tuples:
            return []

        num_cols = len(tuples[0])
        current_tuples = set(tuples)

        for j in range(num_cols):
            groups = defaultdict(list)
            for t in current_tuples:
                key = tuple(t[i] for i in range(num_cols) if i != j)
                groups[key].append(t[j])

            next_tuples = set()
            for key, vars_list in groups.items():
                merged_val = (
                    self._merge_substituent_strings(vars_list)
                    if len(vars_list) > 1
                    else vars_list[0]
                )
                new_t = list(key)
                new_t.insert(j, merged_val)
                next_tuples.add(tuple(new_t))

            current_tuples = next_tuples

        return list(current_tuples)

    def _inject_subs(self, l3_smarts, sub_tuple):
        parts = l3_smarts.split("(*)")
        res = parts[0]
        for i, sub in enumerate(sub_tuple):
            res += f"({sub})" + parts[i + 1]
        return res

    # основной генератор
    def optimize(self):
        new_rows = []
        cols_to_group = [
            "layer0_smarts",
            "layer1_smarts",
            "layer2_smarts",
            "layer3_smarts",
        ]

        grouped = self.df_library.groupby(cols_to_group)

        for keys, group_df in tqdm(grouped, desc="Optimizing L4 structures"):
            l0, l1, l2, l3_smarts = keys

            rare_l4 = []
            freq_l4 = []

            for l4 in group_df["layer4_smarts"]:
                if self.l4_counts.get(l4, 0) < self.threshold:
                    rare_l4.append(l4)
                else:
                    freq_l4.append(l4)

            for l4 in freq_l4:
                new_rows.append((l0, l1, l2, l3_smarts, l4))

            if rare_l4:
                extracted_tuples = [self._extract_subs(l3_smarts, l4) for l4 in rare_l4]
                extracted_tuples = [t for t in extracted_tuples if t is not None]

                if extracted_tuples:
                    compressed_tuples = self._compress_tuples(extracted_tuples)
                    for ct in compressed_tuples:
                        new_l4 = self._inject_subs(l3_smarts, ct)
                        new_rows.append((l0, l1, l2, l3_smarts, new_l4))

        optimized_df = pd.DataFrame(new_rows, columns=self.df_library.columns)
        return optimized_df.drop_duplicates().reset_index(drop=True)
