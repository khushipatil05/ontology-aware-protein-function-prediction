"""Minimal Gene Ontology (OBO) parser + ancestor propagation. No external dependency."""
from collections import defaultdict

ASPECT_MAP = {"biological_process": "P", "molecular_function": "F", "cellular_component": "C"}
ROOTS = {"GO:0008150": "P", "GO:0003674": "F", "GO:0005575": "C"}


class GeneOntology:
    def __init__(self, obo_path, relations=("is_a", "part_of")):
        self.parents = defaultdict(set)   # term -> direct parents
        self.aspect = {}                  # term -> P/F/C
        self.name = {}
        self.alt = {}                     # alt_id -> primary id
        self.obsolete = set()
        self.relations = relations
        self._parse(obo_path)
        self._anc_cache = {}

    def _parse(self, path):
        cur, in_term = None, False
        with open(path, encoding="utf-8", errors="ignore") as f:
            for line in f:
                line = line.strip()
                if line == "[Term]":
                    cur, in_term = {"is_a": [], "part_of": [], "alt": []}, True
                    continue
                if line.startswith("[") and line != "[Term]":
                    self._commit(cur); cur, in_term = None, False
                    continue
                if not in_term or not line:
                    if not line and in_term:
                        self._commit(cur); cur, in_term = None, False
                    continue
                if line.startswith("id: "):
                    cur["id"] = line[4:]
                elif line.startswith("name: "):
                    cur["name"] = line[6:]
                elif line.startswith("namespace: "):
                    cur["ns"] = line[11:]
                elif line.startswith("alt_id: "):
                    cur["alt"].append(line[8:])
                elif line.startswith("is_a: "):
                    cur["is_a"].append(line[6:].split()[0])
                elif line.startswith("relationship: part_of "):
                    cur["part_of"].append(line.split()[2])
                elif line.startswith("is_obsolete: true"):
                    cur["obs"] = True
        self._commit(cur)

    def _commit(self, t):
        if not t or "id" not in t:
            return
        tid = t["id"]
        if t.get("obs"):
            self.obsolete.add(tid); return
        self.name[tid] = t.get("name", "")
        self.aspect[tid] = ASPECT_MAP.get(t.get("ns"), None)
        for a in t["alt"]:
            self.alt[a] = tid
        for rel in self.relations:
            for p in t.get(rel, []):
                self.parents[tid].add(p)

    def canonical(self, go_id):
        """Map alt ids to primary ids; return None for obsolete/unknown terms."""
        go_id = self.alt.get(go_id, go_id)
        return go_id if go_id in self.name else None

    def ancestors(self, go_id):
        """All ancestors (excluding itself)."""
        if go_id in self._anc_cache:
            return self._anc_cache[go_id]
        out, stack = set(), list(self.parents.get(go_id, ()))
        while stack:
            p = stack.pop()
            if p in out:
                continue
            out.add(p)
            stack.extend(self.parents.get(p, ()))
        self._anc_cache[go_id] = out
        return out

    def propagate(self, terms):
        """True-path rule: a protein annotated with a term is annotated with all its ancestors."""
        full = set(terms)
        for t in terms:
            full |= self.ancestors(t)
        return full
