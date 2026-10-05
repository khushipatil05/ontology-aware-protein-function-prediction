"""Download the Gene Ontology file (go-basic.obo) -> data/raw/go-basic.obo

Usage:  python src/download_go.py
Tries several mirrors. If all fail (firewall / college network), download it manually
in a browser and place it in data/raw/ -- see the README.
"""
import sys
import urllib.request
from config import OBO_PATH

MIRRORS = [
    "https://purl.obolibrary.org/obo/go/go-basic.obo",
    "http://purl.obolibrary.org/obo/go/go-basic.obo",
    "https://current.geneontology.org/ontology/go-basic.obo",
    "http://current.geneontology.org/ontology/go-basic.obo",
    "https://release.geneontology.org/2024-11-01/ontology/go-basic.obo",
    "https://raw.githubusercontent.com/geneontology/go-ontology/master/go-basic.obo",
]

def main():
    OBO_PATH.parent.mkdir(parents=True, exist_ok=True)
    for url in MIRRORS:
        try:
            print(f"Trying {url} ...")
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=60) as r, open(OBO_PATH, "wb") as f:
                while chunk := r.read(1 << 20):
                    f.write(chunk)
            head = OBO_PATH.read_text(errors="ignore")[:200]
            if "format-version" in head:
                print(f"OK -> {OBO_PATH}  ({OBO_PATH.stat().st_size/1e6:.1f} MB)")
                return
            print("Downloaded file does not look like an OBO file, trying next mirror")
        except Exception as e:
            print("  failed:", e)
    sys.exit("All mirrors failed. Download go-basic.obo manually (see README) and put it in data/raw/")

if __name__ == "__main__":
    main()
