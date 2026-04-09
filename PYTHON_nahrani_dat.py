import google.generativeai as genai
from pinecone import Pinecone
import csv
import time
import uuid  

# ============================================
# 1. KONFIGURACE
# ============================================

PINECONE_API_KEY = "{pinecone_API_KEY}"
GOOGLE_API_KEY = "{embedding_API_KEY}" 
INDEX_NAME = "{nazev_databaze}" 
CSV_FILE = "{umisteni_CSV_FILE}"
# ============================================
# 2. INICIALIZACE
# ============================================

print("🔧 Inicializace...")
try:
    pc = Pinecone(api_key=PINECONE_API_KEY)
    index = pc.Index(INDEX_NAME)
    genai.configure(api_key=GOOGLE_API_KEY)
except Exception as e:
    print(f"❌ Chyba při připojování k API: {e}")
    exit()

# ============================================
# 3. NAČTENÍ CSV
# ============================================

products = []

COLUMN_MAPPING = {
    "id": "id",
    "name": "nazev",
    "price": "cena",
    "url": "link",
    "imageUrl": "fotka",
    "description": "popis",
    "availability": "dostupnost",
    "sizes": "velikost",
    "category": "kategorie",
     
}

print(f"📂 Načítám CSV: {CSV_FILE}")

def load_products(delimiter):
    """Pomocná funkce pro zkoušení oddělovače"""
    loaded_items = []

with open(CSV_FILE, 'r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f, delimiter=delimiter)
        
        if len(loaded_items) == 0:
            print(f"   🔍 Zkouším oddělovač '{delimiter}'. Vidím sloupce: {reader.fieldnames}")

        for i, row in enumerate(reader):
            product = {}
            for internal_key, csv_col in COLUMN_MAPPING.items():
                # Získáme hodnotu sloupce, pokud existuje
                val = row.get(csv_col, row.get(internal_key, ""))
                if val:
                    product[internal_key] = val.strip()
            
            # Generování ID pokud chybí
            if not product.get('id'):
                product['id'] = f"prod_{i}_{uuid.uuid4().hex[:8]}"

            # Musí mít aspoň jméno, jinak přeskakujeme
            if product.get('name'):
                # Normalizace kategorie (vše na malá písmena)
                if 'category' in product:
                    product['category'] = product['category'].lower()
                
                loaded_items.append(product)
    return loaded_items

try:
    products = load_products(',')
    if len(products) == 0:
        print("   ⚠️ S čárkou nic nenalezeno. Zkouším středník...")
        products = load_products(';')

    if len(products) == 0:
        print("\n❌ CHYBA: Nepodařilo se načíst žádné produkty.")
        print(f"   Očekávané sloupce v CSV (alespoň některé): {list(COLUMN_MAPPING.values())}")
        exit()

    print(f"✅ ÚSPĚCH! Načteno {len(products)} produktů.")

except FileNotFoundError:
    print(f"❌ Soubor {CSV_FILE} nebyl nalezen! Zkontroluj cestu.")
    exit()
except Exception as e:
    print(f"❌ Chyba při čtení CSV: {e}")
    exit()

# ============================================
# 4. UPLOAD (GENERUJEME VEKTORY)
# ============================================

print(f"\n🚀 Zahajuji upload přes Gemini (model gemini-embedding-001)...")

BATCH_SIZE = 50 
successful = 0

for i in range(0, len(products), BATCH_SIZE):
    batch = products[i:i + BATCH_SIZE]
    print(f"📦 Zpracovávám {i+1} až {min(i+len(batch), len(products))}...", end="")
    
    vectors = []
    for prod in batch:
        try:
            # 1. PŘÍPRAVA TEXTU PRO AI (EMBEDDING)
            cat_val = prod.get('category', 'obecné')
            
            text_to_embed = (
                f"Kategorie: {cat_val}. "
                f"Produkt: {prod['name']}. "
                f"Cena: {prod.get('price', '')}. "
                f"Popis: {prod.get('description', '')}. "
                f"Vlastnosti: {prod.get('sizes', '')} {prod.get('availability', '')}."
            )
            
            result = genai.embed_content(
                model="models/gemini-embedding-001",
                content=text_to_embed,
                task_type="retrieval_document"
            )
            
            metadata = {
                "name": prod.get('name', '')[:500],
                "price": prod.get('price', ''),
                "imageUrl": prod.get('imageUrl', ''), 
                "url": prod.get('url', ''),
                "description": prod.get('description', '')[:2000],
                "category": cat_val, 
                "sizes": prod.get('sizes', ''),
                "availability": prod.get('availability', ''),
            }
            
            metadata = {k: v for k, v in metadata.items() if v}

            vectors.append({
                "id": str(prod['id']),
                "values": result['embedding'],
                "metadata": metadata
            })
            
        except Exception as e:
            print(f"\n   ❌ Chyba u produktu {prod.get('name', 'Unknown')}: {e}")

    if vectors:
        try:
            index.upsert(vectors=vectors)
            successful += len(vectors)
            print(" ✅ Uloženo")
        except Exception as e:
            print(f"\n   ❌ Chyba Pinecone Upsert: {e}")
            
    time.sleep(0.5)

print("\n" + "="*70)
print(f"🎉 HOTOVO! V indexu '{INDEX_NAME}' je nyní {successful} produktů.")
print("="*70)
