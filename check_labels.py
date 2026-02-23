import sys, os
sys.path.append('src')
from data_preprocessing import load_and_process_data

d1 = load_and_process_data('dataset.csv')
d2 = load_and_process_data('dataset.csv')

m1 = d1.label_mapping
m2 = d2.label_mapping

print("Run 1 label mapping:", dict(list(m1.items())[:5]))
print("Run 2 label mapping:", dict(list(m2.items())[:5]))
print("Same mapping?", m1 == m2)
print("Total classes:", len(m1))
print("Psoriasis index R1:", m1.get("Psoriasis"))
print("Psoriasis index R2:", m2.get("Psoriasis"))
print("Dengue index R1:", m1.get("Dengue"))
print("Dengue index R2:", m2.get("Dengue"))
