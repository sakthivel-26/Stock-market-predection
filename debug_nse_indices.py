from app import nse_get_all_indices
rows = nse_get_all_indices()
print('rows type:', type(rows))
if rows:
    print('len', len(rows))
    for i, r in enumerate(rows[:10]):
        name = r.get('indexName') or r.get('index') or r.get('indexSymbol') or r.get('symbol')
        print(i, name)
        print(r)
else:
    print('no rows')
