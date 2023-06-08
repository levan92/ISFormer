import json 
from pathlib import Path 

COCOJSON = '/data/PublicDatasets/OCHuman/ochumanEX_coco_format_val_range_0.00_1.00.json'

def main():
    cocojson_path = Path(COCOJSON)

    with cocojson_path.open('r') as rf: 
        cocodict = json.load(rf)
    
    annots = []
    for annot in cocodict['annotations']: 
        if annot.get('ignore', False):
            continue
        annots.append(annot)
    
    print(f"# annots: {len(cocodict['annotations'])}->{len(annots)}")

    cocodict['annotations'] = annots

    out_path = cocojson_path.parent / f"{cocojson_path.stem}-removeignore.json"
    with out_path.open('w') as wf: 
        json.dump(cocodict, wf)

if __name__ == '__main__': 
    main()